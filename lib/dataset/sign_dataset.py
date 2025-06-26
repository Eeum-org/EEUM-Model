import cv2
import numpy as np
import pandas as pd
import os
from .transforms import apply_transform_gens
from typing import List, Optional, Callable, NoReturn, Tuple
from torch.utils.data import Dataset
import torch
from torch import Tensor
import torchvision.io as io
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps
from pytorchvideo.data.encoded_video import EncodedVideo
# from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample

import torchvision.transforms.v2 as T

import torch
# import torchvision.transforms as T

class SignDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        *,
        img_prefix: Optional[str] = None,
        tfm_gens: Optional[list] = None,
        tokenize: Optional[Callable] = None,
        lower: bool = False,
        is_train=False,
        exclude_token=None,
        max_frames: int = 120,  # 최대 프레임 수 제한
        target_fps: Optional[int] = None  # 목표 FPS (다운샘플링용)
    ) -> None:
        ann_file = os.path.join(data_root, "train" if is_train else "val", "annotations", ann_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tfm_gens = tfm_gens
        self.tokenize = tokenize
        self.lower = lower
        self.exclude_token = exclude_token
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.gpu_transforms = torch.nn.Sequential(
            T.ToImage(),  # numpy → tensor 변환
            T.ToDtype(torch.uint8, scale=True),  # uint8로 변환
            T.Resize((224, 224), antialias=True),  # GPU에서 resize
            T.ToDtype(torch.float32, scale=True),  # float32로 변환
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        self.video_prefix = os.path.join(data_root, "train" if is_train else "val", img_prefix.replace("_video", ""), img_prefix) if img_prefix else data_root
        self.examples = self.load_examples_from_csv(ann_file)
        self.is_train = is_train
        if torch.cuda.is_available():
            self.gpu_transforms = self.gpu_transforms.to(self.device)

    def apply_gpu_transforms(self, frames):
        """GPU에서 transform 처리"""
        # numpy → tensor 변환 (B, H, W, C) → (B, C, H, W)
        if isinstance(frames, np.ndarray):
            frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        else:
            frames_tensor = frames
        
        # GPU로 이동
        frames_tensor = frames_tensor.to(self.device)
        
        # GPU에서 transform 적용
        transformed = self.gpu_transforms(frames_tensor)
        
        return transformed #.cpu() #.numpy() #.permute(0, 2, 3, 1).numpy()

    def __len__(self):
        return len(self.examples)
    
    def load_video_frames_torchcodec(self, video_path: str):
        """TorchCodec GPU 디코딩"""
        try:
            # GPU 디코더 생성
            torch.cuda.set_device("cuda:0")
            torch.cuda.empty_cache()
            decoder = VideoDecoder(video_path, device="cuda")
            
            total_frames = len(decoder)
            if total_frames > self.max_frames:
                indices = torch.linspace(0, total_frames-1, self.max_frames).long()
                frames = decoder.get_frames_at(indices=indices)
            else:
                frames = decoder[:]
            
            # 이미 GPU 텐서
            return frames
            
        except Exception as e:
            # cuda 사용 불가시 CPU로 fallback
            print(f"TorchCodec 실패: {e}")
            return self.load_video_frames(video_path)

    def load_video_frames_gpu(self, video_path: str) -> np.ndarray:
        """GPU에서 비디오 디코딩"""
        try:
            # ✅ torchvision.io는 GPU 가속 지원
            video, _, info = io.read_video(
                video_path, 
                pts_unit='sec',
                backend='pyav'  # GPU 가속 백엔드
            )
            
            if video.shape[0] > self.max_frames:
                indices = torch.linspace(0, video.shape[0]-1, self.max_frames).long()
                video = video[indices]
            
            return video.numpy()
            
        except Exception as e:
            # fallback to CPU
            return self.load_video_frames(video_path)

    def load_video_frames(self, video_path: str) -> np.ndarray:
        """영상 파일에서 프레임 추출"""
        if not os.path.exists(video_path):
            print(f"❌ 영상 파일 없음: {video_path}")
            return np.array([])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 영상 파일 열기 실패: {video_path}")
            return np.array([])
        
        # 영상 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_indices = self._get_frame_indices(total_frames, fps)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
            else:
                print(f"⚠️ 프레임 {frame_idx} 읽기 실패")
                break
        
        cap.release()
        
        if frames:
            return np.stack(frames, axis=0)
        else:
            print(f"⚠️ 유효한 프레임 없음: {video_path}")
            return np.array([])
    
    def _get_frame_indices(self, total_frames: int, fps: float) -> List[int]:
        """프레임 인덱스 계산 (샘플링 적용)"""
        if self.target_fps and self.target_fps < fps:
            # FPS 다운샘플링
            step = int(fps / self.target_fps)
            indices = list(range(0, total_frames, step))
        else:
            # 모든 프레임 사용
            indices = list(range(total_frames))
        
        # 최대 프레임 수 제한
        if len(indices) > self.max_frames:
            # 균등 간격으로 샘플링
            indices = np.linspace(0, len(indices)-1, self.max_frames, dtype=int)
            indices = [indices[i] for i in range(len(indices))]
        
        return indices
    
    def __getitem__(self, i):
        assert hasattr(self, "vocab")
        example = self.examples[i]
        
        # 영상 파일 경로
        video_filename = example["Filename"]
        video_path = os.path.join(self.video_prefix, video_filename)
        # directions = ['_F', '_U', '_D', '_L', '_R']
        # vid_base_name = example["Filename"].replace(".mp4", "")
        # for direction in directions:
        #     # 영상 파일 존재 확인
        #     video_path = os.path.join(self.video_prefix, vid_base_name + direction + ".mp4")
        #     if os.path.exists(video_path):
        # #         # 영상에서 프레임 추출
        #         frames = self.load_video_frames(video_path)
        #         # frames = self.load_video_frames_pytorchvideo(video_path)
        #         if frames.size == 0:
        #             # 빈 프레임인 경우 더미 데이터 생성
        #             print(f"⚠️ 더미 프레임 사용: {video_filename}")
        #             frames = np.zeros((10, 224, 224, 3), dtype=np.uint8)
        
        #         # frames = self.load_video_frames_pytorchvideo(video_path)

        frames = self.load_video_frames_torchcodec(video_path)
        if frames.size == 0:
            # 빈 프레임인 경우 더미 데이터 생성
            print(f"⚠️ 더미 프레임 사용: {video_filename}")
            frames = np.zeros((10, 224, 224, 3), dtype = np.uint8)

        # 데이터 증강 (훈련 시에만)
        if self.is_train and len(frames) > 5:
            frames = self._apply_temporal_augmentation(frames)
        
        # 변환 적용
        frames = self.apply_gpu_transforms(frames)
        # if self.tfm_gens is not None:
            # frames, _ = apply_transform_gens(self.tfm_gens, frames)
        
        # 토큰 변환
        tokens = example["Kor"]
        indices = [self.vocab.stoi[token] for token in tokens]
        
        return frames, torch.tensor(indices, device = self.device) #indices
    
    def _apply_temporal_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """시간축 데이터 증강"""
        num_frames = frames.shape[0]
        
        if self.is_train:
            # ✅ 랜덤 프레임 중복 (PyTorch 연산)
            if torch.rand(1).item() < 0.3:
                num_dup = int(num_frames * 0.1)
                dup_indices = torch.randperm(num_frames, device=frames.device)[:num_dup]
                dup_frames = frames[dup_indices]
                
                # GPU에서 직접 결합
                frames = torch.cat([frames, dup_frames], dim=0)
            
            # ✅ 랜덤 프레임 삭제 (PyTorch 연산)
            if torch.rand(1).item() < 0.3 and frames.shape[0] > 10:
                num_keep = int(frames.shape[0] * 0.9)
                keep_indices = torch.randperm(frames.shape[0], device=frames.device)[:num_keep]
                keep_indices = torch.sort(keep_indices)[0]  # 순서 유지
                frames = frames[keep_indices]
        
        """ num_frames = len(frames)
        
        if self.is_train:
            # 랜덤 프레임 중복
            if np.random.random() < 0.3:
                dup_indices = np.random.choice(num_frames, int(num_frames * 0.1), replace=False)
                frames = np.insert(frames, dup_indices, frames[dup_indices], axis=0)
            
            # 랜덤 프레임 삭제
            if np.random.random() < 0.3 and len(frames) > 10:
                drop_indices = np.random.choice(len(frames), int(len(frames) * 0.1), replace=False)
                frames = np.delete(frames, drop_indices, axis=0) """
        
        return frames

    def load_examples_from_csv(self, ann_file: str) -> List[dict]:
        annotations = pd.read_csv(ann_file, sep=",", encoding='utf-8')
        annotations = annotations[["Filename", "Kor"]]
        directions = ['_F', '_U', '_D', '_L', '_R']
        examples = []
        for i in range(len(annotations)):
            example = dict(annotations.iloc[i])
            vid_base_name = example["Filename"].replace(".mp4", "")
            for direction in directions:
                # 영상 파일 존재 확인
                video_path = os.path.join(self.video_prefix, vid_base_name + direction + ".mp4")
                if not os.path.exists(video_path):
                    print(f"⚠️ 영상 파일 없음: {example['Filename']}")
                    continue

                # 토큰화
                glosses_str = example["Kor"]
                if self.tokenize is not None and isinstance(glosses_str, str):
                    if self.lower:
                        glosses_str = glosses_str.lower()
                    tokens = self.tokenize(glosses_str.rstrip("\n"))
                    example["Kor"] = tokens
                    example["Filename"] = video_path
                examples.append(example)
        
        print(f"📊 로드된 영상: {len(examples)}개")
        return examples
    
    @property
    def gloss(self):
        return [example["Kor"] for example in self.examples]

    def load_vocab(self, vocabulary):
        self.vocab = vocabulary
        self.pad_idx = self.vocab.stoi[self.vocab.pad_token]
        self.sil_idx = self.vocab.stoi[self.vocab.sil_token]

    def collate(self, data):
        videos, glosses = list(zip(*data))

        def pad(videos: List[Tensor], glosses: List[int]
                ) -> Tuple[Tuple[List[Tensor], List[int]], Tuple[List[int], List[int]]]:
            video_lengths = [len(v) for v in videos]
            max_video_len = max(video_lengths)
            padded_videos = []
            for video, length in zip(videos, video_lengths):
                C, H, W = video.size(1), video.size(2), video.size(3)
                new_tensor = video.new(max_video_len, C, H, W).fill_(1e-8)
                new_tensor[:length] = video
                padded_videos.append(new_tensor)

            gloss_lengths = [len(s) for s in glosses]
            max_len = max(gloss_lengths)
            glosses = [
                s + [self.pad_idx] * (max_len - len(s)) if len(s) < max_len else s for s in glosses
            ]
            return (padded_videos, video_lengths), (glosses, gloss_lengths)

        (videos, video_lengths), (glosses, gloss_lengths) = pad(videos, glosses)
        video_lengths = torch.tensor(video_lengths, device=self.device, dtype=torch.long)
        glosses = torch.tensor(glosses, device=self.device, dtype=torch.long)
        gloss_lengths = torch.tensor(gloss_lengths, device=self.device, dtype=torch.long)
        videos = torch.stack(videos, dim=0)
        # video_lengths = Tensor(video_lengths).long()
        # glosses = Tensor(glosses).long()
        # gloss_lengths = Tensor(gloss_lengths).long()
        return (videos, video_lengths), (glosses, gloss_lengths)