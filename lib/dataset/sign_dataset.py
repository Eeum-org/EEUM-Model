import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple
from math import ceil
import torch
from torch.utils.data import Dataset
from torch import Tensor
from .vocabulary import build_vocab
class SegmentSignDataset(Dataset):
    """시간 구간별 수화 데이터셋"""
    
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        keypoint_prefix: str,
        is_train: bool = False,
        fps: float = 30.0,
        max_seq_length: int = 500,
        min_seq_length: int = 5,
        vocab = None
    ):
        self.data_root = data_root
        self.keypoint_dir = os.path.join(data_root, keypoint_prefix)
        self.is_train = is_train
        self.fps = fps
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.pad_idx = 0
        print(data_root, ann_file )
        # 시간 구간별 어노테이션 로드
        self.segments, tokens = self._load_segment_annotations(
            os.path.join(data_root, ann_file)
        )
        self.vocab = vocab if vocab else {}
        if not self.vocab:
            self.vocab = build_vocab(tokens)
        print(f"📊 {'Train' if is_train else 'Val'}: {len(self.segments)}개 세그먼트 로드")
        
    def _load_segment_annotations(self, ann_file_path: str) -> List[dict]:
        """시간 구간별 어노테이션 로드"""
        print(f"get_cwd : {os.getcwd()}, path : {ann_file_path}")
        # if not os.path.exists(ann_file_path):
        #     raise FileNotFoundError(f"어노테이션 파일을 찾을 수 없습니다: {ann_file_path}")
            
        annotations = pd.read_csv(ann_file_path, encoding='utf-8')
        tokens = annotations["Kor"]
        segments = []
        for _, row in annotations.iterrows():
            segment = {
                'Filename': row['Filename'],
                'start_time': float(row['start_time']),  # 초 단위
                'end_time': float(row['end_time']),     # 초 단위
                'Kor': str(row['Kor']),
                'tokens': str(row['Kor']).split()
            }
            segments.append(segment)
            
        return segments, tokens
    
    def _load_keypoints_for_segment(self, video_filename: str, start_time: float, end_time: float) -> np.ndarray:
        """특정 시간 구간의 키포인트만 추출"""
        video_filename = video_filename.split(".")[0]

        # 시작/종료 프레임 계산
        start_frame = max(0, int(start_time * self.fps))
        end_frame = ceil(end_time * self.fps)
        
        segment_keypoints = []
        
        for frame_idx in range(start_frame, end_frame + 1):
            json_file = os.path.join(
                self.keypoint_dir, video_filename, f"{video_filename}_{frame_idx:012d}_keypoints.json"
            )
            # print(json_file, os.path.exists(json_file))
            if os.path.exists(json_file):
                keypoints = self._load_single_frame_keypoints(json_file)
                segment_keypoints.append(keypoints)
            else:
                # 프레임이 없으면 이전 프레임 복사 또는 제로 패딩
                if segment_keypoints:
                    segment_keypoints.append(segment_keypoints[-1].copy())
                else:
                    segment_keypoints.append(np.zeros(274, dtype=np.float32))
        
        # 최소/최대 길이 제한
        if len(segment_keypoints) < self.min_seq_length:
            # 패딩으로 최소 길이 보장
            while len(segment_keypoints) < self.min_seq_length:
                if segment_keypoints:
                    segment_keypoints.append(segment_keypoints[-1].copy())
                else:
                    segment_keypoints.append(np.zeros(274, dtype=np.float32))
        elif len(segment_keypoints) > self.max_seq_length:
            # 균등 샘플링으로 길이 제한
            indices = np.linspace(0, len(segment_keypoints)-1, self.max_seq_length, dtype=int)
            segment_keypoints = [segment_keypoints[i] for i in indices]
            
        return np.array(segment_keypoints, dtype=np.float32)
    
    def _load_single_frame_keypoints(self, json_file: str) -> np.ndarray:
        """단일 프레임 키포인트 로드 및 정규화"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return np.zeros(274, dtype=np.float32)
        
        if not data.get('people'):
            return np.zeros(274, dtype=np.float32)

        if type(data['people']) == list:
            person_data = data['people'][0]
        else:
            person_data = data['people']
        
        # OpenPose 키포인트 추출
        pose_kp = np.array(person_data.get('pose_keypoints_2d', [])).reshape(-1, 3)
        face_kp = np.array(person_data.get('face_keypoints_2d', [])).reshape(-1, 3)
        left_hand_kp = np.array(person_data.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
        right_hand_kp = np.array(person_data.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
        
        # 키포인트 개수 맞추기
        pose_kp = pose_kp[:25, :2] if len(pose_kp) >= 25 else np.zeros((25, 2))
        face_kp = face_kp[:70, :2] if len(face_kp) >= 70 else np.zeros((70, 2))
        left_hand_kp = left_hand_kp[:21, :2] if len(left_hand_kp) >= 21 else np.zeros((21, 2))
        right_hand_kp = right_hand_kp[:21, :2] if len(right_hand_kp) >= 21 else np.zeros((21, 2))
        
        # 정규화 개선
        keypoints = np.concatenate([pose_kp, face_kp, left_hand_kp, right_hand_kp], axis=0)
        
        # 코 좌표 기준 상대 좌표 변환
        if pose_kp[0, 0] != 0 and pose_kp[0, 1] != 0:  # 코가 감지된 경우
            nose_coords = pose_kp[0]
            keypoints = keypoints - nose_coords
            
            # 어깨 너비로 스케일링
            if len(pose_kp) > 5:
                left_shoulder, right_shoulder = pose_kp[5], pose_kp[2]
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                if shoulder_width > 1e-4:
                    keypoints = keypoints / shoulder_width
        
        # NaN 처리
        keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return keypoints.flatten()  # (137*2,) = (274,)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        segment = self.segments[index]
        
        # 해당 시간 구간의 키포인트 로드
        keypoints = self._load_keypoints_for_segment(
            segment['Filename'],
            segment['start_time'],
            segment['end_time']
        )
        # print(f"keypoint : {keypoints}")
        # 데이터 증강 (훈련 시에만)
        if self.is_train:
            keypoints = self._apply_augmentation(keypoints)
        
        # 한국어 토큰을 인덱스로 변환
        tokens = segment['tokens']
        if not tokens:
            token_indices = np.array([0], dtype=np.int64)  # blank 토큰
        else:
            token_indices = np.array([
                self.vocab.stoi[token] for token in tokens
            ], dtype=np.int64)
        
        return torch.from_numpy(keypoints), torch.from_numpy(token_indices)
    
    def _apply_augmentation(self, keypoints: np.ndarray) -> np.ndarray:
        """데이터 증강"""
        if np.random.random() < 0.3:
            # 시간 축 노이즈
            noise = np.random.normal(0, 0.02, keypoints.shape)
            keypoints = keypoints + noise.astype(np.float32)
        
        if np.random.random() < 0.2:
            # 시간 축 스케일링
            scale = np.random.uniform(0.9, 1.1)
            seq_len = len(keypoints)
            new_len = max(self.min_seq_length, int(seq_len * scale))
            if new_len != seq_len:
                indices = np.linspace(0, seq_len-1, new_len).astype(int)
                keypoints = keypoints[indices]
        
        return keypoints
    
    def load_vocab(self, vocabulary):
        """어휘 사전 로드"""
        self.vocab = vocabulary
        self.pad_idx = getattr(vocabulary, 'pad_idx', 0)
    
    def collate(self, data: List[Tuple[Tensor, Tensor]]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """배치 콜레이트 함수"""
        keypoint_sequences, token_sequences = zip(*data)
        
        # 시퀀스 길이 계산
        keypoint_lengths = torch.tensor([len(seq) for seq in keypoint_sequences], dtype=torch.long)
        token_lengths = torch.tensor([len(seq) for seq in token_sequences], dtype=torch.long)
        
        # 패딩
        padded_keypoints = torch.nn.utils.rnn.pad_sequence(
            keypoint_sequences, batch_first=True, padding_value=0.0
        )
        padded_tokens    = torch.nn.utils.rnn.pad_sequence(
            token_sequences, batch_first=True, padding_value=self.pad_idx
        )
                # Teacher forcing을 위한 입력/타겟 생성
        teacher_input = []  # <sos> + gloss[:-1]
        teacher_target = []  # gloss + <eos>
        
        for gloss in padded_tokens:
            # 입력: <sos> + 원본 시퀀스의 앞부분
            sos_token = torch.tensor([4], device=gloss.device)  # <sos> = 1
            input_seq = torch.cat([sos_token, gloss[:-1]])
            teacher_input.append(input_seq)
            
            # 타겟: 원본 시퀀스 + <eos>
            eos_token = torch.tensor([5], device=gloss.device)  # <eos> = 2
            target_seq = torch.cat([gloss, eos_token])
            teacher_target.append(target_seq)
        
        teacher_input = torch.stack(teacher_input)
        teacher_target = torch.stack(teacher_target)
        
        return (padded_keypoints, torch.tensor(keypoint_lengths)), (teacher_input, teacher_target, torch.tensor(token_lengths))
        # return (padded_keypoints, keypoint_lengths), (padded_tokens, token_lengths)

class SignDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        keypoint_dir: str,
        is_train: bool = False,
        tokenize=None,
        lower: bool = False,
        keypoint_type: str = "openpose",
        min_seq_length: int = 5,
        max_seq_length: int = 500,
    ) -> None:
        ann_path = os.path.join(data_root, "annotations", ann_file)
        self.keypoint_prefix = os.path.join(data_root, keypoint_dir.replace("_keypoint", ""), keypoint_dir)
        self.is_train = is_train
        self.tokenize = tokenize
        self.lower = lower
        self.keypoint_type = keypoint_type
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.keypoint_dims = {"openpose": 274, "mediapipe": 258}
        self.pad_idx = 2
        print(f"📊 Annotation 파일: {ann_path}")
        self.examples = self._load_examples_from_csv(ann_path)

    def _load_examples_from_csv(self, ann_file_path: str) -> List[dict]:
        try:
            df = pd.read_csv(ann_file_path, sep=",", encoding="euc-kr", dtype=str)
        except UnicodeDecodeError:
            df = pd.read_csv(ann_file_path, sep=",", encoding="utf-8", dtype=str)
        df = df[["Filename", "Kor"]]
        examples = []
        for _, row in df.iterrows():
            gloss = row["Kor"] if isinstance(row["Kor"], str) else ""
            if self.tokenize:
                gloss = gloss.lower().strip() if self.lower else gloss.strip()
                tokens = self.tokenize(gloss) or ["<UNK>"]
            else:
                tokens = ["<UNK>"]
            examples.append({"Filename": row["Filename"], "Kor": tokens})
        print(f"📊 로드된 샘플 수: {len(examples)}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_and_process_keypoints(self, video_filename: str) -> np.ndarray:
        subdir = os.path.join(self.keypoint_prefix, video_filename)
        json_files = []
        views = ["_F", "_L", "_R", "_U", "_D"]
        for view in views:
            view_folder = subdir + view
            if os.path.isdir(view_folder):
                for fname in sorted(os.listdir(view_folder)):
                    if fname.startswith(f"{video_filename}_") and fname.endswith("_keypoints.json"):
                        json_files.append(os.path.join(view_folder, fname))
                # print(f"🔍 '{subdir}'에서 발견된 파일 수: {len(json_files)}")
            else:
                print(f"❌ 서브디렉토리 미존재: {subdir}")

        if not json_files:
            return np.array([], dtype=np.float32)

        all_kps = []
        for path in json_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self.keypoint_type == "openpose":
                    kpt = self._process_openpose(data)
                else:
                    kpt = np.zeros(self.keypoint_dims[self.keypoint_type], dtype=np.float32)
            except Exception as e:
                print(f"⚠️ 처리 오류 {path}: {e}")
                kpt = np.zeros(self.keypoint_dims[self.keypoint_type], dtype=np.float32)
            all_kps.append(kpt)

        seq = np.array(all_kps, dtype=np.float32)
        L = len(seq)
        if L > self.max_seq_length:
            seq = seq[: self.max_seq_length]
        elif L < self.min_seq_length:
            pad = np.zeros((self.min_seq_length - L, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        return seq
    
    def _process_openpose(self, data: dict) -> np.ndarray:
        person = data.get("people") or []
        if not person:
            return np.zeros(self.keypoint_dims["openpose"], dtype=np.float32)

        pts = np.concatenate([
            np.array(person.get("pose_keypoints_2d", [])).reshape(-1, 3),
            np.array(person.get("face_keypoints_2d", [])).reshape(-1, 3),
            np.array(person.get("hand_left_keypoints_2d", [])).reshape(-1, 3),
            np.array(person.get("hand_right_keypoints_2d", [])).reshape(-1, 3)
        ], axis=0)
        xy = pts[:, :2]
        conf = pts[:, 2]

        # 정규화 변경
        if pts.shape[0] >= 6:
            # 유효한 키포인트만 사용
            valid_mask = conf > 0.3  # confidence 임계값 증가
            if valid_mask.sum() > 5:  # 최소 5개 이상의 유효한 키포인트
                valid_xy = xy[valid_mask]
                
                # 중심점 계산 (목 또는 유효한 키포인트들의 중심)
                if conf[1] > 0.3:  # 목 키포인트 사용
                    center = xy[1]
                else:
                    center = valid_xy.mean(axis=0)
                
                # 중심 기준 상대 좌표
                rel = xy - center
                rel[~valid_mask] = 0
                
                # 스케일 정규화 - robust
                distances = np.linalg.norm(rel[valid_mask], axis=1)
                if distances.size > 0:
                    scale = np.percentile(distances, 95) + 1e-8  # 95% 백분위수 사용
                    rel = rel / scale
                    # 극값 클리핑
                    rel = np.clip(rel, -3.0, 3.0)
                else:
                    rel = xy * 0.001  # 매우 작은 값으로 정규화
            else:
                rel = xy * 0.001
        else:
            rel = xy * 0.001
        
        rel = np.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=-1.0)
        return rel.flatten()
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        ex = self.examples[idx]
        
        # 키포인트 시퀀스 로드
        kps_np = self._load_and_process_keypoints(ex["Filename"])
        if kps_np.size == 0:
            kps_np = np.zeros((self.min_seq_length, self.keypoint_dims[self.keypoint_type]), dtype=np.float32)
        
        keypoint_len = kps_np.shape[0]
        
        # 토큰 인덱스 변환 (안전한 방식)
        tokens = ex["Kor"]
        
        gloss_idx = []
        for token in tokens:
            if hasattr(self, 'vocab') and hasattr(self.vocab, 'stoi'):
                if token in self.vocab.stoi:
                    gloss_idx.append(self.vocab.stoi[token])
                elif hasattr(self.vocab, 'unk_token') and self.vocab.unk_token in self.vocab.stoi:
                    gloss_idx.append(self.vocab.stoi[self.vocab.unk_token])
                else:
                    gloss_idx.append(1)  # 기본 UNK 인덱스
            else:
                gloss_idx.append(1)  # vocab이 없는 경우 기본값
        
        gloss_len = len(gloss_idx)
        
        # 텐서 변환
        kps = torch.from_numpy(kps_np)
        gloss = torch.tensor(gloss_idx, dtype=torch.long)
        
        return (kps, keypoint_len), (gloss, gloss_len)
    
    def load_vocab(self, vocab):
        self.vocab = vocab
        self.pad_idx = vocab.stoi.get(vocab.pad_token, 0)

    def collate_fn(self, batch):
        """배치 콜레이트 함수 - 인스턴스 메서드로 정의"""
        kps_list, gloss_list = [], []
        kps_lens, gloss_lens = [], []
        for (kps, kl), (gloss, gl) in batch:
            kps_list.append(kps)
            kps_lens.append(kl)
            gloss_list.append(gloss)
            gloss_lens.append(gl)
        # 패딩 처리
        padded_kps = torch.nn.utils.rnn.pad_sequence(kps_list, batch_first=True, padding_value=0.0)
        padded_gloss = torch.nn.utils.rnn.pad_sequence(gloss_list, batch_first=True, padding_value=self.pad_idx)

        return (padded_kps, torch.tensor(kps_lens)), (padded_gloss, torch.tensor(gloss_lens))