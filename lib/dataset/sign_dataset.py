import os
import cv2
import csv
import glob
import torch
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from .preprocessing import load_available_views, load_json, pad_sequence, extract_kp
from .transforms import apply_transform_gens
from typing import Callable, List, NoReturn, Optional, Tuple
from lib.config.settings import DATA_DIR, VOCAB_SPECIAL_TOKENS, MAX_SEQ_LEN
from torchvision import transforms
from .generate_annotations import _find_video_folder

class SignDataset(Dataset):
    def __init__(self, split='train', vocab=None):
        """
        데이터셋 초기화. annotations/[video_folder]_annotations.csv가 없으면 자동 생성.
        """
        self.split = split
        self.data_dir = os.path.join(DATA_DIR, self.split)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # EfficientNet-B0 입력 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        ])
        # 비디오 폴더명 기반으로 참조할 CSV 경로 지정
        video_folders = _find_video_folder(self.data_dir)

        if not video_folders:
            raise FileNotFoundError(f"No video folder found in '{self.data_dir}' to determine annotation file.")
    
        self.annotations_dir = os.path.join(self.data_dir, 'annotations')
        for folder in video_folders:
            self.base_name = os.path.split(folder)[1].replace('_video', '')
            csv_filename = f"{self.base_name}_annotations.csv"
            self.csv_path = os.path.join(self.annotations_dir, csv_filename)
            # 어노테이션 CSV 파일이 없으면 자동 생성
            if not os.path.exists(self.csv_path):
                print(f"Annotation file '{self.csv_path}' not found. Generating it automatically...")
                self._generate_annotations()

        self.items = self._load_annotations()
        self.vocab = vocab if vocab is not None else {}
        if not self.vocab:
            self._build_vocab()
        self.vocab_itos = {}
        for k, v in self.vocab.items():
            self.vocab_itos[v] = k
    def _generate_annotations(self):
        """
        generate_annotations.py 모듈을 호출하여 CSV 파일 생성
        """
        from .generate_annotations import generate_annotations_csv
        generate_annotations_csv(self.split)
        
        # 생성 후에도 파일이 없는 경우 오류 발생
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Failed to generate annotation file at '{self.csv_path}'")

    def _load_annotations(self):
        annotations = []
        for annotation_file in os.listdir(self.annotations_dir):
            with open(os.path.join(self.annotations_dir, annotation_file), 'r', encoding='utf-8') as f:
                annotations.extend(csv.DictReader(f))
        return annotations
    
    def _build_vocab(self):
        self.vocab.update(VOCAB_SPECIAL_TOKENS)
        idx = len(self.vocab)
        for row in self.items:
            for token in row['meaning'].split():
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
        print(f"Vocabulary built for '{self.split}' split with {len(self.vocab)} words.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        base_folder, base_name = os.path.split(row['file_name'])
        label_str = row['meaning']

        # 사용 가능한 뷰(F,U,D,L,R)의 키포인트만 로드
        keypoint_base_path = os.path.join(self.data_dir, base_folder, base_folder + "_keypoint")
        available_views = load_available_views(keypoint_base_path, base_name)
        
        kp_seq_list = []
        for view in available_views:
            # 실제 데이터 구조에 맞게 수정: 폴더/파일 구조 고려
            view_folder_path = os.path.join(keypoint_base_path, f"{base_name}_{view}")
            
            if os.path.exists(view_folder_path) and os.path.isdir(view_folder_path):
                # 폴더 내 keypoints JSON 파일들 로드
                json_files = [f for f in os.listdir(view_folder_path) if f.endswith('_keypoints.json')]
                json_files.sort()  # 파일명 순서로 정렬
                
                frames = []
                for json_file in json_files:
                    json_path = os.path.join(view_folder_path, json_file)
                    json_data = load_json(json_path)

                    if json_data and 'people' in json_data and json_data['people']:
                        person_data = json_data['people']
                        frame_kp = extract_kp(person_data)
                        frames.append(frame_kp)
                
                if frames:
                    kp_seq_list.append(torch.stack(frames))


        if not kp_seq_list:
            keypoints = torch.zeros(MAX_SEQ_LEN, 1, 75 * 3) # (T, V, D)
        else:
            max_len = max(kp.shape[0] for kp in kp_seq_list)
            padded_views = [pad_sequence(kp, max_len) for kp in kp_seq_list]
            keypoints = torch.stack(padded_views, dim=1) # (T, V, D)
        # if self.transform:
        #     keypoints = self.transform(keypoints)
        keypoints_padded = pad_sequence(keypoints, MAX_SEQ_LEN)
        
        # 빈 문자열 키 대신 '<unk>' 사용
        labels = [self.vocab.get(token, self.vocab['<unk>']) for token in label_str.split()]
        target = torch.tensor(labels, dtype=torch.long)
        return keypoints_padded, target

class SignDataset_old(Dataset):

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
        exclude_token=None
    ) -> NoReturn:
        ann_file = os.path.join(data_root, ann_file)

        # option for videos
        self.tfm_gens = tfm_gens

        # option for tokenization
        self.tokenize = tokenize
        self.lower = lower
        self.exclude_token = exclude_token

        self.img_prefix = os.path.join(data_root, img_prefix)
        self.examples = self.load_examples_from_csv(ann_file)
        self.is_train = is_train

    def __getitem__(self, i):
        assert hasattr(self, "vocab")
        example = self.examples[i]

        # read video -> processing
        frames_path = example["frames"]

        # ramdom duplicate and drop
        frames_inds = np.array([i for i in range(len(frames_path))]).astype(np.int)
        if self.is_train:
            rand_inds = np.random.choice(
                len(frames_path), int(len(frames_path) * 0.2), replace=False
            )

            # random frame insertion
            total_inds = np.concatenate([frames_inds, rand_inds], 0)
            total_inds = np.sort(total_inds)

            # random frame dropping
            rand_inds = np.random.choice(len(total_inds), int(len(total_inds) * 0.2), replace=False)
            selected = np.delete(total_inds, rand_inds)
        else:
            selected = frames_inds
        # frames = np.stack([cv2.imread(f_path, cv2.IMREAD_COLOR) for f_path in frames_path], axis=0)  # noqa

        # read selected images
        
        try:
            frames = np.stack([cv2.imread(frames_path[i], cv2.IMREAD_COLOR) for i in selected], axis=0)
        except ValueError:
            print(example)
            #pdb.set_trace()
        if self.tfm_gens is not None:
            frames, _ = apply_transform_gens(self.tfm_gens, frames)

        # gloss -> CTC supervision signal
        tokens = example["Kor"]
        indices = [self.vocab.stoi[token] for token in tokens]
        return frames, indices

    def __len__(self):
        return len(self.examples)

    def load_examples_from_csv(self, ann_file: str) -> List[dict]:
        annotations = pd.read_csv(ann_file, sep=",",encoding='euc-kr')
        annotations = annotations[["Filename", "Kor"]]

        examples = []
        for i in range(len(annotations)):
            example = dict(annotations.iloc[i])
            # glob all image locations
            frames_path = glob.glob(os.path.join(self.img_prefix, example["Filename"],"*.jpg"))
            frames_path.sort()
            example["frames"] = frames_path

            # tokenization
            glosses_str = example["Kor"]
            if self.tokenize is not None and isinstance(glosses_str, str):
                if self.lower:
                    glosses_str = glosses_str.lower()
                tokens = self.tokenize(glosses_str.rstrip("\n"))
                example["Kor"] = tokens
                '''
                example["Kor"] = [
                    token for token in tokens
                    # exclude some tokens in annotations, i.e., ["__ON__", "__OFF__"].
                    if (self.exclude_token is not None and token not in self.exclude_token)
                ]
                '''
            examples.append(example)

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
        videos = torch.stack(videos, dim=0)
        video_lengths = Tensor(video_lengths).long()
        glosses = Tensor(glosses).long()
        gloss_lengths = Tensor(gloss_lengths).long()
        return (videos, video_lengths), (glosses, gloss_lengths)
