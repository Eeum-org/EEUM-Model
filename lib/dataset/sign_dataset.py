import os
import json
import glob
import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torch import Tensor

class SignDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        ann_file: str,
        keypoint_dir: str,
        is_train: bool = False,
        tokenize: Optional[Callable] = None,
        lower: bool = False,
    ) -> None:
        ann_file_path = os.path.join(data_root, "annotations", ann_file)
        self.keypoint_prefix = os.path.join(data_root, keypoint_dir)
        self.is_train = is_train
        self.tokenize = tokenize
        self.lower = lower
        
        self.examples = self._load_examples_from_csv(ann_file_path)
        
    def _load_examples_from_csv(self, ann_file_path: str) -> List[dict]:
        annotations = pd.read_csv(ann_file_path, sep=",", encoding='utf-8', dtype='str')
        annotations = annotations[["Filename", "Kor"]]
        
        examples = []
        for _, row in annotations.iterrows():
            example = dict(row)
            glosses_str = example["Kor"]
            if pd.isna(glosses_str) or not isinstance(glosses_str, str):
                glosses_str = ""
            
            if self.tokenize:
                if self.lower:
                    glosses_str = glosses_str.lower()
                example["Kor"] = self.tokenize(glosses_str.strip())
            else:
                example["Kor"] = []
            examples.append(example)
        print(f"📊 {ann_file_path}에서 {len(examples)}개의 샘플 로드 완료.")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_and_process_keypoints(self, video_filename: str) -> np.ndarray:
        base_name = os.path.splitext(video_filename)[0]
        json_pattern = os.path.join(self.keypoint_prefix, f"{base_name}_*_keypoints.json")
        json_files = sorted(glob.glob(json_pattern))

        if not json_files:
            return np.array([], dtype=np.float32)

        all_processed_keypoints = []
        for json_file_path in json_files:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data.get('people'):
                person_data = data['people'][0]
                keypoints_data = np.concatenate([
                    np.array(person_data['pose_keypoints_2d']).reshape(-1, 3),
                    np.array(person_data['face_keypoints_2d']).reshape(-1, 3),
                    np.array(person_data['hand_left_keypoints_2d']).reshape(-1, 3),
                    np.array(person_data['hand_right_keypoints_2d']).reshape(-1, 3)
                ], axis=0)

                keypoints_with_conf = keypoints_data.copy()
                keypoints_xy = keypoints_with_conf[:, :2]

                nose_coords = keypoints_xy[0].copy()
                keypoints_relative = keypoints_xy - nose_coords
                keypoints_relative[keypoints_with_conf[:, 2] < 0.1] = 0

                left_shoulder, right_shoulder = keypoints_relative[5], keypoints_relative[2]
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                
                if shoulder_width > 1e-4:
                    keypoints_scaled = keypoints_relative / shoulder_width
                else:
                    keypoints_scaled = keypoints_relative

                all_processed_keypoints.append(keypoints_scaled.flatten())
            else:
                all_processed_keypoints.append(np.zeros(137 * 2, dtype=np.float32))
        
        processed_sequence = np.array(all_processed_keypoints, dtype=np.float32)
        return processed_sequence

    def _apply_keypoint_augmentation(self, keypoints: np.ndarray) -> np.ndarray:
        if self.is_train and np.random.random() < 0.5:
            noise = np.random.normal(loc=0, scale=0.05, size=keypoints.shape)
            keypoints += noise.astype(np.float32)
        return keypoints

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        assert hasattr(self, "vocab"), "어휘 사전(vocab)이 로드되지 않았습니다."
        
        example = self.examples[index]
        keypoints_sequence = self._load_and_process_keypoints(example["Filename"])
        
        if keypoints_sequence.size == 0:
            keypoints_sequence = np.zeros((10, 137 * 2), dtype=np.float32)

        keypoints_sequence = self._apply_keypoint_augmentation(keypoints_sequence)

        tokens = example["Kor"]
        gloss_indices = np.array([self.vocab.stoi.get(token, self.vocab.stoi[self.vocab.unk_token]) for token in tokens], dtype=np.int64)
        
        return torch.from_numpy(keypoints_sequence), torch.from_numpy(gloss_indices)
    
    def load_vocab(self, vocabulary):
        self.vocab = vocabulary
        self.pad_idx = self.vocab.stoi[self.vocab.pad_token]

    def collate(self, data: List[Tuple[Tensor, Tensor]]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        keypoint_sequences, gloss_sequences = zip(*data)
        keypoint_lengths = torch.tensor([len(seq) for seq in keypoint_sequences], dtype=torch.long)
        padded_keypoints = torch.nn.utils.rnn.pad_sequence(keypoint_sequences, batch_first=True, padding_value=0.0)
        gloss_lengths = torch.tensor([len(seq) for seq in gloss_sequences], dtype=torch.long)
        padded_glosses = torch.nn.utils.rnn.pad_sequence(gloss_sequences, batch_first=True, padding_value=self.pad_idx)
        return (padded_keypoints, keypoint_lengths), (padded_glosses, gloss_lengths)