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