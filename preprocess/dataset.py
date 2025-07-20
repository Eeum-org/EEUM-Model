import os
import json
import torch
import numpy as np
import math
import pickle
import hashlib
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random


class SignDataset(Dataset):
    def __init__(self, samples, vocab, config=None, dataset_size=None, cache_dir=None):
        """
        미리 스캔된 샘플 리스트 기반
        """
        self.samples = samples
        self.vocab = vocab
        self.config = config or {}
        self.dataset_size = dataset_size
        self.cache_dir = cache_dir
        
        # 전처리 설정
        self.hand_scale = self.config.get('HAND_SCALE', 1.5)
        self.normalize_center = self.config.get('NORMALIZE_CENTER', True)
        self.shoulder_width_scale = self.config.get('SHOULDER_WIDTH_SCALE', True)
        self.hand_emphasis = self.config.get('HAND_EMPHASIS', True)
        
        # 데이터셋 크기 조정
        if self.dataset_size and self.dataset_size < len(self.samples):
            self.samples = self._resize_dataset(self.samples, self.dataset_size)
        
        print(f"📊 Dataset initialized with {len(self.samples)} samples")
        
        # 캐시 디렉토리 설정
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    
    def _resize_dataset(self, samples: List[Dict], target_size: int) -> List[Dict]:
        """데이터셋 크기 조정"""
        if target_size >= len(samples):
            return samples
        
        # 어휘 분포를 유지하면서 샘플링
        vocab_groups = defaultdict(list)
        for i, sample in enumerate(samples):
            key = tuple(sample['morphemes'])
            vocab_groups[key].append(i)
        
        # 각 어휘 그룹에서 균등하게 샘플링
        selected_indices = []
        samples_per_vocab = max(1, target_size // len(vocab_groups))
        
        for vocab_key, indices in vocab_groups.items():
            selected = random.sample(indices, min(samples_per_vocab, len(indices)))
            selected_indices.extend(selected)
        
        # 부족한 경우 추가 샘플링
        if len(selected_indices) < target_size:
            remaining = target_size - len(selected_indices)
            all_indices = set(range(len(samples)))
            available = list(all_indices - set(selected_indices))
            additional = random.sample(available, min(remaining, len(available)))
            selected_indices.extend(additional)
        
        return [samples[i] for i in selected_indices[:target_size]]
    
    def _get_cache_path(self, sample: Dict) -> str:
        """캐시 파일 경로 생성"""
        if not self.cache_dir:
            return None
        
        # 샘플 정보를 해시화하여 캐시 키 생성
        cache_key = f"{sample['video_id']}"
        config_hash = hashlib.md5(str(self.config).encode()).hexdigest()[:8]
        cache_filename = f"{cache_key}_{config_hash}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _load_from_cache(self, cache_path: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """캐시에서 전처리된 데이터 로드"""
        if not cache_path or not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cache from {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: str, data: Tuple[torch.Tensor, torch.Tensor]):
        """전처리된 데이터를 캐시에 저장"""
        if not cache_path:
            return
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Failed to save cache to {cache_path}: {e}")
    
    def _get_body_center(self, keypoints):
        """골반(8번)과 목(1번)의 중간점 계산"""
        neck_x = keypoints[1*2]
        neck_y = keypoints[1*2+1]
        pelvis_x = keypoints[8*2]
        pelvis_y = keypoints[8*2+1]
        return (neck_x + pelvis_x) / 2, (neck_y + pelvis_y) / 2

    def _get_shoulder_width(self, keypoints):
        """어깨(2번-5번) 간 거리 계산"""
        right_x, right_y = keypoints[2*2], keypoints[2*2+1]
        left_x, left_y = keypoints[5*2], keypoints[5*2+1]
        return math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)

    def _apply_hand_emphasis(self, keypoints, hand_scale=None):
        """손 키포인트 강조"""
        if hand_scale is None:
            hand_scale = self.hand_scale
        
        # 왼손: 인덱스 95~136 (21개 키포인트)
        # 오른손: 인덱스 137~178 (21개 키포인트)
        left_hand_indices = slice(95*2, 136*2)
        right_hand_indices = slice(137*2, 178*2)
        
        keypoints[left_hand_indices] *= hand_scale
        keypoints[right_hand_indices] *= hand_scale
        
        return keypoints
    
    def _preprocess_keypoints(self, keypoints: List[np.ndarray]) -> torch.Tensor:
        """키포인트 전처리"""
        processed_frames = []
        
        for frame_kps in keypoints:
            processed = frame_kps.copy()
            
            if self.normalize_center:
                # 중심점 기준 정규화
                center_x, center_y = self._get_body_center(frame_kps)
                processed[0::2] -= center_x  # x 좌표
                processed[1::2] -= center_y  # y 좌표
            
            if self.shoulder_width_scale:
                # 어깨 너비 스케일링
                shoulder_width = self._get_shoulder_width(frame_kps)
                processed = processed / max(shoulder_width, 1e-5)
            
            if self.hand_emphasis:
                # 손동작 강조
                processed = self._apply_hand_emphasis(processed)
            
            processed_frames.append(processed)
        
        return torch.tensor(np.array(processed_frames), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 캐시 확인
        cache_path = self._get_cache_path(sample)
        cached_data = self._load_from_cache(cache_path)

        if cached_data is not None:
            return cached_data
        
        # 프레임별 키포인트 파일들을 순서대로 로드
        keypoints = []
        for keypoint_file in sample['keypoint_files']:
            keypoint_path = os.path.join(sample['keypoint_dir'], keypoint_file)
            
            try:
                with open(keypoint_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                
                # 키포인트 추출
                if 'people' in frame_data and len(frame_data['people']) > 0:
                    person = frame_data['people']
                    
                    # 각 부위별 키포인트 추출
                    pose = person.get('pose_keypoints_2d', [])
                    face = person.get('face_keypoints_2d', [])
                    hand_left = person.get('hand_left_keypoints_2d', [])
                    hand_right = person.get('hand_right_keypoints_2d', [])
                    
                    # 2D 좌표만 추출 (confidence 제외)
                    pose_xy = []
                    if pose:
                        pose_arr = np.array(pose).reshape(-1, 3)
                        pose_xy = pose_arr[:, :2].flatten()
                    
                    face_xy = []
                    if face:
                        face_arr = np.array(face).reshape(-1, 3)
                        face_xy = face_arr[:, :2].flatten()
                    
                    hand_left_xy = []
                    if hand_left:
                        hand_left_arr = np.array(hand_left).reshape(-1, 3)
                        hand_left_xy = hand_left_arr[:, :2].flatten()
                    
                    hand_right_xy = []
                    if hand_right:
                        hand_right_arr = np.array(hand_right).reshape(-1, 3)
                        hand_right_xy = hand_right_arr[:, :2].flatten()
                    
                    # 전체 키포인트 결합 (274차원)
                    all_kps = np.concatenate([
                        pose_xy if len(pose_xy) == 50 else np.zeros(50),      # 25*2
                        face_xy if len(face_xy) == 140 else np.zeros(140),    # 70*2  
                        hand_left_xy if len(hand_left_xy) == 42 else np.zeros(42),   # 21*2
                        hand_right_xy if len(hand_right_xy) == 42 else np.zeros(42)  # 21*2
                    ])
                    keypoints.append(all_kps)
                else:
                    keypoints.append(np.zeros(274))
                    
            except Exception as e:
                print(f"⚠️ Error loading keypoint file {keypoint_file}: {e}")
                keypoints.append(np.zeros(274))
        
        # 키포인트 전처리
        keypoints_tensor = self._preprocess_keypoints(keypoints)
        
        # 모르픔 토큰화
        morphemes = sample['morphemes']
        gloss_idx = [self.vocab.stoi[token] for token in morphemes]
        gloss = torch.tensor(gloss_idx, dtype=torch.long)
        
        # 캐시에 저장
        result = (keypoints_tensor, gloss)
        self._save_to_cache(cache_path, result)
        
        return result


def collate_fn(batch, vocab):
    """배치 콜레이트 함수"""
    keypoints_list, gloss_list = zip(*batch)
    
    # 패딩
    keypoints_padded = pad_sequence(keypoints_list, batch_first=True, padding_value=0.0)
    keypoints_lengths = torch.tensor([kp.shape[0] for kp in keypoints_list])
    
    gloss_padded = pad_sequence(gloss_list, batch_first=True, padding_value=vocab.pad_idx)
    gloss_lengths = torch.tensor([len(g) for g in gloss_list])
    
    # Teacher forcing 시퀀스
    sos_token = torch.full((len(batch), 1), vocab.sos_idx, dtype=torch.long)
    eos_token = torch.full((len(batch), 1), vocab.eos_idx, dtype=torch.long)
    
    teacher_input = torch.cat([sos_token, gloss_padded], dim=1)
    teacher_target = torch.cat([gloss_padded, eos_token], dim=1)
    
    return (keypoints_padded, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths)