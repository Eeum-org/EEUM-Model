import json
import os
import torch
from lib.config.settings import MAX_SEQ_LEN

def load_json(path):
    # JSON 파일 로드
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON from {path}: {e}")
        return None

def pad_sequence(seq, max_len, pad_val=0):
    """시퀀스 길이 패딩 (패딩/자르기)"""
    length = seq.shape[0]
    if length < max_len:
        pad = torch.full((max_len-length, *seq.shape[1:]), pad_val)
        return torch.cat([seq, pad], dim=0)
    else:
        return seq[:max_len]

def extract_kp(frame_data):
    """단일 프레임에서 얼굴·양손 키포인트 추출"""
    face = torch.tensor(frame_data.get('face_keypoints_2d', [])).view(-1,3) \
           if frame_data.get('face_keypoints_2d') else torch.zeros(70,3)
    lh = torch.tensor(frame_data.get('hand_left_keypoints_2d', [])).view(-1,3) \
         if frame_data.get('hand_left_keypoints_2d') else torch.zeros(21,3)
    rh = torch.tensor(frame_data.get('hand_right_keypoints_2d', [])).view(-1,3) \
         if frame_data.get('hand_right_keypoints_2d') else torch.zeros(21,3)
    return torch.cat([face, lh, rh], dim=0)

def load_available_views(base_path, base_name):
    """
    사용 가능한 방향(F, U, D, L, R) 폴더를 탐색하여 리스트로 반환합니다.
    실제 데이터 구조: base_path/base_name_VIEW/ 폴더 존재 여부 확인
    """
    views = []
    for v in ['F', 'U', 'D', 'L', 'R']:
        view_folder_path = os.path.join(base_path, f"{base_name}_{v}")
        if os.path.exists(view_folder_path) and os.path.isdir(view_folder_path):
            views.append(v)
    
    return views if views else ['F']  # 하나도 없으면 전면(F)을 기본으로 가정