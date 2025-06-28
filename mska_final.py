#!/usr/bin/env python3
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
from typing import List, Optional, Dict
import editdistance

# ==============================================================================
# 1. 어휘 사전
# ==============================================================================
class GlossVocabulary:
    def __init__(self, tokens: Optional[List[str]] = None):
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.UNK_TOKEN = "<UNK>"
        
        self.specials = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
    
        self.itos = []
        self.stoi = defaultdict(self.unk_token)  # UNK index
        
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        
        # 특수 토큰 추가
        for token in self.specials:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1
        
        # 일반 토큰 추가
        if tokens:
            for token in tokens:
                if token not in self.specials:
                    self.itos.append(token)
                    self.stoi[token] = len(self.itos) - 1
    def unk_token(self):
        return 3
    def arrays_to_sentences(self, arrays: List[List[int]]) -> List[List[str]]:
        sentences = []
        for array in arrays:
            sentence = []
            for idx in array:
                if 0 <= idx < len(self.itos):
                    token = self.itos[idx]
                    if token not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                        sentence.append(token)
            sentences.append(sentence)
        return sentences

    def __len__(self):
        return len(self.itos)

# ==============================================================================
# 2. 유틸리티
# ==============================================================================
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def wer_list(hypotheses: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    total_dist = 0
    total_ref_len = 0
    
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp if isinstance(hyp, list) else [hyp]
        ref_words = ref if isinstance(ref, list) else [ref]
        
        dist = editdistance.eval(hyp_words, ref_words)
        total_dist += dist
        total_ref_len += len(ref_words)
    
    wer = (total_dist / max(total_ref_len, 1)) * 100
    return {"wer": wer}

# ==============================================================================
# 3. 데이터셋
# ==============================================================================
class SignDataset(Dataset):
    def __init__(self, data_root, keypoint_subdir, morpheme_subdir, vocab):
        self.data_root = data_root
        self.keypoint_dir = os.path.join(data_root, keypoint_subdir)
        self.morpheme_dir = os.path.join(data_root, morpheme_subdir)
        self.vocab = vocab
        
        print(f"Keypoint dir: {self.keypoint_dir}")
        print(f"Morpheme dir: {self.morpheme_dir}")
        
        # 키포인트 디렉토리의 하위 폴더들 (각 영상별 폴더)
        self.samples = []
        
        if not os.path.exists(self.keypoint_dir):
            print(f"❌ Keypoint directory not found: {self.keypoint_dir}")
            return
            
        if not os.path.exists(self.morpheme_dir):
            print(f"❌ Morpheme directory not found: {self.morpheme_dir}")
            return
        
        # 키포인트 디렉토리의 하위 폴더들 순회
        video_folders = [d for d in os.listdir(self.keypoint_dir) 
                        if os.path.isdir(os.path.join(self.keypoint_dir, d))]
        
        print(f"발견된 영상 폴더들: {video_folders[:5]}...")  # 처음 5개만 출력
        
        for video_id in video_folders:
            video_keypoint_dir = os.path.join(self.keypoint_dir, video_id)
            morpheme_file = os.path.join(self.morpheme_dir, f"{video_id}_morpheme.json")
            
            # morpheme 파일이 있는지 확인
            if os.path.exists(morpheme_file):
                # morpheme.json에서 토큰 추출
                try:
                    with open(morpheme_file, 'r', encoding='utf-8') as f:
                        morpheme_data = json.load(f)
                    
                    # data 배열에서 모든 morpheme 추출
                    morphemes = []
                    for item in morpheme_data.get('data', []):
                        morpheme_name = item.get('attributes', {})[0].get('name', '')
                        if morpheme_name:
                            morphemes.append(morpheme_name)
                    
                    # 해당 영상의 키포인트 파일들 수집
                    keypoint_files = [f for f in os.listdir(video_keypoint_dir) 
                                    if f.endswith('_keypoints.json')]
                    
                    if morphemes and keypoint_files:  # morpheme과 keypoint 파일이 모두 있는 경우
                        # 키포인트 파일들을 프레임 번호 순으로 정렬
                        keypoint_files.sort(key=lambda x: int(x.split('_')[-2]))
                        
                        self.samples.append({
                            'video_id': video_id,
                            'keypoint_dir': video_keypoint_dir,
                            'keypoint_files': keypoint_files,
                            'morphemes': morphemes
                        })
                        
                except Exception as e:
                    print(f"⚠️ Error processing {video_id}: {e}")
                    continue
            else:
                print(f"⚠️ Morpheme file not found for {video_id}")

        print(f"📊 로드된 샘플 수: {len(self.samples)}")
    ########################################################
    # 04:14추가
    ########################################################
    def _get_body_center(self, keypoints):
        """골반(8번)과 목(1번)의 중간점 계산"""
        # OpenPose 포맷: 0:코, 1:목, 8:골반
        neck_x = keypoints[1*2]    # 목 x 좌표
        neck_y = keypoints[1*2+1]  # 목 y 좌표
        pelvis_x = keypoints[8*2]    # 골반 x 좌표
        pelvis_y = keypoints[8*2+1]  # 골반 y 좌표
        return (neck_x + pelvis_x) / 2, (neck_y + pelvis_y) / 2

    def _get_shoulder_width(self, keypoints):
        """어깨(2번-5번) 간 거리 계산 (검색 결과[3] 참조)"""
        # 오른쪽 어깨(2번), 왼쪽 어깨(5번)
        right_x, right_y = keypoints[2*2], keypoints[2*2+1]
        left_x, left_y = keypoints[5*2], keypoints[5*2+1]
        return math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)

    def _apply_hand_emphasis(self, keypoints, hand_scale=1.5):
        """손 키포인트 강조 (검색 결과[4]에서 영감)"""
        # 왼손: 인덱스 95~136 (21개 키포인트)
        # 오른손: 인덱스 137~178 (21개 키포인트)
        left_hand_indices = slice(95*2, 136*2)
        right_hand_indices = slice(137*2, 178*2)
        
        keypoints[left_hand_indices] *= hand_scale
        keypoints[right_hand_indices] *= hand_scale
    ########################################################
    # 04:14추가
    ########################################################
        return keypoints
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 프레임별 키포인트 파일들을 순서대로 로드
        keypoints = []
        for keypoint_file in sample['keypoint_files']:
            keypoint_path = os.path.join(sample['keypoint_dir'], keypoint_file)
            
            try:
                with open(keypoint_path, 'r', encoding='utf-8') as f:
                    frame_data = json.load(f)
                
                # 키포인트 추출 로직 (이전과 동일)
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
                
    ########################################################
    # 04:14추가
    ########################################################
        processed_frames = []
        for frame_kps in keypoints:  # 각 프레임 처리
            # 1. 중심점 기준 정규화
            center_x, center_y = self._get_body_center(frame_kps)
            normalized = frame_kps.copy()
            normalized[0::2] -= center_x  # x 좌표
            normalized[1::2] -= center_y  # y 좌표
            
            # 2. 어깨 너비 스케일링 (검색 결과[3] 참조)
            shoulder_width = self._get_shoulder_width(frame_kps)
            scaled = normalized / max(shoulder_width, 1e-5)  # Zero division 방지
            
            # 3. 손동작 강조
            hand_enhanced = self._apply_hand_emphasis(scaled)
            processed_frames.append(hand_enhanced)
        
        keypoints = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    ########################################################
    # 04:14추가
    ########################################################
        keypoints = torch.tensor(np.array(keypoints), dtype=torch.float32)
        
        # 모르픔 토큰화
        morphemes = sample['morphemes']
        gloss_idx = [self.vocab.stoi[token] for token in morphemes]
        gloss = torch.tensor(gloss_idx, dtype=torch.long)
        
        return keypoints, gloss

def collate_fn(batch, vocab):
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

# ==============================================================================
# 4. 모델
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MSKA_Model(nn.Module):
    def __init__(self, num_classes, input_dim=274, d_model=1024, nhead=8, 
                 num_encoder_layers=10, num_decoder_layers=8, dim_feedforward=2048, dropout=0.2):
    # def __init__(self, num_classes, input_dim=274, d_model=512, nhead=8, 
    #              num_encoder_layers=6, num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Encoder
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        self.target_embedding = nn.Embedding(num_classes, d_model)
        self.target_pos_embedding = nn.Embedding(500, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.output_projection = nn.Linear(d_model, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # NaN 처리
        src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Encoder
        src_proj = self.input_projection(src)
        src_proj = src_proj.transpose(0, 1)
        src_pe = self.pos_encoder(src_proj)
        src_pe = src_pe.transpose(0, 1)
        
        encoder_output = self.transformer_encoder(src_pe, src_key_padding_mask=src_key_padding_mask)
        
        if tgt is not None:  # Training
            # Target embedding
            tgt_emb = self.target_embedding(tgt)
            batch_size, tgt_len = tgt.shape
            positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
            tgt_pos_emb = self.target_pos_embedding(positions)
            tgt_input = tgt_emb + tgt_pos_emb
            
            # Decoder
            decoder_output = self.transformer_decoder(
                tgt_input, encoder_output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            return self.output_projection(decoder_output)
        else:  # Inference
            return self._inference(encoder_output, src_key_padding_mask)

    def _inference(self, encoder_output, src_key_padding_mask, max_len=15, sos_idx=1, eos_idx=2):
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len - 1):
            if finished.all():
                break
                
            tgt_emb = self.target_embedding(generated)
            seq_len = generated.shape[1]
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            tgt_pos_emb = self.target_pos_embedding(positions)
            tgt_input = tgt_emb + tgt_pos_emb
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            decoder_output = self.transformer_decoder(
                tgt_input, encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            logits = self.output_projection(decoder_output[:, -1, :])
            
            # EOS 강제 생성
            if step >= 8:
                logits[:, eos_idx] += 2.0
            
            next_token = logits.argmax(dim=-1).unsqueeze(1)
            next_token = torch.where(finished.unsqueeze(1), 
                                   torch.full_like(next_token, eos_idx), 
                                   next_token)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            newly_finished = (next_token.squeeze(1) == eos_idx)
            finished = finished | newly_finished
        
        return generated

# ==============================================================================
# 5. 손실 함수
# ==============================================================================
def eos_enhanced_loss(predictions, targets, ignore_index=0, eos_weight=3.0):
    vocab_size = predictions.size(-1)
    predictions_flat = predictions.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    weights = torch.ones(vocab_size, device=predictions.device)
    weights[2] = eos_weight  # EOS index
    
    loss = F.cross_entropy(predictions_flat, targets_flat, weight=weights, ignore_index=ignore_index)
    return loss

# ==============================================================================
# 6. 학습/검증 함수
# ==============================================================================
def train_one_epoch(model, train_loader, optimizer, epoch, device, vocab):
    model.train()
    loss_meter = AverageMeter()
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")

    for (keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths) in train_pbar:
        keypoints = keypoints.to(device)
        teacher_input = teacher_input.to(device)
        teacher_target = teacher_target.to(device)
        keypoints_lengths = keypoints_lengths.to(device)
        
        # 마스크 생성
        src_key_padding_mask = (torch.arange(keypoints.size(1), device=device)[None, :] >= keypoints_lengths[:, None])
        tgt_key_padding_mask = (teacher_input == vocab.pad_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(teacher_input.size(1)).to(device)

        optimizer.zero_grad()
        
        # Forward
        logits = model(
            keypoints, tgt=teacher_input,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Loss
        loss = eos_enhanced_loss(logits, teacher_target, ignore_index=vocab.pad_idx)
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_meter.update(loss.item())

        train_pbar.set_postfix({'Loss': f'{loss_meter.avg:.4f}'})

    return loss_meter.avg

def validate(model, val_loader, epoch, device, vocab):
    model.eval()
    all_hypotheses, all_references = [], []
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")

    with torch.no_grad():
        for (keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths) in val_pbar:
            keypoints = keypoints.to(device)
            keypoints_lengths = keypoints_lengths.to(device)
            
            src_key_padding_mask = (torch.arange(keypoints.size(1), device=device)[None, :] >= keypoints_lengths[:, None])
            
            # Inference
            generated_seqs = model(keypoints, tgt=None, src_key_padding_mask=src_key_padding_mask)
            
            # Decode
            hypotheses = vocab.arrays_to_sentences(generated_seqs.cpu().numpy())
            references = vocab.arrays_to_sentences(teacher_target.cpu().numpy())
            
            all_hypotheses.extend(hypotheses)
            all_references.extend(references)

    # WER 계산
    wer = wer_list(hypotheses=all_hypotheses, references=all_references)["wer"]
    print(f"Validation WER: {wer:.2f}%")
    return wer

# ==============================================================================
# 7. 메인 함수
# ==============================================================================
def build_vocab_from_morpheme_dir(morpheme_dir):
    """morpheme 디렉토리의 모든 파일로부터 어휘 사전 구축"""
    all_tokens = []
    
    morpheme_files = [f for f in os.listdir(morpheme_dir) if f.endswith('_morpheme.json')]
    
    for morpheme_file in tqdm(morpheme_files, desc="Building vocab"):
        file_path = os.path.join(morpheme_dir, morpheme_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            morpheme_data = json.load(f)
        
        # data 배열에서 morpheme 추출
        for item in morpheme_data.get('data', []):
            morpheme_name = item.get('attributes', {})[0].get('name', '')
            if morpheme_name:
                all_tokens.append(morpheme_name)
    
    unique_tokens = list(set(all_tokens))
    vocab = GlossVocabulary(tokens=unique_tokens)
    print(f"📚 어휘 사전 구축 완료: {len(vocab)} 토큰")
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='데이터 루트 디렉토리')
    parser.add_argument('--train_keypoint_dir', required=True, help='훈련용 키포인트 디렉토리')
    parser.add_argument('--val_keypoint_dir', required=True, help='검증용 키포인트 디렉토리')
    parser.add_argument('--train_morpheme_dir', required=True, help='훈련용 morpheme.json')
    parser.add_argument('--val_morpheme_dir', required=True, help='검증용 morpheme.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='./outputs')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 어휘 사전
    vocab = build_vocab_from_morpheme_dir(os.path.join(args.data_root, args.train_morpheme_dir))
    
    # 데이터셋
    train_dataset = SignDataset(args.data_root, args.train_keypoint_dir, args.train_morpheme_dir, vocab)
    val_dataset = SignDataset(args.data_root, args.val_keypoint_dir, args.val_morpheme_dir, vocab)
    
    # 데이터로더
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, vocab),
        shuffle=True,
        num_workers=16,  # Windows 호환성
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        collate_fn=lambda b: collate_fn(b, vocab),
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True
    )
    
    # 모델
    model = MSKA_Model(num_classes=len(vocab)).to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, #verbose=True
    )
    
    # 학습 루프
    best_wer = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # 훈련
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device, vocab)
        
        # 검증
        wer = validate(model, val_loader, epoch, device, vocab)
        
        # 스케줄러
        scheduler.step(wer)
        
        # 모델 저장
        if wer < best_wer:
            best_wer = wer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'wer': wer,
                'vocab': vocab
            }, os.path.join(args.output_dir, f'model_epoch_{epoch + 1}_wer_{wer:.2f}.pth'))
            print(f"✅ Best model saved with WER: {wer:.2f}%")
            
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val WER: {wer:.2f}%, Best WER: {best_wer:.2f}%")

if __name__ == "__main__":
    main()