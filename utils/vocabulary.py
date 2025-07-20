import os
import json
import pickle
from collections import defaultdict
from typing import List, Optional, Dict
from tqdm import tqdm

class GlossVocabulary:
    def __init__(self, tokens: Optional[List[str]] = None):
        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.UNK_TOKEN = "<UNK>"
        self.SIL_TOKEN = "<SIL>"  # Silence token for future use
        
        self.specials = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN, self.SIL_TOKEN]
    
        self.itos = []
        self.stoi = defaultdict(self.unk_token)
        
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.sil_idx = 4
        
        # 특수 토큰 추가
        for token in self.specials:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1
        
        # 일반 토큰 추가
        if tokens:
            for token in tokens:
                if token not in self.specials:
                    self.add_token(token)
    
    def unk_token(self):
        return self.unk_idx
    
    def add_token(self, token: str) -> int:
        """토큰을 어휘사전에 추가하고 인덱스 반환"""
        if token not in self.stoi:
            self.itos.append(token)
            self.stoi[token] = len(self.itos) - 1
        return self.stoi[token]
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """여러 토큰을 한번에 추가"""
        return [self.add_token(token) for token in tokens]
    
    def expand_vocab(self, new_tokens: List[str]):
        """어휘사전 확장"""
        print(f"Expanding vocabulary with {len(new_tokens)} new tokens...")
        added_count = 0
        for token in new_tokens:
            if token not in self.stoi:
                self.add_token(token)
                added_count += 1
        print(f"Added {added_count} new tokens. Vocabulary size: {len(self)}")
    
    def arrays_to_sentences(self, arrays: List[List[int]]) -> List[List[str]]:
        sentences = []
        for array in arrays:
            sentence = []
            for idx in array:
                if 0 <= idx < len(self.itos):
                    token = self.itos[idx]
                    if token not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                        sentence.append(token)
                else:
                    # 인덱스가 범위를 벗어나면 UNK 토큰 추가
                    sentence.append(self.UNK_TOKEN)
                    print(f"⚠️ Invalid token index {idx} (vocab size: {len(self.itos)})")
            sentences.append(sentence)
        return sentences
    
    def save(self, filepath: str):
        """어휘사전을 파일로 저장"""
        vocab_data = {
            'itos': self.itos,
            'stoi': dict(self.stoi),
            'special_indices': {
                'pad_idx': self.pad_idx,
                'sos_idx': self.sos_idx,
                'eos_idx': self.eos_idx,
                'unk_idx': self.unk_idx,
                'sil_idx': self.sil_idx
            }
        }
        
        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(vocab_data, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """파일로부터 어휘사전 로드"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                vocab_data = pickle.load(f)
        
        vocab = cls()
        vocab.itos = vocab_data['itos']
        vocab.stoi = defaultdict(vocab.unk_token)
        vocab.stoi.update(vocab_data['stoi'])
        
        # 특수 인덱스 복원
        special_indices = vocab_data.get('special_indices', {})
        vocab.pad_idx = special_indices.get('pad_idx', 0)
        vocab.sos_idx = special_indices.get('sos_idx', 1)
        vocab.eos_idx = special_indices.get('eos_idx', 2)
        vocab.unk_idx = special_indices.get('unk_idx', 3)
        vocab.sil_idx = special_indices.get('sil_idx', 4)
        
        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab)}")
        return vocab

    def __len__(self):
        return len(self.itos)


def build_vocab_from_morpheme_dir(morpheme_dir: str, direction_filter: Optional[str] = None) -> GlossVocabulary:
    """morpheme 디렉토리의 모든 파일로부터 어휘 사전 구축"""
    all_tokens = []
    processed_base_names = set()  # 중복 제거를 위한 기본 이름 추적
    
    # Find all morpheme files recursively
    morpheme_files = []
    for root, dirs, files in os.walk(morpheme_dir):
        for file in files:
            if file.endswith('_morpheme.json'):
                file_path = os.path.join(root, file)
                morpheme_files.append((file_path, file))
    
    for file_path, filename in tqdm(morpheme_files, desc="Building vocab"):
        # 방향 필터링 (F, L, R, U, D 중 하나만 처리)
        if direction_filter:
            if not any(f"_{direction}_" in filename for direction in ['F', 'L', 'R', 'U', 'D']):
                continue
        
        # 기본 이름 추출 (방향 정보 제거)
        base_name = filename
        for direction in ['_F_', '_L_', '_R_', '_U_', '_D_']:
            if direction in base_name:
                base_name = base_name.replace(direction, '_X_')
                break
        
        # 이미 처리된 기본 이름이면 스킵
        if base_name in processed_base_names:
            continue
        processed_base_names.add(base_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            morpheme_data = json.load(f)
        
        # data 배열에서 morpheme 추출
        for item in morpheme_data.get('data', []):
            attributes = item.get('attributes', [])
            if attributes and len(attributes) > 0:
                morpheme_name = attributes[0].get('name', '')
                if morpheme_name:
                    all_tokens.append(morpheme_name)
    
    unique_tokens = list(set(all_tokens))
    vocab = GlossVocabulary(tokens=unique_tokens)
    print(f"📚 어휘 사전 구축 완료: {len(vocab)} 토큰")
    return vocab


def load_or_build_vocab(vocab_path: str, morpheme_dir: str, force_rebuild: bool = False) -> GlossVocabulary:
    """어휘사전 로드 또는 빌드"""
    if os.path.exists(vocab_path) and not force_rebuild:
        print(f"Loading existing vocabulary from {vocab_path}")
        return GlossVocabulary.load(vocab_path)
    else:
        print(f"Building new vocabulary from {morpheme_dir}")
        vocab = build_vocab_from_morpheme_dir(morpheme_dir)
        vocab.save(vocab_path)
        return vocab