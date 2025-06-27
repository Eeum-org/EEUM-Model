from torch.utils.data import Dataset
from typing import List, NoReturn, Optional
from collections import Counter, defaultdict

SIL_TOKEN = "<SIL>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
BLANK_TOKEN = "<BNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

class GlossVocabulary:
    def __init__(self, tokens: Optional[List[str]] = None):
        self.specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, BLANK_TOKEN, SIL_TOKEN]
        # 0: pad, 1: sos, 2: eos, 3: unk, 4: blank, 5: sil
        
        self.itos = []
        self.stoi = defaultdict(self._default_unk_id)
        
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.blank_token = BLANK_TOKEN
        self.sil_token = SIL_TOKEN
        
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.bnk_idx = 4
        self.sil_idx = 5
        
        if tokens is not None:
            self._from_list(tokens)
        else:
            self._from_list([])

    def _default_unk_id(self):
        return self.unk_idx
    
    def _from_list(self, tokens: List[str]):
        for token in self.specials:
            if token not in self.itos:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
        
        for token in tokens:
            if token not in self.itos and token not in self.specials:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    def arrays_to_sentences(self, arrays: List[List[int]]) -> List[List[str]]:
        sentences = []
        for array in arrays:
            sentence = []
            for idx in array:
                if 0 <= idx < len(self.itos):
                    token = self.itos[idx]
                    if token not in [self.pad_token, self.sos_token, self.eos_token, self.blank_token]:
                        sentence.append(token)
                else:
                    sentence.append(self.unk_token)
            sentences.append(sentence)
        return sentences

    def __len__(self):
        return len(self.itos)

def build_vocab(annotations, max_size: int = 10000, min_freq: int = 1) -> GlossVocabulary:
    """JSON 파일들로부터 어휘 사전 구축"""
    import json
    result = set()
    for item in annotations:
        result.update(item.split())
    
    # sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    vocab_tokens = [token for token in result]
    
    vocab = GlossVocabulary(tokens=vocab_tokens)
    
    print(f"📚 어휘 사전 구축 완료: {len(vocab)} 토큰")
    print(f"🔤 특수 토큰: SOS={vocab.sos_idx}, EOS={vocab.eos_idx}, PAD={vocab.pad_idx}, UNK={vocab.unk_idx}")
    return vocab