import editdistance
from typing import List, Dict

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
    """WER 계산"""
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have the same length")
    
    total_dist = 0
    total_ref_len = 0
    
    for hyp, ref in zip(hypotheses, references):
        # 리스트를 문자열로 변환
        hyp_str = " ".join(hyp) if isinstance(hyp, list) else str(hyp)
        ref_str = " ".join(ref) if isinstance(ref, list) else str(ref)
        
        # 단어 단위로 분할
        hyp_words = hyp_str.split()
        ref_words = ref_str.split()
        
        # Edit distance 계산
        dist = editdistance.eval(hyp_words, ref_words)
        total_dist += dist
        total_ref_len += len(ref_words)
    
    wer = (total_dist / max(total_ref_len, 1)) * 100
    return {"wer": wer}

def clean_ksl(text: str) -> str:
    """KSL 텍스트 정제"""
    # 기본적인 텍스트 정제
    text = text.strip()
    text = " ".join(text.split())  # 여러 공백을 하나로
    return text
