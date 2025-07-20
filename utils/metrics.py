import editdistance
from typing import List, Dict


class AverageMeter:
    """Computes and stores the average and current value"""
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
    """
    Calculate Word Error Rate (WER) between hypotheses and references
    """
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


def bleu_score(hypotheses: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculate BLEU score (simplified version)
    Note: This is a basic implementation. For production, consider using nltk.translate.bleu_score
    """
    raise NotImplementedError(
        "BLEU score calculation is not implemented. "
        "Please implement proper BLEU score calculation or use nltk.translate.bleu_score"
    )


def accuracy_score(hypotheses: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    """
    Calculate exact match accuracy
    """
    correct = 0
    total = len(hypotheses)
    
    for hyp, ref in zip(hypotheses, references):
        if hyp == ref:
            correct += 1
    
    accuracy = (correct / max(total, 1)) * 100
    return {"accuracy": accuracy}