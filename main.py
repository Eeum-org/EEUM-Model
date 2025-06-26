import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from lib.dataset.sign_dataset import SignDataset
from lib.model.sign_model import SignModel
from lib.config.settings import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from lib.utils import AverageMeter, clean_ksl, wer_list
import os

def train():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    ds = SignDataset(split='train')
    indices = len(ds) // 100
    ds_sub = Subset(ds, indices = [i for i in range(indices)])
    # loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    loader = DataLoader(ds_sub)
    itos = ds.vocab_itos
    val_loader = DataLoader(ds_sub)

    # model = SignModel(input_dim=input_dim, vocab_size=len(ds.vocab)).to(DEVICE)
    model = SignModel(vocab_size=len(ds.vocab)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        model.train()
        total = 0
        loss_sum = 0
        
        for kp, tgt in loader:
            kp, tgt = kp.to(DEVICE), tgt.to(DEVICE)

            if tgt.dim() > 1:
                tgt = tgt[:, 0]  # 첫 번째 토큰만 사용

            opt.zero_grad()
            logits = model(kp)
            loss = crit(logits, tgt)
            loss.backward()
            opt.step()
            
            total += 1
            loss_sum += loss.item()
            
        print(f"Epoch {ep+1}/{EPOCHS}, Loss: {loss_sum/total:.4f}")
        metrics = validate(model, val_loader, itos)
        print(f"WER={metrics['wer']:.3f}%, sub_rate={metrics['sub_rate']:.3f}, del_rate={metrics['del_rate']:3f}, ins_rate={metrics['ins_rate']:3f}")
def validate(model, val_loader, itos):
    # ds = SignDataset(split=split)

    model.eval()
    all_preds = []
    all_refs = []
    for kp, tgt in val_loader:
        with torch.no_grad():
            pred = model(kp.to(DEVICE)).argmax(dim=1).item()
            all_preds.append(itos[pred])
        all_refs.append(itos[tgt.tolist()[0][0]])

    if all_preds and all_refs:
        # 빈 문자열 처리
        valid_pairs = [(h, r) for h, r in zip(all_preds, all_refs) if r.strip()]
        if valid_pairs:
            gls_hyp, gls_ref = zip(*valid_pairs)
            metrics = wer_list(hypotheses=list(gls_hyp), references=list(gls_ref))
        else:
            metrics = {"wer": 100.0}
            raise ValueError("유효한 참조 문장이 없습니다.")
    else:
        print("예측 또는 참조 문장이 없습니다.")
        metrics = {"wer": 100.0}
    print(f"all_preds : {gls_hyp}, all_refs : {gls_ref}")
    return metrics
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()