import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any
import json

from utils.metrics import AverageMeter, wer_list
from models.MSKA import eos_enhanced_loss


class MSKATrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, vocab, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vocab = vocab
        self.config = config
        self.device = device
        
        # Training state
        self.best_wer = float('inf')
        self.training_history = {
            'train_losses': [],
            'val_wers': [],
            'learning_rates': [],
            'best_wer': float('inf'),
            'best_epoch': 0
        }
        
        # Configuration
        self.epochs = config.training.get('epochs', 100)
        self.grad_clip_norm = config.training.get('grad_clip_norm', 2.0)
        self.eos_weight = config.loss.get('eos_weight', 3.0)
        
        # Output settings
        self.output_dir = config.output.get('dir', './outputs')
        self.save_best = config.output.get('save_best', True)
        self.save_every_n_epochs = config.output.get('save_every_n_epochs', 10)
        
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        loss_meter = AverageMeter()
        train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")

        for batch_idx, ((keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths)) in enumerate(train_pbar):
            keypoints = keypoints.to(self.device)
            teacher_input = teacher_input.to(self.device)
            teacher_target = teacher_target.to(self.device)
            keypoints_lengths = keypoints_lengths.to(self.device)
            
            # 마스크 생성
            src_key_padding_mask = (torch.arange(keypoints.size(1), device=self.device)[None, :] >= keypoints_lengths[:, None])
            tgt_key_padding_mask = (teacher_input == self.vocab.pad_idx)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(teacher_input.size(1)).to(self.device)

            self.optimizer.zero_grad()
            
            # Forward
            logits = self.model(
                keypoints, tgt=teacher_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )
            
            # Loss
            loss = eos_enhanced_loss(logits, teacher_target, ignore_index=self.vocab.pad_idx, eos_weight=self.eos_weight)
            
            # DEBUG: Print training details for first batch
            if batch_idx == 0 and epoch % 5 == 0:  # Every 5 epochs
                print(f"\n🔍 TRAINING DEBUG (Epoch {epoch}):")
                print(f"  Input shape: {keypoints.shape}")
                print(f"  Teacher input: {teacher_input}")
                print(f"  Teacher target: {teacher_target}")
                print(f"  Teacher input tokens: {[self.vocab.itos[i] for seq in teacher_input for i in seq if 0 <= i < len(self.vocab.itos)]}")
                print(f"  Teacher target tokens: {[self.vocab.itos[i] for seq in teacher_target for i in seq if 0 <= i < len(self.vocab.itos)]}")
                print(f"  Logits shape: {logits.shape}")
                print(f"  Loss: {loss.item():.4f}")
                
                # Show what model would predict (argmax)
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=-1)
                    pred_tokens = [self.vocab.itos[i] for seq in predictions for i in seq if 0 <= i < len(self.vocab.itos)]
                    print(f"  Current predictions: {predictions}")
                    print(f"  Predicted tokens: {pred_tokens}")
                    
                    # Show probabilities for the target tokens
                    probs = torch.softmax(logits, dim=-1)
                    for seq_idx in range(teacher_target.size(0)):
                        for pos_idx in range(teacher_target.size(1)):
                            target_token_id = teacher_target[seq_idx, pos_idx].item()
                            if target_token_id != self.vocab.pad_idx:
                                target_prob = probs[seq_idx, pos_idx, target_token_id].item()
                                target_token = self.vocab.itos[target_token_id] if target_token_id < len(self.vocab.itos) else f"UNK({target_token_id})"
                                print(f"  Prob of target '{target_token}' at pos {pos_idx}: {target_prob:.4f}")
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
                self.optimizer.step()
                loss_meter.update(loss.item())

            train_pbar.set_postfix({'Loss': f'{loss_meter.avg:.4f}'})

        return loss_meter.avg

    def validate(self, epoch: int) -> float:
        self.model.eval()
        all_hypotheses, all_references = [], []
        val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")

        with torch.no_grad():
            for batch_idx, ((keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths)) in enumerate(val_pbar):
                keypoints = keypoints.to(self.device)
                keypoints_lengths = keypoints_lengths.to(self.device)
                
                src_key_padding_mask = (torch.arange(keypoints.size(1), device=self.device)[None, :] >= keypoints_lengths[:, None])
                
                # Inference
                generated_seqs = self.model(keypoints, tgt=None, src_key_padding_mask=src_key_padding_mask)
                
                # DEBUG: Print detailed prediction info
                if batch_idx == 0:  # Only debug first batch
                    print(f"\n🔍 VALIDATION DEBUG (Epoch {epoch}):")
                    print(f"  Input shape: {keypoints.shape}")
                    print(f"  Input lengths: {keypoints_lengths}")
                    print(f"  Generated raw: {generated_seqs}")
                    print(f"  Teacher target raw: {teacher_target}")
                    
                    # Decode both
                    hyp_decoded = self.vocab.arrays_to_sentences(generated_seqs.cpu().numpy())
                    ref_decoded = self.vocab.arrays_to_sentences(teacher_target.cpu().numpy())
                    
                    print(f"  Generated tokens: {[self.vocab.itos[i] for seq in generated_seqs for i in seq if 0 <= i < len(self.vocab.itos)]}")
                    print(f"  Target tokens: {[self.vocab.itos[i] for seq in teacher_target for i in seq if 0 <= i < len(self.vocab.itos)]}")
                    print(f"  Hypothesis (after decoding): {hyp_decoded}")
                    print(f"  Reference (after decoding): {ref_decoded}")
                
                # Decode
                hypotheses = self.vocab.arrays_to_sentences(generated_seqs.cpu().numpy())
                references = self.vocab.arrays_to_sentences(teacher_target.cpu().numpy())
                
                all_hypotheses.extend(hypotheses)
                all_references.extend(references)

        # WER 계산
        wer = wer_list(hypotheses=all_hypotheses, references=all_references)["wer"]
        print(f"\n📊 Final validation results:")
        print(f"  All hypotheses: {all_hypotheses}")
        print(f"  All references: {all_references}")
        print(f"  Validation WER: {wer:.2f}%")
        return wer

    def save_checkpoint(self, epoch: int, wer: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'wer': wer,
            'vocab': self.vocab,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Best model saved with WER: {wer:.2f}%")
        
        # 정기적 저장
        if epoch % self.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(self.output_dir, f'model_epoch_{epoch}_wer_{wer:.2f}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"📁 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def save_training_history(self):
        history_path = os.path.join(self.output_dir, 'training_history.json')
        
        # Convert any tensors to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"📊 Training history saved to {history_path}")

    def train(self, start_epoch: int = 1) -> Dict[str, Any]:
        print(f"\n🚀 Starting training from epoch {start_epoch}")
        print(f"Total epochs: {self.epochs}")
        print(f"Best WER so far: {self.best_wer:.2f}%")
        
        for epoch in range(start_epoch, self.epochs + 1):
            print(f"\n=== Epoch {epoch}/{self.epochs} ===")
            
            # 훈련
            train_loss = self.train_one_epoch(epoch)
            
            # 검증
            wer = self.validate(epoch)
            
            # 스케줄러 업데이트
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(wer)
                    else:
                        self.scheduler.step()
            
            # 기록 업데이트
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_wers'].append(wer)
            
            if self.optimizer.param_groups:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
            
            # 최고 성능 모델 저장
            is_best = False
            if wer < self.best_wer:
                self.best_wer = wer
                self.training_history['best_wer'] = wer
                self.training_history['best_epoch'] = epoch
                is_best = True
            
            # 체크포인트 저장
            self.save_checkpoint(epoch, wer, is_best)
            
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val WER: {wer:.2f}%, Best WER: {self.best_wer:.2f}%")
        
        # 훈련 완료 후 기록 저장
        self.save_training_history()
        
        return {
            'best_wer': self.best_wer,
            'best_epoch': self.training_history['best_epoch'],
            'training_history': self.training_history
        }

