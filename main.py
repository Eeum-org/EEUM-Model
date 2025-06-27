import os
import time
import shutil
import logging
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from itertools import groupby
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from lib.config import get_cfg
from lib.dataset import build_data_loader
from lib.engine import default_argument_parser, default_setup
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl, wer_list
from lib.model import KeypointTransformerWithAttention

best_wer = 100
def eos_enhanced_loss(predictions, targets, target_lengths, ignore_index=0, eos_weight=2.0):
    """EOS 토큰 생성을 강화하는 손실 함수 - 배치 크기 불일치 해결"""
    batch_size, pred_max_len, vocab_size = predictions.shape
    target_batch_size, target_max_len = targets.shape
    
    # 배치 크기가 다른 경우 처리 (맨 처음에 추가)
    if batch_size != target_batch_size:
        min_batch = min(batch_size, target_batch_size)
        predictions = predictions[:min_batch]
        targets = targets[:min_batch]
        target_lengths = target_lengths[:min_batch]
        batch_size = min_batch
    
    # 시퀀스 길이가 다른 경우 처리
    max_len = min(pred_max_len, target_max_len)
    predictions = predictions[:, :max_len, :]
    targets = targets[:, :max_len]
    
    # 기본 cross-entropy
    predictions_flat = predictions.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')(predictions_flat, targets_flat)
    loss = loss.view(batch_size, max_len)
    
    # EOS 토큰 위치에 가중치 적용
    eos_mask = (targets == 2).float()  # EOS 토큰 위치
    weights = torch.ones_like(targets, dtype=torch.float)
    weights = weights + eos_mask * (eos_weight - 1)  # EOS 위치에만 가중치 증가
    
    # 패딩 마스크
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for i, length in enumerate(target_lengths):
        if length < max_len:
            mask[i, :length] = True
        else:
            mask[i, :] = True
    
    # 가중 손실 계산
    weighted_loss = loss * weights * mask.float()
    total_loss = weighted_loss.sum() / (mask.sum().float() + 1e-8)
    
    return total_loss

def sequence_mask(lengths, max_len):
    """시퀀스 마스크 생성"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask

def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg

def main(args):
    global best_wer
    logger = logging.getLogger()
    logging.basicConfig(filename="main_log_logger.log", level=logging.INFO)

    EPOCHS = 500
    start_epoch = 0
    cfg = setup(args)

    # timezone settings - for logging
    os.environ['TZ'] = "ROK"
    time.tzset()
    # make log with timestamp
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    cfg.freeze()
    is_gpu = torch.cuda.is_available()
    device = "cuda" if is_gpu else "cpu"
    train_loader, val_loader = build_data_loader(cfg)

    loss_gls = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True).to(device)

    # OpenPose 키포인트: 몸(25)+얼굴(70)+양손(21*2) = 137개 * 2차원(x,y) = 274
    input_dim = 274
    input_dim = getattr(cfg, 'KEYPOINT_DIM', 274)
    num_classes = len(train_loader.dataset.vocab)
    
    # 모델 하이퍼파라미터 변경
    model = KeypointTransformerWithAttention(
        num_classes=num_classes,
        input_dim=input_dim,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
        )
    
    model = model.to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    
    if cfg.RESUME:
        assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
        checkpoint = torch.load(cfg.RESUME)
        best_wer=checkpoint['best_wer']
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['state_dict'])
        # model = nn.DataParallel(model).cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Loaded checkpoint from {}.  "
            "start_epoch: {cp[epoch]} current_lr: {lr:.5f}  "
            "recoded WER: {cp[wer]:.3f} (best: {cp[best_wer]:.3f})".format(
                cfg.RESUME, cp=checkpoint, lr=current_lr
            )
        )
    
    if args.eval_only:
        metricss=validate(cfg, model, val_loader, loss_gls)
        print(f"valildation loss: {metricss['loss']:.3f}  validation WER: {metricss['wer']:.3f}")
        return
    
    writer = SummaryWriter(cfg.OUTPUT_DIR)
    data_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    epoch_time_meter = AverageMeter()

    last_best_idx = 0
    for epoch in range(start_epoch, EPOCHS):
        torch.cuda.empty_cache()
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1} / Train", mininterval=1)
        model.train()
        epoch_start = time.perf_counter()
        print(f"🚀 Epoch {epoch + 1}/{EPOCHS} - Segment-based KSL Training")
        print(f"📊 Best WER so far: {best_wer:.3f}")
        
        # 에포크가 진행될수록 teacher forcing 비율 감소
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.02)  # 에포크마다 2%씩 감소
        
        # 메트릭 초기화
        data_time_meter.reset()
        loss_meter.reset()
        iter_time_meter.reset()
        # for _iter, batch in enumerate(train_loader):
        for _iter, batch in enumerate(train_pbar):
            start = time.perf_counter()
            
            (keypoints, keypoint_lengths), (teacher_input, teacher_target, gloss_lengths) = batch

            # 빈 시퀀스 필터링 먼저 (GPU 이동 전에)
            valid_indices = (keypoint_lengths > 0) & (gloss_lengths > 0)
            if not valid_indices.any():
                print(f"⚠️  배치 {_iter}: 유효한 시퀀스가 없음, 건너뜀")
                continue
            
            # 필터링 적용
            keypoints = keypoints[valid_indices]
            keypoint_lengths = keypoint_lengths[valid_indices]
            teacher_input = teacher_input[valid_indices]
            teacher_target = teacher_target[valid_indices]
            gloss_lengths = gloss_lengths[valid_indices]
            
            data_time = time.perf_counter() - start
            data_time_meter.update(data_time, n=keypoints.size(0))
            
            # GPU로 이동 (한 번만)
            keypoints = keypoints.to(device, non_blocking=True)
            teacher_input = teacher_input.to(device, non_blocking=True)
            teacher_target = teacher_target.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
            # NaN 처리 추가
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Transformer를 위한 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]
            
            optimizer.zero_grad()
            
            try:
                # 배치 크기 확인 (디버깅)
                if _iter < 3:
                    print(f"🔍 Debug - Keypoints: {keypoints.shape}, Teacher Input: {teacher_input.shape}, Teacher Target: {teacher_target.shape}")
                    print(f"🔍 Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
                
                # Scheduled Sampling 적용 (5에포크부터)
                if epoch >= 5 and teacher_input.shape[1] > 2:
                    # Encoder 먼저 실행
                    src = torch.nan_to_num(keypoints, nan=0.0, posinf=1e3, neginf=-1e3)
                    src_proj = model.input_projection(src)
                    src_proj = src_proj.transpose(0, 1)
                    src_pe = model.pos_encoder(src_proj)
                    src_pe = src_pe.transpose(0, 1)
                    encoder_output = model.transformer_encoder(src_pe, src_key_padding_mask=src_key_padding_mask)
                    
                    # Scheduled sampling
                    gloss_scores = model._forward_training_with_scheduled_sampling(
                        encoder_output, teacher_input, src_key_padding_mask, teacher_forcing_ratio
                    )
                else:
                    # 기본 teacher forcing
                    gloss_scores = model(keypoints, target_tokens=teacher_input, src_key_padding_mask=src_key_padding_mask)

                # EOS 강화 손실 계산
                loss = eos_enhanced_loss(gloss_scores, teacher_target, gloss_lengths + 1, ignore_index=0, eos_weight=5.0)
                
                with torch.no_grad():
                    predictions = torch.argmax(gloss_scores, dim=-1)
                    
                    # 배치별 blank 통계
                    for b_idx in range(predictions.shape[0]):
                        pred_seq = predictions[b_idx][:keypoint_lengths[b_idx]]
                        blank_count = (pred_seq == 0).sum().item()
                        total_count = len(pred_seq)
                        blank_ratio = blank_count / max(total_count, 1)
                        
                        if _iter % 50 == 0 and b_idx == 0:  # 첫 번째 샘플만 출력
                            print(f"🔍 Batch {_iter}, Sample {b_idx}: "
                                f"Blank ratio: {blank_ratio:.3f} ({blank_count}/{total_count})")
                            
                            # 예측 시퀀스 샘플 출력
                            non_blank_preds = pred_seq[pred_seq != 0][:10]  # 처음 10개 non-blank
                            if len(non_blank_preds) > 0:
                                print(f"    Non-blank predictions: {non_blank_preds.tolist()}")

                # CTC Greedy Decoding 개선
                predictions = torch.argmax(gloss_scores, dim=-1)
                predictions_cpu = predictions.cpu().numpy()
                all_hypotheses, all_references = [], []
                vocab = train_loader.dataset.vocab
                # print(f"vocab : {vocab.itos}")
                # 배치별 디코딩
                for b_idx in range(predictions_cpu.shape[0]):
                    pred_seq = predictions_cpu[b_idx][:keypoint_lengths[b_idx].cpu().item()]
                    # CTC 디코딩: blank(0)와 연속 토큰 제거
                    decoded_indices = []
                    prev_token = -1
                    for token in pred_seq:
                        if token != 0 and token != prev_token:  # blank와 연속 토큰 제거
                            decoded_indices.append(token)
                        prev_token = token
                    
                    # 예측 문장 생성
                    if decoded_indices:
                        try:
                            hypothesis = vocab.arrays_to_sentences([decoded_indices])[0]
                            all_hypotheses.append(hypothesis)
                        except Exception as e:
                            print(f"⚠️  디코딩 오류: {e}")
                            all_hypotheses.append([])
                    else:
                        all_hypotheses.append([])
                    
                    # 참조 문장 생성 (teacher_target 사용)
                    ref_seq = teacher_target[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
                    
                    reference = vocab.arrays_to_sentences([ref_seq])[0]
                    all_references.append(reference)

                # 손실 검증
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Invalid loss detected at iteration {_iter}: {loss.item()}, skipping...")
                    continue
                
                loss_meter.update(loss.item(), n=keypoints.size(0))
                
                # 역전파
                loss.backward()
                
                # 그래디언트 클리핑 (더 강하게)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()
                
            except Exception as e:
                print(f"❌ Error in training step {_iter}: {e}")
                continue
            
            iter_time = time.perf_counter() - start
            iter_time_meter.update(iter_time, n=keypoints.size(0))
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'LR': f'{current_lr:.2e}'
            })
                # TensorBoard 로깅
            writer.add_scalar('train/loss', loss_meter.avg, epoch * len(train_loader) + _iter)
            writer.add_scalar('train/lr', current_lr, epoch * len(train_loader) + _iter)
        
        # 에포크별 스케줄러 업데이트 (MultiStep 등)
        if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' not in str(type(scheduler)):
            scheduler.step()

        epoch_time = time.perf_counter() - epoch_start
        epoch_time_meter.update(epoch_time, n=1)

        remain = EPOCHS - (epoch + 1)
        writer.add_scalar("misc/eta", remain * epoch_time_meter.avg, epoch)
        writer.flush()

        # validate
        metrics = validate(cfg, model, val_loader, 0, epoch)

        # TensorBoard 로깅
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.flush()

        logger.info(f"epoch: {epoch + 1}/{EPOCHS} Val loss: {metrics['loss']:.3f} Val WER: {metrics['wer']:.3f}")
        print(f"Epoch {epoch + 1}: Val Loss={metrics['loss']:.3f}, Val WER={metrics['wer']:.3f}")
        print(f"epoch: {epoch + 1}/{EPOCHS} loss: {loss_meter.avg:.3f} Val loss: {metrics['loss']:.3f} Val WER: {metrics['wer']:.3f}, lr : {current_lr:.1e}")
        
        # checkpoint
        is_best = metrics["wer"] < best_wer
        if is_best:
            print(f"New WER! since {epoch - last_best_idx + 1} epoch. \nWER: {best_wer:.3f} → {metrics['wer']:.3f}")
            best_wer = min(best_wer, metrics["wer"])
            # 체크포인트 저장
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'wer': metrics["wer"],
                'best_wer': best_wer,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, cfg.OUTPUT_DIR, f"checkpoint.epoch_{epoch + 1}.tar")
            last_best_idx = epoch + 1
        print(f"Diff : {best_wer - metrics["wer"]}")


def validate(cfg, model, val_loader, criterion, epoch) -> dict:
    logger = logging.getLogger()
    model.eval()
    val_loss_meter = AverageMeter()
    all_hypotheses, all_references = [], []
    vocab = val_loader.dataset.vocab
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 검증 시작: {len(val_loader)} 배치")
    val_pbar = tqdm(val_loader, desc = f"Epoch {epoch + 1} / Valid", mininterval=1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_pbar):
            (keypoints, keypoint_lengths), (teacher_input, teacher_target, gloss_lengths) = batch
            
            # 빈 시퀀스 필터링 먼저
            valid_indices = (keypoint_lengths > 0) & (gloss_lengths > 0)
            if not valid_indices.any():
                continue
            
            # 필터링 적용
            keypoints = keypoints[valid_indices]
            keypoint_lengths = keypoint_lengths[valid_indices]
            teacher_input = teacher_input[valid_indices]
            teacher_target = teacher_target[valid_indices]
            gloss_lengths = gloss_lengths[valid_indices]
            
            # GPU로 이동 (한 번만)
            keypoints = keypoints.to(device, non_blocking=True)
            teacher_input = teacher_input.to(device, non_blocking=True)
            teacher_target = teacher_target.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
            # NaN 처리
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]

            try:
                # 모델 추론 (teacher forcing으로 loss 계산)
                gloss_scores = model(keypoints, target_tokens=teacher_input, src_key_padding_mask=src_key_padding_mask)
                
                # 손실 계산
                loss = eos_enhanced_loss(gloss_scores, teacher_target, gloss_lengths + 1, ignore_index=0, eos_weight=5.0)
                
                # Inference 모드로 실제 예측 생성
                pred_sequences = model(keypoints, target_tokens=None, src_key_padding_mask=src_key_padding_mask)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_meter.update(loss.item(), n=keypoints.size(0))
                
                # 예측 결과 처리 (inference 결과 사용)
                predictions_cpu = pred_sequences.cpu().numpy()
                
                # 배치별 디코딩
                for b_idx in range(predictions_cpu.shape[0]):
                    pred_seq = predictions_cpu[b_idx]
                    
                    # 디버깅: 생성된 시퀀스 길이 확인
                    if b_idx == 0 and batch_idx < 3:
                        print(f"🔍 Generated sequence length: {len(pred_seq)}, tokens: {pred_seq[:8]}")
                    
                    # EOS 토큰(2) 위치 찾기
                    eos_pos = None
                    for i, token in enumerate(pred_seq):
                        if token == 2:  # EOS 토큰
                            eos_pos = i
                            break
                    
                    # EOS까지만 사용, 없으면 매우 강하게 제한
                    if eos_pos is not None:
                        pred_seq = pred_seq[:eos_pos]
                    else:
                        pred_seq = pred_seq[:5]  # 최대 5토큰으로 매우 강력 제한
                    
                    # CTC 디코딩: blank(0)와 연속 토큰 제거
                    decoded_indices = []
                    prev_token = -1
                    for token in pred_seq:
                        if token != 0 and token != prev_token:  # blank와 연속 토큰 제거
                            decoded_indices.append(token)
                        prev_token = token
                    
                    # 예측 문장 생성
                    if decoded_indices:
                        try:
                            hypothesis = vocab.arrays_to_sentences([decoded_indices])[0]
                            all_hypotheses.append(hypothesis)
                        except Exception as e:
                            print(f"⚠️  디코딩 오류: {e}")
                            all_hypotheses.append([])
                    else:
                        all_hypotheses.append([])
                    
                    # 참조 문장 생성 (teacher_target 사용)
                    ref_seq = teacher_target[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
                    
                    reference = vocab.arrays_to_sentences([ref_seq])[0]
                    all_references.append(reference)
                    # print(f"ref_seq : {ref_seq}, reference : {reference}, all_hyp : {list(filter(lambda x : x, all_hypotheses))}")
                
            except Exception as e:
                print(f"❌ Validation error at batch {batch_idx}: {e}")
                continue

    # 결과 샘플 출력 (디버깅용)
    print(f"\n📝 검증 결과 샘플 (총 {len(all_hypotheses)}개):")
    for i in range(min(5, len(all_hypotheses) // 5)):
        hyp_str = " ".join(all_hypotheses[i * 5]) if all_hypotheses[i * 5] else "<EMPTY>"
        ref_str = " ".join(all_references[i * 5]) if all_references[i * 5] else "<EMPTY>"
        print(f"  Sample {i+1}. Real : '{ref_str}, Pred: '{hyp_str}'")
        print(f"'")
    
    # WER 계산
    if all_hypotheses and all_references:
        # 텍스트 정제
        gls_ref = [clean_ksl(" ".join(ref)) if ref else "" for ref in all_references]
        gls_hyp = [clean_ksl(" ".join(hyp)) if hyp else "" for hyp in all_hypotheses]
        
        # 빈 문자열 처리
        valid_pairs = [(h, r) for h, r in zip(gls_hyp, gls_ref) if r.strip()]
        if valid_pairs:
            gls_hyp, gls_ref = zip(*valid_pairs)
            metrics = wer_list(hypotheses=list(gls_hyp), references=list(gls_ref))
        else:
            print("⚠️  유효한 참조 문장이 없습니다.")
            metrics = {"wer": 100.0}
            raise ValueError("유효한 참조 문장이 없습니다.")
    else:
        print("⚠️  예측 또는 참조 문장이 없습니다.")
        metrics = {"wer": 100.0}
    
    metrics.update({"loss": val_loss_meter.avg})
    
    print(f"📊 검증 완료: Loss={metrics['loss']:.4f}, WER={metrics['wer']:.3f}%")
    return metrics

def save_checkpoint(state_dict, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))

if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()
    # args.config_file='/workspace/AI_model/NIA_CSLR/configs/config.yaml'
    args.config_file='../config.yaml'

    main(args)