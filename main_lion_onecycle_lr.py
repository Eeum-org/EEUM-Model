import logging
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
import time
import torch.optim as optim
from lion_pytorch import Lion
from tqdm.auto import tqdm
from lib.config import get_cfg
from lib.dataset import build_data_loader
from torch.utils.tensorboard import SummaryWriter
from lib.engine import default_argument_parser, default_setup
from lib.model.sign_model_keypoint import KeypointTransformer
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl, wer_list
import sys
best_wer = 100
def check_weight_initialization(model):
    """가중치 초기화 상태 확인"""
    print("=== 가중치 초기화 상태 ===")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 가중치 통계
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            print(f"{name}:")
            print(f"  형태: {param.shape}")
            print(f"  평균: {mean_val:.6f}, 표준편차: {std_val:.6f}")
            print(f"  범위: [{min_val:.6f}, {max_val:.6f}]")
            
            # 초기화 문제 감지
            if std_val < 1e-6:
                print(f"  ⚠️ 표준편차가 너무 작음 (거의 0)")
            if abs(mean_val) > 1.0:
                print(f"  ⚠️ 평균값이 너무 큼")
            if std_val > 2.0:
                print(f"  ⚠️ 표준편차가 너무 큼")
            
            # 모든 값이 같은지 확인
            if param.data.unique().numel() == 1:
                print(f"  ⚠️ 모든 가중치가 동일함!")
            
            print()
def check_gradient_flow(model, loss):
    """그래디언트 흐름 확인"""
    print("=== 그래디언트 흐름 확인 ===")
    
    # 역전파 수행
    loss.backward(retain_graph=True)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"{name}: grad_norm = {grad_norm:.6f}")
            
            # 그래디언트 문제 감지
            if grad_norm < 1e-7:
                print(f"  ⚠️ 그래디언트가 너무 작음 (vanishing)")
            elif grad_norm > 100:
                print(f"  ⚠️ 그래디언트가 너무 큼 (exploding)")
            elif torch.isnan(param.grad).any():
                print(f"  ⚠️ 그래디언트에 NaN 존재")
        else:
            print(f"{name}: 그래디언트 없음!")
def debug_forward_pass(model, input_tensor):
    """Forward pass 단계별 디버깅"""
    print("=== Forward Pass 디버깅 ===")
    
    x = input_tensor
    print(f"입력: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    
    # 각 모듈별로 출력 확인
    for i, (name, module) in enumerate(model.named_children()):
        if hasattr(module, '__call__'):
            x = module(x)
            print(f"{i+1}. {name}: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
            
            # 이상 값 체크
            if torch.isnan(x).any():
                print(f"  ⚠️ {name}에서 NaN 발생!")
                break
            if torch.isinf(x).any():
                print(f"  ⚠️ {name}에서 Inf 발생!")
                break
            if x.std() < 1e-6:
                print(f"  ⚠️ {name}에서 출력이 거의 상수!")
    
    return x

def comprehensive_model_diagnosis(model, sample_input, sample_target):
    """종합적인 모델 진단"""
    print("=" * 50)
    print("모델 종합 진단 시작")
    print("=" * 50)
    
    # 1. 모델 구조 확인
    print("1. 모델 구조:")
    print(model)
    
    # 2. 가중치 초기화 확인
    check_weight_initialization(model)
    
    # 3. Forward pass 테스트
    output = debug_forward_pass(model, sample_input)
    
    # 4. Loss 계산 및 그래디언트 확인
    if sample_target is not None:
        criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        try:
            loss = criterion(output.log_softmax(dim=-1).permute(1, 0, 2), 
                           sample_target, 
                           torch.tensor([output.size(1)]), 
                           torch.tensor([len(sample_target)]))
            print(f"Loss 계산 성공: {loss.item():.4f}")
            check_gradient_flow(model, loss)
        except Exception as e:
            print(f"⚠️ Loss 계산 실패: {e}")
    
    print("=" * 50)
    print("진단 완료")
    print("=" * 50)

def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = default_setup(cfg, args)
    return cfg

####################################################################################################

def main(args):
    global best_wer
    logger = logging.getLogger()
    logging.basicConfig(filename="segment_ksl_log.log", level=logging.INFO)
    
    EPOCHS = 2000  # 에포크 수 증가
    start_epoch = 0
    
    cfg = setup(args)
    cfg.GPU_ID = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.environ['TZ'] = "ROK"
    time.tzset()
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + time.strftime("_%Y-%m-%d_%H_%M_%S", time.localtime())
    cfg.freeze()
    # 시간 구간별 데이터로더 빌드
    train_loader, val_loader = build_data_loader(cfg)
    
    # CTC 손실 함수 (zero_infinity=True로 변경)
    loss_gls = nn.CTCLoss(blank=0, zero_infinity=True).to(cfg.GPU_ID)
    
    # 키포인트 차원을 동적으로 계산
    # OpenPose: 몸(25)+얼굴(70)+양손(21*2) = 137개 * 2차원(x,y) = 274
    input_dim = getattr(cfg, 'KEYPOINT_DIM', 274)
    num_classes = len(train_loader.dataset.vocab)
    
    print(f"🔧 모델 설정: input_dim={input_dim}, num_classes={num_classes}")
    print(f"📊 Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # 모델 하이퍼파라미터 개선
    model = KeypointTransformer(
        num_classes=num_classes,
        input_dim=input_dim,
        d_model=512,  # 모델 용량 증가
        nhead=8,
        num_encoder_layers=8,  # 레이어 수 증가
        dim_feedforward=4096,  # FFN 크기 증가
        dropout=0.1
    )
    # sample_input = torch.randn(1, 100, 274).to(cfg.GPU_ID)
    # sample_target = torch.tensor([1, 2, 3])
    # comprehensive_model_diagnosis(model, sample_input, sample_target)
    # sys.exit(-1)
    model = model.to(cfg.GPU_ID)
    
    # 옵티마이저 설정 개선
    # optimizer = build_optimizer(cfg, model)
    # scheduler = build_lr_scheduler(cfg, optimizer)
    
    # # 학습률 스케줄러 개선 (ReduceLROnPlateau 사용)
    # if not hasattr(scheduler, 'step') or 'MultiStep' in str(type(scheduler)):
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         # optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-7
    #         optimizer, mode='min', factor=0.6, patience=5, verbose=True, min_lr=1e-8, cooldown = 2, threshold = 5e-3, threshold_mode='rel'
    #     )
    #     print("🔄 ReduceLROnPlateau 스케줄러로 변경")
    ####################################################################################################
    optimizer = Lion(
        model.parameters(),
        lr=1e-3,              # max_lr과 동일하게 설정
        betas=(0.9, 0.99),    # Lion 기본값
        weight_decay=1e-2     # AdamW의 10배
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,                    # 실제 핵심 학습률
        steps_per_epoch=len(train_loader),
        epochs=200,                      # 수어 인식 권장 에포크
        pct_start=0.3,                  # 30% 워밍업
        div_factor=10,                  # 시작 LR = 1e-3/25 = 4e-5
        final_div_factor=1000,         # 최종 LR = 1e-3/10000 = 1e-7
        anneal_strategy='cos'           # 코사인 어닐링
    )
    ####################################################################################################
    
    # 체크포인트 로딩
    if cfg.RESUME:
        assert os.path.isfile(cfg.RESUME), "Error: no checkpoint directory found!"
        checkpoint = torch.load(cfg.RESUME, map_location=cfg.GPU_ID, weights_only=False)
        best_wer = checkpoint['best_wer']
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(f"Loaded checkpoint from {cfg.RESUME}. Resuming at epoch {start_epoch}.")
        print(f"✅ 체크포인트 로드: epoch {start_epoch}, best WER: {best_wer:.3f}")
    
    # 검증만 실행하는 경우
    if args.eval_only:
        metrics = validate(cfg, model, val_loader, loss_gls, start_epoch)
        print(f"Validation loss: {metrics['loss']:.3f}, Validation WER: {metrics['wer']:.3f}")
        return
    
    # TensorBoard 설정
    writer = SummaryWriter(cfg.OUTPUT_DIR)
    data_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    iter_time_meter = AverageMeter()
    epoch_time_meter = AverageMeter()

    ####################################################################################################
    
    # 학습 루프
    last_best_idx = 0
    for epoch in range(start_epoch, EPOCHS):
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1} / Train", mininterval=1)
        model.train()
        epoch_start = time.perf_counter()
        print(f"🚀 Epoch {epoch + 1}/{EPOCHS} - Segment-based KSL Training")
        print(f"📊 Best WER so far: {best_wer:.3f}")
        
        # 메트릭 초기화
        data_time_meter.reset()
        loss_meter.reset()
        iter_time_meter.reset()
        # for _iter, batch in enumerate(train_loader):
        for _iter, batch in enumerate(train_pbar):
            torch.cuda.empty_cache()
            start = time.perf_counter()
            
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch

            # 빈 시퀀스 필터링 (개선)
            valid_indices = (keypoint_lengths > 0) & (gloss_lengths > 0)
            if not valid_indices.any():
                print(f"⚠️  배치 {_iter}: 유효한 시퀀스가 없음, 건너뜀")
                continue
            
            keypoints = keypoints[valid_indices]
            keypoint_lengths = keypoint_lengths[valid_indices]
            glosses = glosses[valid_indices]
            gloss_lengths = gloss_lengths[valid_indices]
            
            data_time = time.perf_counter() - start
            data_time_meter.update(data_time, n=keypoints.size(0))
            
            # GPU로 이동
            keypoints = keypoints.to(cfg.GPU_ID, non_blocking=True)
            glosses = glosses.to(cfg.GPU_ID, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(cfg.GPU_ID, non_blocking=True)
            gloss_lengths = gloss_lengths.to(cfg.GPU_ID, non_blocking=True)
            
            # NaN 처리 추가
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Transformer를 위한 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]
            
            optimizer.zero_grad()
            
            try:
                # 모델 추론
                gloss_scores = model(keypoints, src_key_padding_mask=src_key_padding_mask)

                # CTC 손실 계산을 위한 형태 변환
                gloss_probs = F.log_softmax(gloss_scores, dim=-1).permute(1, 0, 2)  # (T, N, C)

                # CTC 손실 계산
                loss = loss_gls(gloss_probs, glosses, keypoint_lengths.long(), gloss_lengths.long())
                ##########################
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
                    
                    # 참조 문장 생성
                    ref_seq = glosses[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
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
            # 로그 출력
            if (_iter + 1) % cfg.PERIODS.LOG_ITERS == 0:
                """ current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{EPOCHS}][{_iter+1}/{len(train_loader)}] "
                      f"Loss: {loss_meter.avg:.4f}, "
                      f"LR: {current_lr:.2e}, "
                      f"Data Time: {data_time_meter.avg:.3f}s, "
                      f"Iter Time: {iter_time_meter.avg:.3f}s") """
                pass
                # TensorBoard 로깅
            writer.add_scalar('train/loss', loss_meter.avg, epoch * len(train_loader) + _iter)
            writer.add_scalar('train/lr', current_lr, epoch * len(train_loader) + _iter)
        
        # 에포크별 스케줄러 업데이트 (MultiStep 등)
        if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' not in str(type(scheduler)):
            scheduler.step()

        ####################################################################################################
        
        # 에포크 시간 측정
        epoch_time = time.perf_counter() - epoch_start
        epoch_time_meter.update(epoch_time, n=1)
        
        (f"⏱️  Epoch {epoch + 1} 완료: {epoch_time:.1f}초 소요")
        
        # 검증 실행
        print(f"🔍 검증 시작...")
        metrics = validate(cfg, model, val_loader, loss_gls, epoch)

        ####################################################################################################
        
        # ReduceLROnPlateau 스케줄러 사용 시
        if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in str(type(scheduler)):
            scheduler.step(metrics['wer'])  # WER 기준으로 스케줄링
        
        ####################################################################################################
        
        # TensorBoard 로깅
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        # 로그 출력
        logger.info(f"Epoch {epoch + 1}: Val Loss={metrics['loss']:.3f}, Val WER={metrics['wer']:.3f}")
        print(f"✅ Epoch {epoch + 1}: Val Loss={metrics['loss']:.3f}, Val WER={metrics['wer']:.3f}")
        
        # 최고 성능 업데이트
        is_best = metrics["wer"] < best_wer
        if is_best:
            print(f"🎉 새로운 최고 성능! since {epoch - last_best_idx + 1} epoch. \nWER: {best_wer:.3f} → {metrics['wer']:.3f}")
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
        
        
        # 조기 종료 조건 (선택사항)
        if best_wer < 5.0:  # WER 5% 이하 달성 시
            print(f"🎯 목표 성능 달성! WER: {best_wer:.3f}% < 5.0%")
            break

    ####################################################################################################
    
    writer.close()
    print(f"🏁 학습 완료! 최종 Best WER: {best_wer:.3f}")

####################################################################################################

def validate(cfg, model, val_loader, criterion, epoch) -> dict:
    """검증 함수 - 시간 구간별 수화 인식용으로 개선"""
    logger = logging.getLogger()
    model.eval()
    val_loss_meter = AverageMeter()
    all_hypotheses, all_references = [], []
    vocab = val_loader.dataset.vocab
    
    print(f"🔍 검증 시작: {len(val_loader)} 배치")
    val_pbar = tqdm(val_loader, desc = f"Epoch {epoch + 1} / Valid", mininterval=1)
    
    with torch.no_grad():
        # for batch_idx, batch in enumerate(val_loader):
        for batch_idx, batch in enumerate(val_pbar):
            torch.cuda.empty_cache()
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch
            
            # 빈 시퀀스 필터링
            valid_indices = (keypoint_lengths > 0) & (gloss_lengths > 0)
            if not valid_indices.any():
                continue
            
            keypoints = keypoints[valid_indices]
            keypoint_lengths = keypoint_lengths[valid_indices]
            glosses = glosses[valid_indices]
            gloss_lengths = gloss_lengths[valid_indices]
            
            # GPU로 이동
            keypoints = keypoints.to(cfg.GPU_ID, non_blocking=True)
            glosses = glosses.to(cfg.GPU_ID, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(cfg.GPU_ID, non_blocking=True)
            gloss_lengths = gloss_lengths.to(cfg.GPU_ID, non_blocking=True)
            
            # NaN 처리
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]
            # print(f"valid ind : {valid_indices}")
            try:
                # 모델 추론
                gloss_scores = model(keypoints, src_key_padding_mask=src_key_padding_mask)
                
                # 손실 계산
                gloss_probs = F.log_softmax(gloss_scores, dim=-1).permute(1, 0, 2)
                loss = criterion(gloss_probs, glosses, keypoint_lengths.long(), gloss_lengths.long())
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss_meter.update(loss.item(), n=keypoints.size(0))
                
                # CTC Greedy Decoding 개선
                predictions = torch.argmax(gloss_scores, dim=-1)
                predictions_cpu = predictions.cpu().numpy()
                
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
                            # with val_pbar.external_write_mode():
                            #     print(f"decoded_indices : {decoded_indices}")
                            # val_pbar.refresh()
                            hypothesis = vocab.arrays_to_sentences([decoded_indices])[0]
                            all_hypotheses.append(hypothesis)
                        except Exception as e:
                            print(f"⚠️  디코딩 오류: {e}")
                            all_hypotheses.append([])
                    else:
                        all_hypotheses.append([])
                    
                    # 참조 문장 생성
                    ref_seq = glosses[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
                    reference = vocab.arrays_to_sentences([ref_seq])[0]
                    all_references.append(reference)
                    # print(f"ref_seq : {ref_seq}, reference : {reference}")
                    # print(f"ref_seq : {ref_seq}, reference : {reference}, all_hyp : {list(filter(lambda x : x, all_hypotheses))}")
                
            except Exception as e:
                print(f"❌ Validation error at batch {batch_idx}: {e}")
                continue
            
            # # 진행률 출력
            # if (batch_idx + 1) % 50 == 0:
            #     print(f"검증 진행률: {batch_idx + 1}/{len(val_loader)}")
    
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

####################################################################################################

def save_checkpoint(state_dict, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """체크포인트 저장"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_path)
        print(f"💾 최고 성능 모델 저장: {best_path}")

####################################################################################################

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = default_argument_parser().parse_args()
    args.config_file = '/home/azureuser/abcd/test/NIA_CSLR/configs/config.yaml'
    
    print("🚀 시간 구간별 수화 인식 학습 시작")
    print(f"📁 설정 파일: {args.config_file}")

    main(args)