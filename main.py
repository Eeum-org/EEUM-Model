import os
import time
import shutil
import logging
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from itertools import groupby
from torch.utils.tensorboard import SummaryWriter
from lib.config import get_cfg
from lib.dataset import build_data_loader
from lib.engine import default_argument_parser, default_setup
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl, wer_list
from lib.model import KeypointTransformer

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
                print(f"  ⚠️ 모든 가중치가 동일함")
            
            print()
def check_gradient_flow(model, loss):
    """그래디언트 흐름 확인"""
    print("=== 그래디언트 흐름 확인 ===")
    
    # 역전파 수행
    loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)

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
    
    # 모델 구조 확인
    print("모델 구조:")
    print(model)
    
    # 가중치 초기화 확인
    check_weight_initialization(model)
    
    # Forward pass 테스트
    output = debug_forward_pass(model, sample_input)
    
    # Loss 계산 및 그래디언트 확인
    if sample_target is not None:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        try:
            print(f"shape : {sample_target.shape}, {sample_input.shape}")
            loss = criterion(output.log_softmax(dim=-1).permute(1, 0, 2), 
                           sample_target, 
                           torch.tensor([output.size(1)]), 
                           torch.tensor([len(sample_target)]))
            print(f"Loss 계산 성공: {loss.item():.4f}")
            check_gradient_flow(model, loss)
        except Exception as e:
            print(f"⚠️ Loss 계산 실패: {e}")
    
    print("=" * 50)
    print("END")
    print("=" * 50)

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
    EPOCHS = 80
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

    loss_gls = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # OpenPose 키포인트: 몸(25)+얼굴(70)+양손(21*2) = 137개 * 2차원(x,y) = 274
    input_dim = 274
    num_classes = len(train_loader.dataset.vocab)
    
    # 모델 하이퍼파라미터 변경
    model = KeypointTransformer(
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

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        if is_gpu:
            torch.cuda.empty_cache()
        # epoch start
        epoch_start = time.perf_counter()
        train_pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1} / Train", mininterval=1)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Best WER so far: {best_wer:.3f}")
        for _iter, batch in enumerate(train_pbar):
            start = time.perf_counter()
            
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch
            data_time = time.perf_counter() - start
            data_time_meter.update(data_time, n=keypoints.size(0))

            keypoints = keypoints.to(device, non_blocking=True)
            glosses = glosses.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
            # NaN 처리 추가
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)

            # Transformer를 위한 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]

            optimizer.zero_grad()
            
            gloss_scores = model(keypoints, src_key_padding_mask=src_key_padding_mask)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)

            loss = loss_gls(gloss_probs, glosses, keypoint_lengths.long(), gloss_lengths.long())
            loss_meter.update(loss.item(), n=keypoints.size(0))

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            iter_time = time.perf_counter() - start
            iter_time_meter.update(iter_time, n=keypoints.size(0))

            current_lr = optimizer.param_groups[0]["lr"]

            # log
            if (_iter + 1) % cfg.PERIODS.LOG_ITERS == 0:
                current_iter = epoch * len(train_loader) + _iter
                logger.info(
                    "epoch: {}/{}, iter: {}/{}  "
                    "loss: {loss.val:.3f} (avg: {loss.avg:.3f})  "
                    "iter_time: {iter_time.val:.3f} (avg: {iter_time.avg:.3f})  "
                    "data_time: {data_time.val:.3f} (avg: {data_time.avg:.3f})  "
                    "lr: {lr:.5f}".format(
                        epoch + 1,
                        EPOCHS,
                        _iter + 1,
                        len(train_loader),
                        loss=loss_meter,
                        iter_time=iter_time_meter,
                        data_time=data_time_meter,
                        lr=current_lr
                    )
                )

                if (_iter + 1) % cfg.PERIODS.LOG_ITERS == 0:
                    # write log
                    writer.add_scalar("misc/data_time", data_time_meter.avg, current_iter)
                    writer.add_scalar("misc/iter_time", iter_time_meter.avg, current_iter)
                    writer.add_scalar("train/loss", loss_meter.avg, current_iter)
                    writer.add_scalar("misc/lr", current_lr, current_iter)
                    writer.flush()

                    data_time_meter.reset()
                    iter_time_meter.reset()
                    loss_meter.reset()
            train_pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        # end of epoch
        scheduler.step(loss)
        epoch_time = time.perf_counter() - epoch_start
        epoch_time_meter.update(epoch_time, n=1)

        remain = EPOCHS - (epoch + 1)
        writer.add_scalar("misc/eta", remain * epoch_time_meter.avg, epoch)
        writer.flush()

        # validate
        metrics = validate(cfg, model, val_loader, loss_gls, epoch)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
            writer.flush()

        logger.info(f"epoch: {epoch + 1}/{EPOCHS} Val loss: {metrics['loss']:.3f} Val WER: {metrics['wer']:.3f}")
        print(f"epoch: {epoch + 1}/{EPOCHS} loss: {loss_meter.avg:.3f} Val loss: {metrics['loss']:.3f} Val WER: {metrics['wer']:.3f}, lr : {current_lr:.1e}")
        # checkpoint
        is_best = metrics["wer"] < best_wer
        best_wer = min(best_wer, metrics["wer"])
        if is_best:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'wer': metrics["wer"],
                    'best_wer': best_wer,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, is_best, cfg.OUTPUT_DIR, f"checkpoint.epoch_{epoch + 1}.tar"
            )


def validate(cfg, model, val_loader, criterion, epoch) -> dict:
    logger = logging.getLogger()
    is_gpu = torch.cuda.is_available()
    device = "cuda" if is_gpu else "cpu"
    model.eval()
    val_loss_meter = AverageMeter()
    all_hypotheses, all_references = [], []
    vocab = val_loader.dataset.vocab
    val_pbar = tqdm(val_loader, desc = f"Epoch {epoch + 1} / Valid", mininterval=1)
    for iter, batch in enumerate(val_pbar):
        if is_gpu:
            torch.cuda.empty_cache()
        with torch.no_grad():
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch
            keypoints = keypoints.to(device, non_blocking=True)
            glosses = glosses.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
            # NaN 처리
            keypoints = torch.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=-1.0)

            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]

            gloss_scores = model(keypoints, src_key_padding_mask=src_key_padding_mask)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)
            
            loss = criterion(gloss_probs, glosses, keypoint_lengths.long(), gloss_lengths.long())
            val_loss_meter.update(loss.item(), n=keypoints.size(0))

            # CTC Greedy Decoding
            predictions = torch.argmax(gloss_scores, dim=-1)
            predictions_cpu = predictions.cpu().numpy()
            
            for b_idx in range(predictions_cpu.shape[0]):
                pred_seq = predictions_cpu[b_idx][:keypoint_lengths[b_idx].cpu().item()]
                decoded_indices = [k for k, g in groupby(pred_seq) if k != 0] # 0은 blank 토큰
                all_hypotheses.append(vocab.arrays_to_sentences([decoded_indices])[0])
            
            for b_idx in range(glosses.shape[0]):
                ref_seq = glosses[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
                all_references.append(vocab.arrays_to_sentences([ref_seq])[0])
    gls_ref = [clean_ksl(" ".join(t)) for t in all_references]
    gls_hyp = [clean_ksl(" ".join(t)) for t in all_hypotheses]
    print(all_references, all_hypotheses, gls_hyp)
    
    metrics = wer_list(hypotheses=gls_hyp, references=gls_ref)
    metrics.update({"loss": val_loss_meter.avg})
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