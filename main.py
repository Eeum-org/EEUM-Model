import os
import time
import shutil
import logging
import torch
import torch.nn as nn
from itertools import groupby
from torch.utils.tensorboard import SummaryWriter
from lib.config import get_cfg
from lib.dataset import build_data_loader
from lib.engine import default_argument_parser, default_setup
from lib.solver import build_lr_scheduler, build_optimizer
from lib.utils import AverageMeter, clean_ksl, wer_list
from lib.model import KeypointTransformer

best_wer = 100

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
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_data_loader(cfg)

    loss_gls = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # OpenPose 키포인트: 몸(25)+얼굴(70)+양손(21*2) = 137개 * 2차원(x,y) = 274
    input_dim = 274
    num_classes = len(train_loader.dataset.vocab)

    model = KeypointTransformer(num_classes=num_classes, input_dim=input_dim)
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

        # epoch start
        epoch_start = time.perf_counter()
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Best WER so far: {best_wer:.3f}")
        for _iter, batch in enumerate(train_loader):
            start = time.perf_counter()
            
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch
            
            data_time = time.perf_counter() - start
            data_time_meter.update(data_time, n=keypoints.size(0))

            keypoints = keypoints.to(device, non_blocking=True)
            glosses = glosses.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
            # Transformer를 위한 패딩 마스크 생성
            max_len = keypoints.size(1)
            src_key_padding_mask = torch.arange(max_len, device=keypoints.device)[None, :] >= keypoint_lengths[:, None]

            optimizer.zero_grad()
            
            gloss_scores = model(keypoints, src_key_padding_mask=src_key_padding_mask)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)
            
            loss = loss_gls(gloss_probs, glosses, keypoint_lengths.long(), gloss_lengths.long())
            loss_meter.update(loss.item(), n=keypoints.size(0))

            loss.backward()
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

        # end of epoch
        scheduler.step(loss)
        epoch_time = time.perf_counter() - epoch_start
        epoch_time_meter.update(epoch_time, n=1)

        remain = EPOCHS - (epoch + 1)
        writer.add_scalar("misc/eta", remain * epoch_time_meter.avg, epoch)
        writer.flush()

        # validate
        metrics = validate(cfg, model, val_loader, loss_gls)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
            writer.flush()

        logger.info(f"epoch: {epoch + 1}/{EPOCHS} Val loss: {metrics['loss']:.3f} Val WER: {metrics['wer']:.3f}")

        # checkpoint
        is_best = metrics["wer"] < best_wer
        best_wer = min(best_wer, metrics["wer"])
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


def validate(cfg, model, val_loader, criterion) -> dict:
    logger = logging.getLogger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    val_loss_meter = AverageMeter()
    all_hypotheses, all_references = [], []
    vocab = val_loader.dataset.vocab

    for batch in val_loader:
        with torch.no_grad():
            (keypoints, keypoint_lengths), (glosses, gloss_lengths) = batch
            keypoints = keypoints.to(device, non_blocking=True)
            glosses = glosses.to(device, non_blocking=True)
            keypoint_lengths = keypoint_lengths.to(device, non_blocking=True)
            gloss_lengths = gloss_lengths.to(device, non_blocking=True)
            
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
                pred_seq = predictions_cpu[b_idx]
                decoded_indices = [k for k, g in groupby(pred_seq) if k != 0] # 0은 blank 토큰
                all_hypotheses.append(vocab.arrays_to_sentences([decoded_indices])[0])
            
            for b_idx in range(glosses.shape[0]):
                ref_seq = glosses[b_idx][:gloss_lengths[b_idx]].cpu().numpy()
                all_references.append(vocab.arrays_to_sentences([ref_seq])[0])
                
    gls_ref = [clean_ksl(" ".join(t)) for t in all_references]
    gls_hyp = [clean_ksl(" ".join(t)) for t in all_hypotheses]
    
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