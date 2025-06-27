import logging
from .vocabulary import build_vocab
from .sign_dataset import SegmentSignDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset

def build_data_loader(cfg):
    """시간 구간별 데이터로더 빌드"""
    logger = logging.getLogger()
    
    # 학습 데이터셋
    train_dataset = SegmentSignDataset(
        data_root=cfg.DATASET.DATA_ROOT + "/train",
        ann_file=cfg.DATASET.TRAIN.ANN_FILE,
        keypoint_prefix=cfg.DATASET.TRAIN.KEYPOINT_PREFIX,
        is_train=True,
        fps=getattr(cfg.DATASET, 'FPS', 30.0),
        max_seq_length=getattr(cfg.DATASET, 'MAX_SEQ_LENGTH', 500),
        min_seq_length=getattr(cfg.DATASET, 'MIN_SEQ_LENGTH', 5)
    )
    # train_dataset_sub = train_dataset
    indices = list(range(len(train_dataset) // 100))
    train_dataset_sub = Subset(train_dataset, indices)
    # 검증 데이터셋
    val_dataset = SegmentSignDataset(
        data_root=cfg.DATASET.DATA_ROOT + "/train",
        ann_file=cfg.DATASET.VAL.ANN_FILE,
        keypoint_prefix=cfg.DATASET.VAL.KEYPOINT_PREFIX,
        is_train=False,
        fps=getattr(cfg.DATASET, 'FPS', 30.0),
        max_seq_length=getattr(cfg.DATASET, 'MAX_SEQ_LENGTH', 500),
        min_seq_length=getattr(cfg.DATASET, 'MIN_SEQ_LENGTH', 5)
    )
    val_dataset_sub = Subset(val_dataset, indices)
    
    # # 어휘 사전 구축 (학습 데이터 기반)
    # vocab = build_vocab(train_dataset, max_size=10000, min_freq=1)

    # # 데이터셋에 어휘 사전 로드
    # train_dataset.load_vocab(vocab)
    # val_dataset.load_vocab(vocab)
    train_dataset_sub.vocab = train_dataset.vocab
    val_dataset_sub.vocab = val_dataset.vocab
    
    # 데이터로더 생성
    train_loader = DataLoader(
        # train_dataset,
        train_dataset_sub,
        batch_size=cfg.SOLVER.BATCH_PER_GPU,
        num_workers=cfg.DATASET.WORKER_PER_GPU,
        collate_fn=train_dataset.collate,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8
    )
    
    val_loader = DataLoader(
        # val_dataset,
        val_dataset_sub,
        batch_size=cfg.SOLVER.BATCH_PER_GPU,
        num_workers=cfg.DATASET.WORKER_PER_GPU,
        collate_fn=val_dataset.collate,
        # collate_fn=val_dataset.collate,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=8
    )
    
    return train_loader, val_loader