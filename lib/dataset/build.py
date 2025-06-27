import sys
import logging
from typing import List
from yacs.config import CfgNode
from .vocabulary import build_vocab
from .sign_dataset import SignDataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset

def tokenize_text(text: str) -> List[str]:
    return text.split()

def build_dataset(cfg: CfgNode) -> Dataset:
    logger = logging.getLogger()

    data_root = cfg.DATASET.DATA_ROOT
    keypoint_train = cfg.DATASET.TRAIN.KEYPOINT_PREFIX
    ann_file_train = cfg.DATASET.TRAIN.ANN_FILE
    ann_file_val = cfg.DATASET.VAL.ANN_FILE
    keypoint_val = cfg.DATASET.VAL.KEYPOINT_PREFIX

    train_dataset = SignDataset(
        data_root + "/train",
        ann_file_train,
        keypoint_dir=keypoint_train,
        tokenize=tokenize_text,
        is_train=True,
    )
    vocab = build_vocab(cfg, train_dataset, sys.maxsize, min_freq=1)

    val_dataset = SignDataset(
        data_root + "/val",
        ann_file_val,
        keypoint_dir=keypoint_val,
        tokenize=tokenize_text,
        is_train=False,
    )
 
    # load vocabulary to dataset
    train_dataset.load_vocab(vocab)
    val_dataset.load_vocab(vocab)
    print()
    logger.info(
        "{} examples for Train, {} examples for Valid. Number of Vocabulary: {}".format(
            len(train_dataset), len(val_dataset), len(vocab.stoi)
        )
    )
    print()

    return train_dataset, val_dataset

def build_data_loader(cfg) -> DataLoader:
    batch_per_gpu = cfg.SOLVER.BATCH_PER_GPU
    worker_per_gpu = cfg.DATASET.WORKER_PER_GPU
    GPU_ID = cfg.GPU_ID
    if not isinstance(GPU_ID, list):
        GPU_ID = [GPU_ID]

    train_dataset, val_dataset = build_dataset(cfg)
    train_indices = len(train_dataset)
    val_indices = len(val_dataset)
    train_sub = Subset(train_dataset, indices = [i for i in range(train_indices // 10)])
    val_sub = Subset(val_dataset, indices = [i for i in range(val_indices // 10)])
    train_sub.vocab = train_dataset.vocab
    val_sub.vocab = val_dataset.vocab

    # multiple data loaders
    train_loader = DataLoader(
        dataset=train_sub, #train_dataset,
        collate_fn= train_dataset.collate_fn,
        batch_size = batch_per_gpu * len(GPU_ID),
        shuffle=False,
        drop_last=False,
        num_workers=0, #worker_per_gpu * len(GPU_ID),
        # multiprocessing_context='spawn'
    )
    val_loader = DataLoader(
        dataset=val_sub, #val_dataset,
        collate_fn= val_dataset.collate_fn,
        batch_size=batch_per_gpu * len(GPU_ID),
        shuffle=False,
        drop_last=False,
        num_workers=0, #worker_per_gpu * len(GPU_ID),
        # multiprocessing_context='spawn'
    )

    return train_loader, val_loader