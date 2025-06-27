import logging
import sys
from collections import OrderedDict
from random import shuffle


from torch.utils.data import DataLoader, Dataset, Sampler
from yacs.config import CfgNode
from torch.utils.data import Subset
from typing import List

from .sign_dataset import SignDataset
from .transforms import build_transform_gen
from .vocabulary import build_vocab


def tokenize_text(text: str) -> List[str]:
    return text.split()


def build_dataset(cfg: CfgNode) -> Dataset:
    logger = logging.getLogger()

    data_root = cfg.DATASET.DATA_ROOT
    keypoint_train = cfg.DATASET.TRAIN.KEYPOINT_PREFIX
    ann_file_train = cfg.DATASET.TRAIN.ANN_FILE
    ann_file_val = cfg.DATASET.VAL.ANN_FILE
    keypoint_val = cfg.DATASET.VAL.KEYPOINT_PREFIX

    tfm_gens = build_transform_gen(cfg, is_train=False)
    train_dataset = SignDataset(
        data_root + "/train",
        ann_file_train,
        keypoint_dir=keypoint_train,
        tokenize=tokenize_text,
        is_train=True,
    )
    vocab = build_vocab(cfg, train_dataset, sys.maxsize, min_freq=1)



    tfm_gens = build_transform_gen(cfg, is_train=False)

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
    train_sub = Subset(train_dataset, indices = [i for i in range(train_indices // 100)])
    val_sub = Subset(val_dataset, indices = [i for i in range(val_indices // 100)])
    train_sub.vocab = train_dataset.vocab
    val_sub.vocab = val_dataset.vocab
    
    # def gpu_collate_fn(batch):
    #     """배치를 GPU로 이동하는 collate function"""
    #     videos, glosses = list(zip(*batch))
        
    #     # 기존 collate 로직 적용
    #     (videos, video_lengths), (glosses, gloss_lengths) = train_dataset.collate(batch)
        
    #     # GPU로 이동
    #     device = f'cuda:{cfg.GPU_ID}' if isinstance(cfg.GPU_ID, int) else cfg.GPU_ID
    #     videos = videos.to(device)
    #     video_lengths = video_lengths.to(device)
    #     glosses = glosses.to(device)
    #     gloss_lengths = gloss_lengths.to(device)
        
    #     return (videos, video_lengths), (glosses, gloss_lengths)
    
    # multiple data loaders
    train_loader = DataLoader(
        dataset=train_sub, #train_dataset,
        collate_fn= train_dataset.collate,
        # batch_sampler=BucketBatchSampler(
        #     [example["frames"] for example in train_dataset.examples], batch_per_gpu * len(GPU_ID)
        # ),
        batch_size = batch_per_gpu * len(GPU_ID),
        shuffle=False,
        drop_last=False,
        num_workers=0, #worker_per_gpu * len(GPU_ID),
        # multiprocessing_context='spawn'
    )
    val_loader = DataLoader(
        dataset=val_sub, #val_dataset,
        collate_fn= val_dataset.collate,
        batch_size=batch_per_gpu * len(GPU_ID),
        shuffle=False,
        drop_last=False,
        num_workers=0, #worker_per_gpu * len(GPU_ID),
        # multiprocessing_context='spawn'
    )

    return train_loader, val_loader