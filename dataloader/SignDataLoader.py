import os
import json
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from preprocess.dataset import SignDataset


class SignDataSplitter:
    """
    Manages intelligent dataset splitting with vocabulary preservation
    """
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.dataset.get('data_root', './data')
        self.split_ratios = {
            'train': config.dataset.split_ratios.get('train', 0.8),
            'val': config.dataset.split_ratios.get('val', 0.2)
        }
        self.min_vocab_count = config.dataset.get('min_vocab_count', 2)
        self.directions = config.dataset.get('directions', ['F', 'L', 'R', 'U', 'D'])
        
    def scan_morpheme_datasets(self, split_name: str) -> List[Dict]:
        """
        새로운 구조에서 morpheme 기준으로 데이터 스캔
        Structure: data/{split}/morpheme/{dataset_name}_morpheme/
        """
        morpheme_base_path = os.path.join(self.data_root, split_name, 'morpheme')
        
        if not os.path.exists(morpheme_base_path):
            print(f"❌ Morpheme directory not found: {morpheme_base_path}")
            return []
        
        samples = []
        
        # morpheme 하위 데이터셋들 순회
        for dataset_dir in os.listdir(morpheme_base_path):
            dataset_path = os.path.join(morpheme_base_path, dataset_dir)
            if os.path.isdir(dataset_path) and dataset_dir.endswith('_morpheme'):
                dataset_name = dataset_dir.replace('_morpheme', '')
                
                # morpheme 파일들 처리
                for morpheme_file in os.listdir(dataset_path):
                    if morpheme_file.endswith('_morpheme.json'):
                        morpheme_path = os.path.join(dataset_path, morpheme_file)
                        dataset_samples = self._process_morpheme_file(morpheme_path, dataset_name, split_name)
                        samples.extend(dataset_samples)
        
        print(f"📊 Scanned {len(samples)} samples from {split_name}")
        return samples
    
    def _process_morpheme_file(self, morpheme_path: str, dataset_name: str, split_name: str) -> List[Dict]:
        """morpheme 파일 처리 및 keypoint 매칭"""
        try:
            with open(morpheme_path, 'r', encoding='utf-8') as f:
                morpheme_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Error reading {morpheme_path}: {e}")
            return []
        
        # morpheme 추출
        morphemes = []
        for item in morpheme_data.get('data', []):
            morpheme_name = item.get('attributes', [{}])[0].get('name', '')
            if morpheme_name:
                morphemes.append(morpheme_name)
        
        if not morphemes:
            return []
        
        # 기본 video ID
        base_video_id = os.path.basename(morpheme_path).replace('_morpheme.json', '')
        
        samples = []
        
        # 방향별로 keypoint 확인
        for direction in self.directions:
            # 방향 정보가 포함된 video ID 생성
            if f"_{direction}_" in base_video_id or base_video_id.endswith(f"_{direction}"):
                video_id = base_video_id
            else:
                parts = base_video_id.split("_")
                parts.insert(-1, direction)
                video_id = "_".join(parts)
            
            # keypoint 경로 확인
            keypoint_dir = os.path.join(self.data_root, split_name, 'keypoint', f"{dataset_name}_keypoint", video_id)
            
            if os.path.exists(keypoint_dir):
                keypoint_files = [f for f in os.listdir(keypoint_dir) if f.endswith('_keypoints.json')]
                if keypoint_files:
                    keypoint_files.sort(key=lambda x: int(x.split('_')[-2]))
                    
                    samples.append({
                        'video_id': video_id,
                        'direction': direction,
                        'keypoint_dir': keypoint_dir,
                        'keypoint_files': keypoint_files,
                        'morphemes': morphemes,
                        'base_name': base_video_id,
                        'dataset_name': dataset_name
                    })
        
        return samples
    
    def create_splits_from_train(self, vocab, save_annotations: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        train 데이터를 train/val로 분할
        """
        print("🔀 Creating train/val split from train data...")
        
        # train 데이터 스캔
        samples = self.scan_morpheme_datasets('train')
        
        if not samples:
            raise ValueError("No training samples found")
        
        # 어휘 분포 분석
        self._analyze_vocabulary_distribution(samples)
        
        # train/val 분할
        train_samples, val_samples = self._intelligent_train_val_split(
            samples=samples,
            train_ratio=self.split_ratios['train'],
            val_ratio=self.split_ratios['val'],
            min_vocab_count=self.min_vocab_count
        )
        
        # 분할 결과 검증
        self._validate_splits(train_samples, val_samples)
        
        # 어노테이션 저장
        if save_annotations:
            self._save_split_annotations(train_samples, val_samples)
        
        return train_samples, val_samples
    
    def _analyze_vocabulary_distribution(self, samples: List[Dict]):
        """어휘 분포 분석"""
        print("\n📊 Analyzing vocabulary distribution...")
        
        vocab_counter = Counter()
        direction_counter = Counter()
        data_type_counter = Counter()
        
        for sample in samples:
            # 어휘 빈도
            morphemes_key = tuple(sample['morphemes'])
            vocab_counter[morphemes_key] += 1
            
            # 방향 분포
            direction_counter[sample['direction']] += 1
            
            # 데이터 타입 추정 (파일명 기반)
            video_id = sample['video_id']
            if 'SYN' in video_id:
                data_type_counter['synthetic'] += 1
            elif 'REAL' in video_id:
                data_type_counter['crowd_sourcing'] += 1
            else:
                data_type_counter['studio'] += 1
        
        print(f"   Unique vocabularies: {len(vocab_counter)}")
        print(f"   Most frequent vocab: {vocab_counter.most_common(3)}")
        print(f"   Direction distribution: {dict(direction_counter)}")
        print(f"   Data type distribution: {dict(data_type_counter)}")
        
        # 어휘별 샘플 수 분포
        sample_counts = list(vocab_counter.values())
        print(f"   Vocab sample count - Min: {min(sample_counts)}, Max: {max(sample_counts)}, Avg: {sum(sample_counts)/len(sample_counts):.1f}")
        
        # 최소 요구 개수보다 적은 어휘들
        insufficient_vocabs = [vocab for vocab, count in vocab_counter.items() if count < self.min_vocab_count]
        if insufficient_vocabs:
            print(f"   ⚠️  {len(insufficient_vocabs)} vocabularies have insufficient samples (< {self.min_vocab_count})")
    
    def _validate_splits(self, train_samples: List[Dict], val_samples: List[Dict]):
        """분할 결과 검증"""
        print("\n🔍 Validating dataset splits...")
        
        # 어휘 교집합 확인
        train_vocabs = set(tuple(s['morphemes']) for s in train_samples)
        val_vocabs = set(tuple(s['morphemes']) for s in val_samples)
        
        train_val_overlap = train_vocabs & val_vocabs
        
        print(f"   Train vocabulary count: {len(train_vocabs)}")
        print(f"   Val vocabulary count: {len(val_vocabs)}")
        print(f"   Train-Val vocabulary overlap: {len(train_val_overlap)} ({len(train_val_overlap)/len(train_vocabs)*100:.1f}%)")
        
        # 방향 분포 확인
        for split_name, split_samples in [('Train', train_samples), ('Val', val_samples)]:
            direction_dist = Counter(s['direction'] for s in split_samples)
            print(f"   {split_name} direction distribution: {dict(direction_dist)}")
        
    
    def _save_split_annotations(self, train_samples: List[Dict], val_samples: List[Dict]):
        """분할 어노테이션 저장"""
        print("\n💾 Saving split annotations...")
        
        annotations_dir = os.path.join(self.data_root, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        
        for split_name, split_samples in [('train', train_samples), ('val', val_samples)]:
            annotations = {
                'split_info': {
                    'split_name': split_name,
                    'total_samples': len(split_samples),
                    'creation_timestamp': str(pd.Timestamp.now()),
                    'split_ratios': self.split_ratios,
                    'min_vocab_count': self.min_vocab_count
                },
                'samples': []
            }
            
            for sample in split_samples:
                annotations['samples'].append({
                    'video_id': sample['video_id'],
                    'direction': sample['direction'],
                    'base_name': sample['base_name'],
                    'morphemes': sample['morphemes'],
                    'keypoint_files_count': len(sample['keypoint_files'])
                })
            
            annotation_path = os.path.join(annotations_dir, f'{split_name}_annotations.json')
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            print(f"   Saved {split_name} annotations: {annotation_path}")
    
    def _intelligent_train_val_split(self, samples: List[Dict], 
                                   train_ratio: float = 0.8, 
                                   val_ratio: float = 0.2,
                                   min_vocab_count: int = 2) -> Tuple[List[Dict], List[Dict]]:
        """
        5개 방향 데이터를 고려한 지능적 train/val 분할
        동일 단어의 다른 방향들을 적절히 분배
        """
        assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
        
        # 단어(어휘)별 그룹화 (base_name 기준)
        vocab_groups = defaultdict(lambda: defaultdict(list))
        for i, sample in enumerate(samples):
            vocab_key = tuple(sample['morphemes'])
            direction = sample['direction']
            vocab_groups[vocab_key][direction].append(i)
        
        print(f"\n📊 단어별 방향 분포 분석:")
        
        train_indices, val_indices = [], []
        
        for vocab_key, direction_dict in vocab_groups.items():
            total_samples = sum(len(indices) for indices in direction_dict.values())
            
            if total_samples < min_vocab_count:
                print(f"   ⚠️ 어휘 {vocab_key}: 샘플 수({total_samples}) < 최소 요구수({min_vocab_count})")
                continue
            
            # 방향별 인덱스 모음
            all_indices = []
            for direction, indices in direction_dict.items():
                all_indices.extend(indices)
            
            random.shuffle(all_indices)
            
            # train/val 분할 비율 계산
            n_total = len(all_indices)
            n_train = max(1, int(n_total * train_ratio))
            n_val = n_total - n_train
            
            # 방향 균형 유지를 위한 전략
            if len(direction_dict) >= 3:  # 3개 이상 방향이 있으면 균등 분배
                train_indices.extend(all_indices[:n_train])
                val_indices.extend(all_indices[n_train:])
            else:  # 적은 방향의 경우 비율 분배
                train_indices.extend(all_indices[:n_train])
                val_indices.extend(all_indices[n_train:])
            
            print(f"   어휘 {vocab_key}: {len(direction_dict)}방향, 총 {n_total}샘플 -> Train:{n_train}, Val:{n_val}")
        
        # 인덱스를 실제 샘플로 변환
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        
        print(f"\n🎯 분할 완료:")
        print(f"   Train: {len(train_samples)} 샘플 ({len(train_samples)/len(samples)*100:.1f}%)")
        print(f"   Val: {len(val_samples)} 샘플 ({len(val_samples)/len(samples)*100:.1f}%)")
        
        return train_samples, val_samples


class SignDataLoader:
    """
    크기 조정 가능한 데이터로더 관리자
    """
    
    def __init__(self, dataset_class, config):
        self.dataset_class = dataset_class
        self.config = config
        
    def create_dataloader(self, split_name: str, samples: List[Dict], vocab, 
                         dataset_size: Optional[int] = None, **kwargs):
        """
        새로운 구조에서 샘플 리스트 기반 데이터로더 생성
        """
        from torch.utils.data import DataLoader
        from preprocess.dataset import collate_fn
        
        # 캐시 디렉토리 설정
        cache_dir = None
        if self.config.preprocessing.get('cache_preprocessed', False):
            cache_dir = self.config.preprocessing.get('cache_dir', './cache')
        
        dataset = self.dataset_class(
            samples=samples,
            vocab=vocab,
            config=self.config.get('PREPROCESSING', {}),
            dataset_size=dataset_size,
            cache_dir=cache_dir
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.dataset.get('batch_size', 16),
            collate_fn=lambda b: collate_fn(b, vocab),
            shuffle=kwargs.get('shuffle', False),
            num_workers=self.config.dataset.get('num_workers', 0),
            persistent_workers=self.config.dataset.get('persistent_workers', False),
            pin_memory=self.config.dataset.get('pin_memory', False)
        )
        
        print(f"🔄 Created {split_name} dataloader with {len(dataset)} samples")
        return dataloader, dataset
    
    def create_progressive_dataloaders(self, split_name: str, samples: List[Dict], 
                                     vocab, size_schedule: List[int], **kwargs):
        """
        점진적 크기 증가 데이터로더들 생성
        """
        dataloaders = []
        
        for size in size_schedule:
            dataloader, dataset = self.create_dataloader(
                split_name=split_name,
                samples=samples,
                vocab=vocab,
                dataset_size=size,
                **kwargs
            )
            dataloaders.append((size, dataloader, dataset))
        
        print(f"📈 Created {len(dataloaders)} progressive dataloaders with sizes: {size_schedule}")
        return dataloaders


# Pandas import for timestamp
try:
    import pandas as pd
except ImportError:
    # Fallback if pandas is not available
    from datetime import datetime
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                return datetime.now()