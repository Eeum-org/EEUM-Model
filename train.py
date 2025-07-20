#!/usr/bin/env python3
import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.MSKA import MSKA_Model
from trainer.trainer import MSKATrainer
from preprocess.dataset import SignDataset
from dataloader.SignDataLoader import SignDataSplitter, SignDataLoader
from utils.vocabulary import load_or_build_vocab
from utils.settings import validate_model_compatibility, get_device_config, create_output_dir, print_config_summary

def load_config(config_path: str = "configs/config.yaml") -> DictConfig:
    """Load configuration from file"""
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        print(f"📁 Loaded config from {config_path}")
        return config
    else:
        print(f"❌ Config file not found: {config_path}")
        # Return minimal default config
        return OmegaConf.create({
            'model': {'name': 'MSKA_Model', 'input_dim': 274, 'd_model': 512},
            'dataset': {'data_root': './data', 'batch_size': 16},
            'training': {'epochs': 10, 'learning_rate': 1e-4},
            'output': {'dir': './outputs'},
            'device': {'use_cuda': True, 'gpu_id': 0}
        })

def main(config_path: str = "configs/config.yaml") -> None:
    """Main training function"""
    config = load_config(config_path)
    # Validate and adjust configuration for model compatibility
    config = validate_model_compatibility(config)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Save resolved configuration
    output_dir = create_output_dir(config)
    config_save_path = os.path.join(output_dir, 'resolved_config.yaml')
    OmegaConf.save(config, config_save_path)
    
    # Setup device
    device = get_device_config(config)
    
    # Check available splits first
    data_splitter = SignDataSplitter(config)
    train_samples = data_splitter.scan_morpheme_datasets('train')
    val_samples = data_splitter.scan_morpheme_datasets('val')
    test_samples = data_splitter.scan_morpheme_datasets('test') if config.get('use_test', False) else []
    
    # Collect all morphemes from all splits
    all_morphemes = set()
    for samples in [train_samples, val_samples, test_samples]:
        for sample in samples:
            all_morphemes.update(sample['morphemes'])
    
    print(f"🔍 Found {len(all_morphemes)} unique morphemes across all splits")
    
    # Build or load vocabulary
    vocab_path = os.path.join(output_dir, config.vocab.get('save_path', 'vocabulary.json'))
    morpheme_dir = os.path.join(config.dataset.data_root, config.dataset.train.morpheme_dir)
    
    vocab = load_or_build_vocab(
        vocab_path=vocab_path,
        morpheme_dir=morpheme_dir,
        force_rebuild=config.vocab.get('force_rebuild', False)
    )
    
    # Expand vocab with any missing morphemes
    missing_morphemes = [token for token in all_morphemes if token not in vocab.stoi]
    if missing_morphemes:
        print(f"🔧 Expanding vocabulary with {len(missing_morphemes)} new morphemes")
        vocab.expand_vocab(missing_morphemes)
        vocab.save(vocab_path)  # Save updated vocab
    
    print(f"📚 Final vocabulary size: {len(vocab)}")
    
    # Determine if we need to split train data
    need_split = len(val_samples) == 0 and config.get('auto_split', True)
    
    if need_split:
        print("🔀 No validation data found. Creating train/val split...")
        train_samples, val_samples = data_splitter.create_splits_from_train(vocab, save_annotations=True)
    else:
        print("📁 Using existing train/val splits...")
    
    # Create data loader manager
    loader_manager = SignDataLoader(SignDataset, config)
    
    # Create dataloaders
    train_loader, train_dataset = loader_manager.create_dataloader(
        'train', train_samples, vocab,
        dataset_size=config.dataset.get('train_size', None),
        shuffle=True
    )
    
    val_loader, val_dataset = loader_manager.create_dataloader(
        'val', val_samples, vocab,
        dataset_size=config.dataset.get('val_size', None),
        shuffle=False
    )
    
    # Test dataset (if requested and available)
    test_loader = None
    if test_samples:
        test_loader, test_dataset = loader_manager.create_dataloader(
            'test', test_samples, vocab,
            dataset_size=config.dataset.get('test_size', None),
            shuffle=False
        )
    
    print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    if test_loader:
        print(f"                  Test: {len(test_dataset)}")
    
    # Create model
    model = MSKA_Model(
        num_classes=len(vocab),
        input_dim=config.model.get('input_dim', 274),
        d_model=config.model.get('d_model', 1024),
        nhead=config.model.get('nhead', 8),
        num_encoder_layers=config.model.get('num_encoder_layers', 10),
        num_decoder_layers=config.model.get('num_decoder_layers', 8),
        dim_feedforward=config.model.get('dim_feedforward', 2048),
        dropout=config.model.get('dropout', 0.2)
    ).to(device)
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.get('learning_rate', 1e-4),
        weight_decay=config.training.get('weight_decay', 1e-4)
    )
    
    # Create scheduler
    scheduler = None
    if 'scheduler' in config and config.scheduler.get('name') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.get('mode', 'min'),
            factor=config.scheduler.get('factor', 0.5),
            patience=config.scheduler.get('patience', 5)
        )
    
    # Create trainer with legacy config wrapper for compatibility
    
    trainer = MSKATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        vocab=vocab,
        config=config,
        device=device
    )

    # Training mode
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if config.get('checkpoint'):
        print(f"🔄 Resuming training from checkpoint: {config.checkpoint}")
        checkpoint = trainer.load_checkpoint(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        trainer.best_wer = checkpoint.get('wer', float('inf'))
    
    # Progressive training with increasing dataset sizes (if configured)
    if config.get('progressive_training', {}).get('enabled', False):
        size_schedule = config.progressive_training.get('size_schedule', [1000, 5000, 10000, -1])  # -1 means full dataset
        
        print(f"📈 Starting progressive training with schedule: {size_schedule}")
        
        for stage, target_size in enumerate(size_schedule):
            print(f"\n🎯 Progressive training stage {stage + 1}: size {target_size}")
            
            if target_size == -1:
                # Use full dataset
                current_train_loader = train_loader
            else:
                # Create smaller dataset
                current_train_loader, _ = loader_manager.create_dataloader(
                    'train', train_samples, vocab,
                    dataset_size=target_size,
                    shuffle=True
                )
            
            # Update trainer with new dataloader
            trainer.train_loader = current_train_loader
            
            # Train for specified epochs for this stage
            stage_epochs = config.progressive_training.get('epochs_per_stage', 10)
            end_epoch = start_epoch + stage_epochs
            
            # Temporarily update trainer epochs
            original_epochs = trainer.epochs
            trainer.epochs = end_epoch - 1
            
            training_history = trainer.train(start_epoch=start_epoch)
            start_epoch = end_epoch
            
            # Restore original epochs for final stage
            trainer.epochs = original_epochs
    
    else:
        # Standard training
        print("🚀 Starting standard training...")
        training_history = trainer.train(start_epoch=start_epoch)
    
    print("✅ Training completed!")
    print(f"🏆 Best WER: {training_history['best_wer']:.2f}% at epoch {training_history['best_epoch']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MSKA Model Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)