import os
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any

def validate_model_compatibility(config: DictConfig) -> DictConfig:
    """
    Validate and adjust model configuration based on compatibility requirements
    """
    model_name = config.model.get('name', 'MSKA_Model')
    
    if model_name == 'CustomModel':
        print("🔧 Applying CustomModel compatibility adjustments...")
        
        # Check preprocessing compatibility
        if 'compatible_preprocessing' in config.model:
            compat_preproc = config.model.compatible_preprocessing
            
            # Update preprocessing settings to be compatible
            if config.preprocessing.hand_scale != compat_preproc.get('hand_scale', config.preprocessing.hand_scale):
                print(f"   Adjusting hand_scale: {config.preprocessing.hand_scale} -> {compat_preproc.hand_scale}")
                config.preprocessing.hand_scale = compat_preproc.hand_scale
            
            if config.preprocessing.shoulder_width_scale != compat_preproc.get('shoulder_width_scale', config.preprocessing.shoulder_width_scale):
                print(f"   Adjusting shoulder_width_scale: {config.preprocessing.shoulder_width_scale} -> {compat_preproc.shoulder_width_scale}")
                config.preprocessing.shoulder_width_scale = compat_preproc.shoulder_width_scale
        
        # Check vocab compatibility
        if 'compatible_vocab' in config.model:
            compat_vocab = config.model.compatible_vocab
            
            # Create vocab section if it doesn't exist
            if 'vocab' not in config:
                config.vocab = {}
            
            for key, value in compat_vocab.items():
                if key not in config.vocab:
                    config.vocab[key] = value
                    print(f"   Adding vocab setting: {key} = {value}")
    
    return config


def save_config(config: DictConfig, output_path: str):
    """
    Save configuration to file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    OmegaConf.save(config, output_path)
    print(f"Configuration saved to {output_path}")


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """
    Merge configuration with overrides
    """
    override_conf = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_config, override_conf)
    return merged


def get_device_config(config: DictConfig):
    """
    Get device configuration
    """
    import torch
    
    use_cuda = config.device.get('use_cuda', True)
    gpu_id = config.device.get('gpu_id', 0)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"🖥️  Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("🖥️  Using CPU")
    
    return device


def create_output_dir(config: DictConfig) -> str:
    """
    Create output directory based on configuration
    """
    output_dir = config.output.get('dir', './outputs')
    
    # Add experiment name if specified
    if 'experiment' in config and 'name' in config.experiment:
        output_dir = os.path.join(output_dir, config.experiment.name)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    return output_dir


def print_config_summary(config: DictConfig):
    """
    Print configuration summary
    """
    print("\n" + "="*50)
    print("📋 CONFIGURATION SUMMARY")
    print("="*50)
    
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.data_root}")
    print(f"Batch size: {config.dataset.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Output dir: {config.output.dir}")
    
    if 'experiment' in config:
        print(f"Experiment: {config.experiment.get('name', 'N/A')}")
    
    print("="*50 + "\n")