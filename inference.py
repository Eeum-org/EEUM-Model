#!/usr/bin/env python3
"""
Complete inference module for MSKA model
Handles both single video inference and batch inference
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.keypoint_extractor import KeypointExtractor
from utils.settings import get_device_config
from preprocess.dataset import collate_fn


class MSKAInference:
    
    def __init__(self, model, vocab, device, config=None):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.config = config
        
        # Inference settings from config or defaults
        if config and hasattr(config, 'inference'):
            self.max_len = config.inference.get('max_len', 15)
            self.eos_boost_threshold = config.inference.get('eos_boost_threshold', 8)
            self.eos_boost_value = config.inference.get('eos_boost_value', 2.0)
        else:
            self.max_len = 15
            self.eos_boost_threshold = 8
            self.eos_boost_value = 2.0
    
    def predict_batch(self, data_loader: DataLoader) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of data
        Returns list of predictions without requiring reference labels
        """
        self.model.eval()
        all_predictions = []
        
        print("🔍 Running inference...")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Inference")):
                # Handle different batch data formats
                if len(batch_data) == 2:
                    # Format: ((keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths))
                    (keypoints, keypoints_lengths), (teacher_input, teacher_target, gloss_lengths) = batch_data
                else:
                    # Simple format: (keypoints, keypoints_lengths)
                    keypoints, keypoints_lengths = batch_data
                
                keypoints = keypoints.to(self.device)
                keypoints_lengths = keypoints_lengths.to(self.device)
                
                # Create source padding mask
                src_key_padding_mask = (
                    torch.arange(keypoints.size(1), device=self.device)[None, :] >= 
                    keypoints_lengths[:, None]
                )
                
                # Generate predictions
                generated_seqs = self.model(
                    keypoints, 
                    tgt=None, 
                    src_key_padding_mask=src_key_padding_mask
                )
                
                # Decode predictions
                predictions = self.vocab.arrays_to_sentences(generated_seqs.cpu().numpy())
                
                # Store results
                for i, pred in enumerate(predictions):
                    prediction_text = ' '.join(pred) if pred else '<EMPTY>'
                    all_predictions.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'predicted_text': prediction_text,
                        'predicted_tokens': pred
                    })
        
        return all_predictions
    
    def predict_single(self, keypoints: torch.Tensor, keypoints_length: int) -> Dict[str, Any]:
        """
        Run inference on a single sample
        """
        self.model.eval()
        
        with torch.no_grad():
            # Ensure proper dimensions [1, seq_len, features]
            if keypoints.dim() == 2:
                keypoints = keypoints.unsqueeze(0)
            
            keypoints = keypoints.to(self.device)
            keypoints_length = torch.tensor([keypoints_length], device=self.device)
            
            # Create source padding mask
            src_key_padding_mask = (
                torch.arange(keypoints.size(1), device=self.device)[None, :] >= 
                keypoints_length[:, None]
            )
            
            # Generate prediction
            generated_seq = self.model(
                keypoints, 
                tgt=None, 
                src_key_padding_mask=src_key_padding_mask
            )
            
            # Decode prediction
            prediction = self.vocab.arrays_to_sentences(generated_seq.cpu().numpy())[0]
            prediction_text = ' '.join(prediction) if prediction else '<EMPTY>'
            
            return {
                'predicted_text': prediction_text,
                'predicted_tokens': prediction,
                'sequence_length': keypoints.size(1),
                'valid_frames': keypoints_length.item()
            }
    
    def predict_from_samples(self, samples: List[Dict], collate_fn) -> List[Dict[str, Any]]:
        """
        Run inference on a list of samples using dataset format
        """
        from preprocess.dataset import SignDataset
        from torch.utils.data import DataLoader
        
        # Create temporary dataset
        dataset = SignDataset(
            samples=samples,
            vocab=self.vocab,
            config=self.config.preprocessing if (self.config and hasattr(self.config, 'preprocessing')) else {},
            dataset_size=None,
            cache_dir=None
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False
        )
        
        # Run inference
        predictions = self.predict_batch(dataloader)
        
        # Add video_id information from original samples
        for i, (pred, sample) in enumerate(zip(predictions, samples)):
            pred['video_id'] = sample.get('video_id', f'sample_{i}')
            pred['direction'] = sample.get('direction', 'unknown')
        
        return predictions


def load_model_for_inference(model_path: str, vocab, device, config=None):
    """
    Load a trained model for inference
    """
    from models.MSKA import MSKA_Model
    
    print(f"🔄 Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    if config and hasattr(config, 'model'):
        model_config = config.model
    else:
        # Use config from checkpoint if available
        model_config = checkpoint.get('config', {}).get('model', {})
    
    model = MSKA_Model(
        num_classes=len(vocab),
        input_dim=model_config.get('input_dim', 274),
        d_model=model_config.get('d_model', 1024),
        nhead=model_config.get('nhead', 8),
        num_encoder_layers=model_config.get('num_encoder_layers', 10),
        num_decoder_layers=model_config.get('num_decoder_layers', 8),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.2)
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model


def create_inference_engine(model_path: str, vocab_path: str, device, config=None):
    from utils.vocabulary import GlossVocabulary
    
    # Load vocabulary
    vocab = GlossVocabulary.load(vocab_path)
    print(f"📚 Loaded vocabulary with {len(vocab)} tokens")
    
    # Load model
    model = load_model_for_inference(model_path, vocab, device, config)
    
    # Create inference engine
    return MSKAInference(model, vocab, device, config)


def extract_keypoints_from_video(video_path: str, output_dir: str) -> str:
    """
    Extract keypoints from video file using OpenPose
    Returns: path to keypoint directory
    """
    print(f"🎬 Extracting keypoints from video: {video_path}")
    
    extractor = KeypointExtractor()
    
    # Create output directory for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keypoint_dir = os.path.join(output_dir, "keypoints", video_name)
    os.makedirs(keypoint_dir, exist_ok=True)
    
    # Extract keypoints frame by frame
    keypoint_files = extractor.extract_from_video(video_path, keypoint_dir)
    
    print(f"📊 Extracted {len(keypoint_files)} keypoint files")
    return keypoint_dir


def prepare_keypoint_data(keypoint_dir: str, config: DictConfig):
    """
    Prepare keypoint data for inference
    """
    from preprocess.dataset import SignDataset
    
    # Get all keypoint files
    keypoint_files = [f for f in os.listdir(keypoint_dir) if f.endswith('_keypoints.json')]
    keypoint_files.sort(key=lambda x: int(x.split('_')[-2]) if '_' in x else 0)
    
    if not keypoint_files:
        raise ValueError(f"No keypoint files found in {keypoint_dir}")
    
    # Create a sample in the format expected by SignDataset
    sample = {
        'video_id': os.path.basename(keypoint_dir),
        'direction': 'F',  # Assume front direction for video input
        'keypoint_dir': keypoint_dir,
        'keypoint_files': keypoint_files,
        'morphemes': ['<UNK>'],  # Unknown morphemes for inference
        'dataset_name': 'video_input',
        'base_name': os.path.basename(keypoint_dir)
    }
    
    return [sample]


def get_model_and_vocab_paths(config):
    """Get model and vocabulary paths from config"""
    if hasattr(config, 'model_path') and hasattr(config, 'vocab_path'):
        model_path = config.model_path
        vocab_path = config.vocab_path
    else:
        # Fallback to default paths
        model_path = os.path.join(config.output.dir, 'best_model.pth')
        vocab_path = os.path.join(config.output.dir, config.vocab.save_path)
    
    return model_path, vocab_path


def inference_from_keypoints(samples, config, device):
    # Get model and vocab paths
    model_path, vocab_path = get_model_and_vocab_paths(config)
    
    # Create inference engine
    inference_engine = create_inference_engine(model_path, vocab_path, device, config)
    
    # Run inference
    predictions = inference_engine.predict_from_samples(
        samples, 
        lambda b: collate_fn(b, inference_engine.vocab)
    )
    
    # Format results
    results = []
    for pred in predictions:
        results.append({
            'video_id': pred['video_id'],
            'predicted_text': pred['predicted_text'],
            'predicted_tokens': pred['predicted_tokens']
        })
    
    return results


def run_test_dataset_inference(config, device, output_dir):
    """Run inference on entire test dataset"""
    from dataloader.SignDataLoader import SignDataSplitter
    
    print("🔍 Running inference on test dataset...")
    
    # Create data splitter to scan test data
    data_splitter = SignDataSplitter(config)
    test_samples = data_splitter.scan_morpheme_datasets('test')
    
    if not test_samples:
        print("❌ No test samples found")
        return
    
    print(f"📊 Found {len(test_samples)} test samples")
    
    # Run inference
    results = inference_from_keypoints(test_samples, config, device)
    
    # Save results
    output_file = os.path.join(output_dir, 'test_dataset_inference_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': 'test',
            'total_samples': len(test_samples),
            'results': results,
            'config': OmegaConf.to_container(config)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Test dataset inference complete. Results: {len(results)} predictions")


def main():
    parser = argparse.ArgumentParser(description='MSKA Model Inference')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--video', type=str, help='Path to single input video file')
    mode_group.add_argument('--test-dataset', action='store_true', help='Run inference on entire test dataset')
    mode_group.add_argument('--keypoint-dir', type=str, help='Pre-extracted keypoint directory for single video')
    
    # Configuration and output
    parser.add_argument('--config', type=str, default='configs/inference_config.yaml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./inference_outputs', help='Output directory')
    parser.add_argument('--extract-keypoints', action='store_true', help='Extract keypoints from video (requires OpenPose)')
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = OmegaConf.load(args.config)
    
    # Setup device
    device = get_device_config(config)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        if args.test_dataset:
            # Run inference on entire test dataset
            run_test_dataset_inference(config, device, args.output)
        else:
            # Single video inference
            if args.keypoint_dir and os.path.exists(args.keypoint_dir):
                print(f"📁 Using existing keypoint directory: {args.keypoint_dir}")
                keypoint_dir = args.keypoint_dir
                video_name = os.path.basename(args.keypoint_dir)
            elif args.video:
                if args.extract_keypoints:
                    keypoint_dir = extract_keypoints_from_video(args.video, args.output)
                else:
                    raise ValueError("For video input, use --extract-keypoints to generate keypoints")
                video_name = os.path.splitext(os.path.basename(args.video))[0]
            else:
                raise ValueError("Either --video, --keypoint-dir, or --test-dataset must be provided")
            
            # Prepare data
            samples = prepare_keypoint_data(keypoint_dir, config)
            
            # Run inference
            results = inference_from_keypoints(samples, config, device)
            
            # Save results
            output_file = os.path.join(args.output, f'{video_name}_inference_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_name': video_name,
                    'keypoint_dir': keypoint_dir,
                    'results': results,
                    'config': OmegaConf.to_container(config)
                }, f, ensure_ascii=False, indent=2)
            
            # Print results
            print(f"\n🎯 Inference Results:")
            for result in results:
                print(f"   Video: {result['video_id']}")
                print(f"   Predicted: {result['predicted_text']}")
                print(f"   Tokens: {result['predicted_tokens']}")
            
            print(f"\n📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())