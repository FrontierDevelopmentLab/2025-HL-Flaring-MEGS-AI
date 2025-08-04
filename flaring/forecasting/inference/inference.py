import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
from datetime import datetime
import pandas as pd
import os
import sys
import re
from typing import Optional, Tuple, List, Dict, Any

# Get the absolute path to the forecasting directory
current_dir = Path(__file__).parent  # inference/
forecasting_dir = current_dir.parent  # forecasting/
sys.path.insert(0, str(forecasting_dir))

# Import models
from models.vision_transformer_custom import ViT
from models.FastSpectralNet import FastViTFlaringModel
from training.train import Seq2SeqViTWrapper

# Import data modules
from data_loaders.SDOAIA_dataloader import AIA_GOESSequenceDataset

def resolve_config_variables(config_dict, base_vars=None):
    """Resolve ${variable} references within the config"""
    variables = base_vars.copy() if base_vars else {}

    # First pass - collect non-referencing variables
    for key, value in config_dict.items():
        if isinstance(value, str) and not value.startswith('${'):
            variables[key] = value

    def substitute_value(value, variables):
        if isinstance(value, str):
            pattern = r'\$\{([^}]+)\}'
            for match in re.finditer(pattern, value):
                var_name = match.group(1)
                if var_name in variables:
                    value = value.replace(f'${{{var_name}}}', variables[var_name])
        return value

    def recursive_substitute(obj, variables):
        if isinstance(obj, dict):
            return {k: recursive_substitute(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_substitute(item, variables) for item in obj]
        else:
            return substitute_value(obj, variables)

    return recursive_substitute(config_dict, variables)

def load_configs(model_config_path: str, inference_config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load both model and inference configs with proper variable resolution"""
    # Load model config first
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # Resolve model config variables
    model_config = resolve_config_variables(model_config)

    # Load inference config
    with open(inference_config_path, 'r') as f:
        inference_config = yaml.safe_load(f)

    # Resolve inference config variables using model config as base
    inference_config = resolve_config_variables(inference_config, model_config)

    return model_config, inference_config

def load_model(model_config: Dict[str, Any], model_path: str, device: torch.device):
    """Load trained model based on config"""
    input_seq_length = model_config['training']['sequence']['input_length']
    output_seq_length = model_config['training']['sequence']['output_length']
    sxr_norm_path = model_config['data']['paths']['sxr_norm']
    sxr_norm = np.load(sxr_norm_path, allow_pickle=True)
    print(f"Loaded sxr_norm from {sxr_norm_path}: type={type(sxr_norm)}, content={sxr_norm}")
    if isinstance(sxr_norm, (np.ndarray, list)) and len(sxr_norm) == 2:
        sxr_norm = tuple(float(x) for x in sxr_norm)
        print(f"Converted sxr_norm to tuple: {sxr_norm}")
    if not isinstance(sxr_norm, tuple) or len(sxr_norm) != 2:
        raise ValueError(f"sxr_norm must be a tuple of (mean, std), got type={type(sxr_norm)}, content={sxr_norm}")
    if not all(np.isfinite(sxr_norm)):
        raise ValueError(f"sxr_norm contains non-finite values: {sxr_norm}")

    # Initialize model based on config
    if model_config['model']['type'] == 'ViT':
        vit_kwargs = {
            **model_config['ViT']['architecture'],
            **model_config['ViT']['training'],
            'eve_norm': sxr_norm
        }
        model = Seq2SeqViTWrapper(
            model_kwargs=vit_kwargs,
            input_sequence_length=input_seq_length,
            output_sequence_length=output_seq_length
        )
    elif model_config['model']['type'] == 'FastViT':
        fastvit_kwargs = {
            **model_config['FastViT']['architecture'],
            **model_config['FastViT']['training'],
            'eve_norm': sxr_norm
        }
        model = Seq2SeqFastViTWrapper(
            d_input=6,
            d_output=output_seq_length,
            eve_norm=sxr_norm,
            image_size=512,
            patch_size=32,
            input_sequence_length=input_seq_length,
            output_sequence_length=output_seq_length,
            **fastvit_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['model']['type']}")

    # Load weights if available
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Successfully loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}. Using initialized model. Error: {str(e)}")
    else:
        print(f"No model weights found at {model_path}. Using initialized model with config parameters.")

    model = model.to(device)
    model.eval()
    print(f"Model moved to device: {device}")
    return model

def denormalize_sxr(sxr_values: np.ndarray, sxr_norm: Tuple[float, float]) -> np.ndarray:
    """Convert normalized SXR values back to original scale"""
    return np.power(10, sxr_values * sxr_norm[1] + sxr_norm[0]) - 1e-8

def predict_sequence(
        model: torch.nn.Module,
        dataset: AIA_GOESSequenceDataset,
        sample_idx: int,
        device: torch.device,
        sxr_norm: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make predictions for a single sequence sample"""
    input_sequence, true_values = dataset[sample_idx]
    print(f"Dataset sample {sample_idx}: input shape={input_sequence.shape}, true_values shape={true_values.shape}")
    input_tensor = input_sequence.unsqueeze(0).to(device)
    print(f"Input tensor shape: {input_tensor.shape}, device: {input_tensor.device}")

    with torch.no_grad():
        pred_values = model(input_tensor).cpu().numpy().squeeze()
    print(f"Predicted values shape: {pred_values.shape}")

    input_sequence = input_sequence.cpu().numpy()
    true_values = true_values.cpu().numpy()

    true_values = denormalize_sxr(true_values, sxr_norm)
    pred_values = denormalize_sxr(pred_values, sxr_norm)

    return input_sequence, true_values, pred_values

def collect_sequence_results(
        true_values: np.ndarray,
        pred_values: np.ndarray,
        sample_idx: int,
        output_seq_length: int,
        timestamp: str
) -> pd.DataFrame:
    """Collect true and predicted SXR values for a single sample"""
    time_steps = np.arange(output_seq_length)
    data = {
        'Sample_Index': [sample_idx] * output_seq_length,
        'Time_Step': time_steps,
        'True_SXR': true_values,
        'Predicted_SXR': pred_values,
        'Timestamp': [timestamp] * output_seq_length
    }
    return pd.DataFrame(data)

def run_inference(
        model_config_path: str,
        inference_config_path: str,
        output_dir: str,
        num_samples: int = 5,
        device: str = 'cuda'
):
    """Main inference function"""
    # Load configurations
    model_config, inference_config = load_configs(model_config_path, inference_config_path)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model_path = inference_config.get('model_path')
    model = load_model(model_config, model_path, device)

    # Load sxr_norm
    sxr_norm_path = inference_config['data']['sxr_norm_path']
    sxr_norm = np.load(sxr_norm_path, allow_pickle=True)
    print(f"Loaded sxr_norm from {sxr_norm_path}: type={type(sxr_norm)}, content={sxr_norm}")
    if isinstance(sxr_norm, (np.ndarray, list)) and len(sxr_norm) == 2:
        sxr_norm = tuple(float(x) for x in sxr_norm)
        print(f"Converted sxr_norm to tuple: {sxr_norm}")
    if not isinstance(sxr_norm, tuple) or len(sxr_norm) != 2:
        raise ValueError(f"sxr_norm must be a tuple of (mean, std), got type={type(sxr_norm)}, content={sxr_norm}")

    # Setup dataset
    dataset = AIA_GOESSequenceDataset(
        aia_dir=inference_config['data']['aia_dir'],
        sxr_dir=inference_config['data']['sxr_dir'],
        sequence_length=model_config['training']['sequence']['input_length'],
        stride=1,
        sxr_transform=lambda x: (np.log10(x + 1e-8) - sxr_norm[0]) / sxr_norm[1],
        target_size=(512, 512)
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Collect all results
    results_dfs = []
    num_samples = min(num_samples, len(dataset))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run inference
    for sample_idx in range(num_samples):
        input_seq, true_sxr, pred_sxr = predict_sequence(
            model, dataset, sample_idx, device, sxr_norm)

        # Collect results for this sample
        sample_df = collect_sequence_results(
            true_sxr, pred_sxr,
            sample_idx,
            model_config['training']['sequence']['output_length'],
            timestamp)

        results_dfs.append(sample_df)

        #print(f"\nSample {sample_idx} Results:")
        #print(f"Input Sequence Length: {model_config['training']['sequence']['input_length']}")
       # print(f"Prediction Horizon: {model_config['training']['sequence']['output_length']} steps")
        print("True SXR Values:", true_sxr)
        print("Predicted SXR Values:", pred_sxr)

    # Combine all results and save to a single CSV
    output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    combined_df = pd.concat(results_dfs, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    print(f"All results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run sequence-to-sequence solar flare prediction')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--inference_config', type=str, required=True, help='Path to inference config YAML')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    run_inference(
        model_config_path=args.model_config,
        inference_config_path=args.inference_config,
        output_dir=args.output,
        num_samples=args.num_samples,
        device=args.device
    )