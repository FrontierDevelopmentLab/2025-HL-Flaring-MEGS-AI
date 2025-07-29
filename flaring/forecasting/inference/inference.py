import argparse
import re
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import yaml
import os
import glob
from datetime import datetime
import torchvision.transforms as T

# Get the absolute path to the forecasting directory
current_dir = Path(__file__).parent  # inference/
forecasting_dir = current_dir.parent  # forecasting/
sys.path.insert(0, str(forecasting_dir))

from data_loaders.SDOAIA_dataloader import AIA_GOESSequenceDataset
from training.train import SequenceViTWrapper, SequenceFastViTWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unnormalize_sxr(normalized_data, norm_params):
    """Reverse log10 and z-score normalization for SXR values."""
    if isinstance(normalized_data, torch.Tensor):
        normalized_data = normalized_data.cpu().numpy()
    mean, std = norm_params  # Assume norm_params is [mean, std]
    if std == 0:
        raise ValueError("Standard deviation in norm_params is zero")
    log_sxr = normalized_data * std + mean
    sxr = 10 ** log_sxr - 1e-8
    return sxr

def predict_sequence(model, dataset, batch_size=1, visualize_attention=False, output_dir=None):
    """Generator yielding predictions for time series data with optional attention visualization."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    if visualize_attention and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            aia_sequences = batch[0].to(device)
            sxr_targets = batch[1].to(device)

            if visualize_attention:
                preds, attn_weights = model(aia_sequences, return_attention=True)
                if batch_idx == 0:
                    visualize_attention_maps(
                        aia_sequences[0].cpu().numpy(),
                        attn_weights[:, 0].cpu().numpy(),
                        output_dir=output_dir,
                        timestamp=str(datetime.now()))
            else:
                preds = model(aia_sequences)

            # Debug: Print raw predictions and targets
            print(f"Batch {batch_idx}: Raw preds = {preds.cpu().numpy()}, Raw targets = {sxr_targets.cpu().numpy()}")

            yield {
                'predictions': preds.cpu().numpy(),
                'targets': sxr_targets.cpu().numpy(),
                'timestamps': dataset.dataset.sequences[batch_idx*batch_size : (batch_idx+1)*batch_size] if isinstance(dataset, Subset) else dataset.sequences[batch_idx*batch_size : (batch_idx+1)*batch_size]
            }

def visualize_attention_maps(sequence, attn_weights, output_dir, timestamp):
    """Save attention visualization for a sequence."""
    last_layer_attn = attn_weights[-1].mean(0)

    plt.figure(figsize=(15, 5))
    for i in range(min(3, sequence.shape[0])):
        plt.subplot(1, 3, i+1)
        # Normalize AIA image to [0, 1] for visualization (assuming channels 0:3 are RGB-like)
        img = sequence[i][..., :3]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0, 1]
        plt.imshow(img)
        plt.title(f"Frame {i}")
        plt.axis('off')

    seq_path = os.path.join(output_dir, f"sequence_{timestamp.replace(':', '-')}.png")
    plt.savefig(seq_path)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(last_layer_attn, cmap='hot')
    plt.title("Attention Weights (Last Layer)")
    plt.colorbar()
    attn_path = os.path.join(output_dir, f"attention_{timestamp.replace(':', '-')}.png")
    plt.savefig(attn_path)
    plt.close()

def resolve_config_variables(config_dict):
    """Recursively resolve ${variable} references within the config."""
    variables = {}
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

def load_model(train_config, checkpoint_path):
    """Load model using parameters from training config."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device)

    # Get model parameters from training config
    model_type = train_config['model']['type']
    sequence_length = train_config['training']['sequence']['length']

    if model_type == 'ViT':
        vit_kwargs = {
            'patch_size': train_config['ViT']['architecture']['patch_size'],
            'num_channels': train_config['ViT']['architecture']['num_channels'],
            'embed_dim': train_config['ViT']['architecture']['embed_dim'],
            'num_heads': train_config['ViT']['architecture']['num_heads'],
            'num_classes': train_config['ViT']['architecture']['num_classes'],
            'num_patches': train_config['ViT']['architecture']['num_patches'],
            'num_layers': train_config['ViT']['architecture']['num_layers'],
            'hidden_dim': train_config['ViT']['architecture']['hidden_dim']
        }

        model = SequenceViTWrapper(
            model_kwargs=vit_kwargs,
            sequence_length=sequence_length
        )
    elif model_type == 'FastViT':
        fastvit_kwargs = {
            'd_input': train_config['ViT']['architecture']['num_channels'],
            'd_output': train_config['ViT']['architecture']['num_classes'],
            'embed_dim': train_config['FastViT']['architecture']['embed_dim'],
            'num_heads': train_config['FastViT']['architecture']['num_heads'],
            'depth': train_config['FastViT']['architecture']['depth'],
            'mlp_ratio': train_config['FastViT']['architecture']['mlp_ratio'],
            'qkv_bias': train_config['FastViT']['architecture']['qkv_bias'],
            'drop_rate': train_config['FastViT']['architecture']['drop_rate'],
            'attn_drop_rate': train_config['FastViT']['architecture']['attn_drop_rate'],
            'image_size': 512,
            'patch_size': 32
        }

        model = SequenceFastViTWrapper(
            sequence_length=sequence_length,
            **fastvit_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    if isinstance(state, dict):
        if 'model' in state:
            model.load_state_dict(state['model'].state_dict())
        elif 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
    else:
        model.load_state_dict(state)

    return model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-config', type=str, required=True,
                        help='Path to training config YAML')
    parser.add_argument('--inference-config', type=str, required=True,
                        help='Path to inference config YAML')
    parser.add_argument('--visualize-attention', action='store_true',
                        help='Generate attention visualizations')
    args = parser.parse_args()

    # Load and resolve both configs
    with open(args.train_config, 'r') as stream:
        train_config = yaml.safe_load(stream)
    train_config = resolve_config_variables(train_config)

    with open(args.inference_config, 'r') as stream:
        inference_config = yaml.safe_load(stream)
    inference_config = resolve_config_variables(inference_config)

    # Get paths from inference config
    checkpoint_path = inference_config['data']['checkpoint_path']
    aia_test_dir = os.path.join(inference_config['data']['aia_dir'], 'test')
    sxr_test_dir = os.path.join(inference_config['data']['sxr_dir'], 'test')
    sxr_norm_path = inference_config['data']['sxr_norm_path']
    output_path = inference_config['output_path']

    # Debug: Inspect normalization parameters
    norm_params = np.load(sxr_norm_path)
    try:
        mean, std = norm_params['mean'], norm_params['std']
    except (IndexError, TypeError):
        mean, std = norm_params  # Assume [mean, std]
    print(f"Normalization params: mean={mean}, std={std}")
    norm_params = (mean, std)

    # Debug: Inspect SXR data files
    sxr_files = sorted(glob.glob(os.path.join(sxr_test_dir, "*.npy")))
    print(f"Found {len(sxr_files)} SXR files in {sxr_test_dir}")
    for f in sxr_files[:5]:
        sxr_val = np.load(f)
        print(f"SXR file {os.path.basename(f)}: {sxr_val}")

    # Load model using training config
    model = load_model(train_config, checkpoint_path)

    # Define sxr_transform to match training
    sxr_transform = T.Lambda(lambda x: (np.log10(x + 1e-8) - mean) / std)

    # Initialize sequence dataset with sxr_transform
    full_dataset = AIA_GOESSequenceDataset(
        aia_dir=aia_test_dir,
        sxr_dir=sxr_test_dir,
        sequence_length=train_config['training']['sequence']['length'],
        stride=train_config['training']['sequence']['stride'],
        sxr_transform=sxr_transform
    )

    # Limit to first 100 sequences
    num_sequences = min(8000, len(full_dataset))
    dataset = Subset(full_dataset, range(num_sequences))

    # Debug: Inspect dataset samples
    print(f"Dataset size: {len(dataset)} sequences")
    for i in range(min(5, len(dataset))):
        aia_seq, sxr_target = dataset[i]
        timestamp = full_dataset.sequences[i][-1]
        # Unnormalize sxr_target for debugging
        unnorm_target = unnormalize_sxr(sxr_target, norm_params)
        print(f"Sample {i}: timestamp={timestamp}, sxr_target={sxr_target}, unnorm_target={unnorm_target}, aia_seq_shape={aia_seq.shape}")

    # Prepare output
    results = []
    output_dir = os.path.dirname(output_path)
    attention_dir = os.path.join(output_dir, 'attention_plots') if args.visualize_attention else None

    # Run inference
    for batch in predict_sequence(
            model,
            dataset,
            batch_size=train_config['training']['batch_size'],
            visualize_attention=args.visualize_attention,
            output_dir=attention_dir
    ):
        for i, (pred, target, timestamp) in enumerate(zip(
                batch['predictions'],
                batch['targets'],
                batch['timestamps']
        )):
            # Use the last timestamp in the sequence
            timestamp = timestamp[-1]
            # Debug: Print raw and unnormalized values
            print(f"Timestamp: {timestamp}, Raw pred: {pred}, Raw target: {target}")
            unnorm_pred = unnormalize_sxr(pred, norm_params).item()
            unnorm_target = unnormalize_sxr(target, norm_params).item()
            print(f"Unnormalized pred: {unnorm_pred}, Unnormalized target: {unnorm_target}")

            results.append({
                'Timestamp': timestamp,
                'Prediction': unnorm_pred,
                'Ground_Truth': unnorm_target
            })

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    if args.visualize_attention and attention_dir:
        print(f"Attention visualizations saved to {attention_dir}")

if __name__ == '__main__':
    main()