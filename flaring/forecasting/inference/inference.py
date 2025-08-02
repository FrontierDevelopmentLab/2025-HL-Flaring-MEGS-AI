import argparse
import re
import sys
import pandas as pd
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from flaring.forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataset
import flaring.forecasting.models as models
from flaring.forecasting.models.vision_transformer_custom import ViT
from flaring.forecasting.models.linear_and_hybrid import HybridIrradianceModel  # Add your hybrid model import
from flaring.forecasting.training.callback import unnormalize_sxr
import yaml
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def has_attention_weights(model):
    """Check if model supports attention weights"""
    return hasattr(model, 'attention') or isinstance(model, ViT)


def evaluate_model_on_dataset(model, dataset, batch_size=16, times=None, config_data=None, save_weights=True, input_size = 512, patch_size = 16):
    """Optimized generator with batch processing and weight saving"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Check if this model supports attention weights
    supports_attention = has_attention_weights(model) and save_weights

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Correct unpacking based on your data structure
            aia_imgs = batch[0]  # Get aia_img from inputs
            sxr = batch[1]
            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device, non_blocking=True)

            # Get model predictions for entire batch
            pred = model(aia_imgs)

            # Handle different model output formats
            if isinstance(pred, tuple) and len(pred) > 1:
                predictions = pred[0]  # Shape: [batch_size, ...]
                weights = pred[1] if supports_attention else None  # Shape: [batch_size, heads, L, S ...]
            else:
                predictions = pred
                weights = None

            # Process entire batch at once for weights if needed
            batch_weights = []
            if supports_attention and weights is not None:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    # Process attention weights for this item - matching callback approach
                    #select last layer and appropriate item from batch
                    last_layer_attention = weights[-1][i]  # Get i-th item from batch [num_heads, seq_len, seq_len]
                    # Average across attention heads
                    avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]

                    # Get attention from CLS token to patches (exclude CLS->CLS)
                    cls_attention = avg_attention[0, 1:].cpu()  # [num_patches] - 1D array

                    # Calculate grid size based on patch size (assuming 8x8 patches)
                    grid_h, grid_w = input_size // patch_size, input_size // patch_size  # Should be 64, 64

                    # Reshape CLS attention to spatial grid
                    attention_map = cls_attention.reshape(grid_h, grid_w)  # [64, 64]

                    batch_weights.append(attention_map.numpy())

                # Save all weights in this batch at once
                if config_data and 'weight_path' in config_data:
                    save_batch_weights(batch_weights, batch_idx, batch_size, times, config_data['weight_path'])

            # Yield batch results
            current_batch_size = predictions.shape[0]
            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i
                weight_data = batch_weights[i] if (supports_attention and batch_weights) else None
                yield (predictions[i].cpu().numpy(), sxr[i].cpu().numpy(),
                       weight_data, global_idx)


def save_batch_weights(batch_weights, batch_idx, batch_size, times, weight_path):
    """Save all weights in a batch efficiently"""
    weight_dir = Path(weight_path)
    weight_dir.mkdir(parents=True, exist_ok=True)

    # Use ThreadPoolExecutor to save files in parallel
    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    # Prepare arguments for parallel saving
    save_args = []
    for i, weight in enumerate(batch_weights):
        global_idx = batch_idx * batch_size + i
        if global_idx < len(times):  # Make sure we don't go out of bounds
            filepath = weight_path + f"{times[global_idx]}"
            save_args.append((weight, filepath))

    # Save all weights in this batch in parallel
    with ThreadPoolExecutor(max_workers=min(4, len(save_args))) as executor:
        executor.map(save_single_weight, save_args)


def save_weights_async(weight_data_queue, weight_path):
    """Async function to save weights to disk"""

    def save_single_weight(args):
        weight, filepath = args
        np.savetxt(filepath, weight, delimiter=",")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(save_single_weight, weight_data_queue)


def load_model_from_config(config_data):
    """Load model based on config specifications"""
    checkpoint_path = config_data['data']['checkpoint_path']
    model_type = config_data['model']  # Default to ViT for backward compatibility

    print(f"Loading {model_type} model...")

    if ".ckpt" in checkpoint_path:
        # Lightning checkpoint format
        if model_type.lower() == 'vit':
            model = ViT.load_from_checkpoint(checkpoint_path)
        elif model_type.lower() == 'hybrid' or model_type.lower() == 'hybridirradiancemodel':
            model = HybridIrradianceModel.load_from_checkpoint(checkpoint_path)
        else:
            # Try to dynamically load the model class
            try:
                model_class = getattr(models, model_type)
                model = model_class.load_from_checkpoint(checkpoint_path)
            except AttributeError:
                raise ValueError(f"Unknown model type: {model_type}. Available types: ViT, HybridIrradianceModel")
    else:
        # Regular PyTorch checkpoint
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = state['model']

    model.eval()
    model.to(device)
    return model


def main():
    def resolve_config_variables(config_dict):
        """Recursively resolve ${variable} references within the config"""
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML.')
    parser.add_argument('-input_size', type=int, default=512, help='Input size for the model')
    parser.add_argument('-patch_size', type=int, default=16, help='Patch size for the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--no_weights', action='store_true', help='Skip saving attention weights to speed up')
    args = parser.parse_args()

    # Load config with variable substitution
    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models

    # Load model based on config
    model = load_model_from_config(config_data)

    # Check if model supports attention and user wants to save weights
    save_weights = not args.no_weights and has_attention_weights(model)

    if args.no_weights:
        print("Skipping attention weight saving (--no_weights flag)")
    elif not has_attention_weights(model):
        print(f"Model {type(model).__name__} doesn't support attention weights - skipping weight saving")
    else:
        print("Will save attention weights during inference")

    # Enable optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Dataset
    print("Loading dataset...")
    dataset = AIA_GOESDataset(
        aia_dir=config_data['data']['aia_dir'] + '/test',
        sxr_dir=config_data['data']['sxr_dir'] + '/test',
    )

    times = dataset.samples
    sxr_norm = np.load(config_data['data']['sxr_norm_path'])

    # Pre-allocate lists for better memory performance
    total_samples = len(times)
    timestamp = []
    predictions = []
    ground = []

    print(f"Processing {total_samples} samples with batch size {args.batch_size}...")

    # Process in batches
    for prediction, sxr, weight, idx in evaluate_model_on_dataset(
            model, dataset, args.batch_size, times, config_data, save_weights, args.input_size, args.patch_size
    ):
        # Unnormalize prediction
        pred = unnormalize_sxr(prediction, sxr_norm)

        # Store results
        predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
        ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
        timestamp.append(str(times[idx]))

        # Progress update
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total_samples}")

    if save_weights:
        print("All weights saved during batch processing!")
    else:
        print("Inference completed (no weights saved)!")

    # Create and save results DataFrame
    print("Creating output DataFrame...")
    output_df = pd.DataFrame({
        'timestamp': timestamp,
        'predictions': predictions,
        'groundtruth': ground
    })

    print(output_df.head())
    #Make output directory if it doesn't exist
    output_dir = Path(config_data['output_path']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config_data['output_path'], index=False)
    print(f"Predictions saved to {config_data['output_path']}")


if __name__ == '__main__':
    main()