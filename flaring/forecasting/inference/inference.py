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
from flaring.forecasting.training.callback import unnormalize_sxr
import yaml
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_log_outputs_batch(model, dataset, batch_size=8, times=None, config_data=None, save_weights=True):
    """Optimized generator with batch processing and weight saving"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Correct unpacking based on your data structure
            aia_imgs = batch[0]  # Get aia_img from inputs
            sxr = batch[1]
            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device, non_blocking=True)

            # Get model predictions for entire batch
            pred = model(aia_imgs)
            if len(pred) > 1:
                predictions = pred[0]  # Shape: [batch_size, ...]
                weights = pred[1]  # Shape: [batch_size, ...]

            # Process entire batch at once for weights if needed
            batch_weights = []
            if save_weights:
                current_batch_size = predictions.shape[0]
                for i in range(current_batch_size):
                    # Process attention weights for this item
                    # Process attention weights for this item - matching callback approach
                    last_layer_attention = weights[-1][i]  # Get i-th item from batch [num_heads, seq_len, seq_len]

                    # Average across attention heads
                    avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]

                    # Get attention from CLS token to patches (exclude CLS->CLS)
                    cls_attention = avg_attention[0, 1:].cpu()  # [num_patches] - 1D array

                    # Calculate grid size based on patch size (assuming 8x8 patches)
                    patch_size = 8
                    grid_h, grid_w = 512 // patch_size, 512 // patch_size  # Should be 64, 64

                    #print(f"CLS attention shape: {cls_attention.shape}, Expected grid: {grid_h}x{grid_w}")

                    # Reshape CLS attention to spatial grid
                    attention_map = cls_attention.reshape(grid_h, grid_w)  # [64, 64]

                    #attention_map_upsampled = attention_map
                    # Upsample attention map from 64x64 to 512x512 to match image size
                    # attention_map_upsampled = F.interpolate(
                    #     attention_map.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                    #     size=(512, 512),
                    #     mode='bilinear',
                    #     align_corners=False
                    # ).squeeze()  # Remove batch and channel dims

                    batch_weights.append(attention_map.numpy())

                # Save all weights in this batch at once
                if config_data:
                    save_batch_weights(batch_weights, batch_idx, batch_size, times, config_data['weight_path'])

            # Yield batch results
            current_batch_size = predictions.shape[0]
            for i in range(current_batch_size):
                global_idx = batch_idx * batch_size + i
                weight_data = batch_weights[i] if save_weights else None
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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--no_weights', action='store_true', help='Skip saving attention weights to speed up')
    args = parser.parse_args()

    # Load config with variable substitution
    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models

    # Load model
    print("Loading model...")
    state = torch.load(config_data['data']['checkpoint_path'], map_location=device, weights_only=False)
    model = state['model']
    model.to(device)

    # Enable optimizations
    model.eval()
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
    for prediction, sxr, weight, idx in predict_log_outputs_batch(
            model, dataset, args.batch_size, times, config_data, not args.no_weights
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

    print("All weights saved during batch processing!")

    # Create and save results DataFrame
    print("Creating output DataFrame...")
    output_df = pd.DataFrame({
        'Timestamp': timestamp,
        'Predictions': predictions,
        'ground_truth': ground
    })

    print(output_df.head())
    output_df.to_csv(config_data['output_path'], index=False)
    print(f"Predictions saved to {config_data['output_path']}")


if __name__ == '__main__':
    main()


# import argparse
# import re
# import sys
# import pandas as pd
# import torch
# import numpy as np
# from torch.utils.checkpoint import checkpoint
# from torch.utils.data import DataLoader
# from flaring.forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataset
# import flaring.forecasting.models as models
# from flaring.forecasting.training.callback import unnormalize_sxr
# import yaml
# import torch.nn.functional as F
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def predict_log_outputs(model, dataset, batch_size=1, times=None):
#     """Generator yielding raw log-space model outputs"""
#     model.eval()
#     loader = DataLoader(dataset, batch_size=batch_size)
#
#     with torch.no_grad():
#         for batch in loader:
#             # Correct unpacking based on your data structure
#             aia_imgs = batch[0] # Get aia_img from inputs
#             sxr = batch[1]
#             # Move to device (it's already a tensor)
#             aia_imgs = aia_imgs.to(device)
#
#             # Get model predictions
#             pred = model(aia_imgs)
#             if len(pred) > 1:
#                 predictions = pred[0][0]
#                 weights = pred[1]
#             #print(weights)
#             # Move to CPU and convert to numpy before yielding
#             last_layer_attention = weights[-1]
#             last_layer_attention = last_layer_attention.mean(dim=0)
#             last_layer_attention = last_layer_attention[0, 1:, 1:].cpu()
#             # Add batch and channel dimensions for F.interpolate
#             last_layer_attention = last_layer_attention.unsqueeze(0).unsqueeze(0)
#
#             # Downsample from 4096x4096 to 512x512
#             last_layer_attention = F.interpolate(
#                 last_layer_attention,
#                 size=(512, 512),
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze()  # Remove batch and channel dims
#             # Reshape attention to spatial grid
#             yield (predictions.cpu().numpy(), sxr.cpu().numpy(), last_layer_attention)
#
# def main():
#     def resolve_config_variables(config_dict):
#         """Recursively resolve ${variable} references within the config"""
#
#         # Extract variables defined at root level (like base_data_dir, base_checkpoint_dir)
#         variables = {}
#         for key, value in config_dict.items():
#             if isinstance(value, str) and not value.startswith('${'):
#                 variables[key] = value
#
#         def substitute_value(value, variables):
#             if isinstance(value, str):
#                 # Replace ${var_name} with actual values
#                 pattern = r'\$\{([^}]+)\}'
#                 for match in re.finditer(pattern, value):
#                     var_name = match.group(1)
#                     if var_name in variables:
#                         value = value.replace(f'${{{var_name}}}', variables[var_name])
#             return value
#
#         def recursive_substitute(obj, variables):
#             if isinstance(obj, dict):
#                 return {k: recursive_substitute(v, variables) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [recursive_substitute(item, variables) for item in obj]
#             else:
#                 return substitute_value(obj, variables)
#
#         return recursive_substitute(config_dict, variables)
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML.')
#     args = parser.parse_args()
#
#     # Load config with variable substitution
#     with open(args.config, 'r') as stream:
#         config_data = yaml.load(stream, Loader=yaml.SafeLoader)
#
#     # Resolve variables like ${base_data_dir}
#     config_data = resolve_config_variables(config_data)
#     sys.modules['models'] = models
#     state = torch.load(config_data['data']['checkpoint_path'], map_location=device, weights_only=False)
#     model = state['model']
#     model.to(device)
#
#
#     # Dataset without any output transformation
#     dataset = AIA_GOESDataset(
#         aia_dir=config_data['data']['aia_dir']+ '/test',
#         sxr_dir=config_data['data']['sxr_dir']+ '/test',
#     )
#
#     df = pd.DataFrame(columns=['Timestamp','Prediction', 'Ground'])# If you have a specific normalization for SXR, load it
#
#     times = dataset.samples
#     sxr_norm = np.load(config_data['data']['sxr_norm_path'])
#
#     timestamp = []
#     predictions = []
#     ground = []
#
#     for i, (prediction, sxr, weight) in enumerate(predict_log_outputs(model, dataset, 1, times)):
#         pred = unnormalize_sxr(prediction, sxr_norm)
#         predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
#         ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
#         timestamp.append(str(times[i]))
#         w = weight.numpy()
#         np.savetxt(config_data['weight_path']+f"{times[i]}", w, delimiter=",")
#         print(f"Processed {i+1}/{len(times)}: Timestamp={times[i]}, Prediction={pred}, Ground Truth={sxr}")
#
#
#
#     output_df = pd.DataFrame({
#         'Timestamp': timestamp,
#         'Predictions': predictions,
#         'ground_truth': ground
#     })
#     print(output_df)
#     output_df.to_csv(config_data['output_path'], index=False)
#
#     #f.write(f"{times[i]},{sxr[0]},{pred[0]}\n")
#     print(f"Predictions saved to {config_data['output_path']}")
#     # print("These are raw model outputs in log10 space before any exponentiation")
#
# if __name__ == '__main__':
#     main()