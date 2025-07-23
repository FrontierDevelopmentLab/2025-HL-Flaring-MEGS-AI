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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def predict_log_outputs(model, dataset, batch_size=1, times=None):
    """Generator yielding raw log-space model outputs"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in loader:
            # Correct unpacking based on your data structure
            aia_imgs = batch[0] # Get aia_img from inputs
            sxr = batch[1]
            # Move to device (it's already a tensor)
            aia_imgs = aia_imgs.to(device)

            # Get model predictions
            pred = model(aia_imgs)
            if len(pred) > 1:
                pred = pred[0][0]
                #weights = pred[1]

            # Move to CPU and convert to numpy before yielding
            yield (pred.cpu().numpy(), sxr.cpu().numpy())

def main():
    def resolve_config_variables(config_dict):
        """Recursively resolve ${variable} references within the config"""

        # Extract variables defined at root level (like base_data_dir, base_checkpoint_dir)
        variables = {}
        for key, value in config_dict.items():
            if isinstance(value, str) and not value.startswith('${'):
                variables[key] = value

        def substitute_value(value, variables):
            if isinstance(value, str):
                # Replace ${var_name} with actual values
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
    args = parser.parse_args()

    # Load config with variable substitution
    with open(args.config, 'r') as stream:
        config_data = yaml.load(stream, Loader=yaml.SafeLoader)

    # Resolve variables like ${base_data_dir}
    config_data = resolve_config_variables(config_data)
    sys.modules['models'] = models
    state = torch.load(config_data['data']['checkpoint_path'], map_location=device, weights_only=False)
    model = state['model']
    model.to(device)


    # Dataset without any output transformation
    dataset = AIA_GOESDataset(
        aia_dir=config_data['data']['aia_dir']+ '/test',
        sxr_dir=config_data['data']['sxr_dir']+ '/test',
    )

    df = pd.DataFrame(columns=['Timestamp','Prediction', 'Ground'])# If you have a specific normalization for SXR, load it

    times = dataset.samples
    sxr_norm = np.load(config_data['data']['sxr_norm_path'])
    # Save log-space predictions

    timestamp = []
    predictions = []
    ground = []

    for i, (log_pred, sxr) in enumerate(predict_log_outputs(model, dataset, 1, times)):
        pred = unnormalize_sxr(log_pred, sxr_norm)

        predictions.append(pred.item() if hasattr(pred, 'item') else float(pred))
        ground.append(sxr.item() if hasattr(sxr, 'item') else float(sxr))
        timestamp.append(str(times[i]))
        print(f"Processed {i+1}/{len(times)}: Timestamp={times[i]}, Prediction={pred}, Ground Truth={sxr}")



    output_df = pd.DataFrame({
        'Timestamp': timestamp,
        'Predictions': predictions,
        'ground_truth': ground
    })
    print(output_df)
    output_df.to_csv(config_data['output_path'], index=False)

    #f.write(f"{times[i]},{sxr[0]},{pred[0]}\n")
    print(f"Predictions saved to {config_data['output_path']}")
    # print("These are raw model outputs in log10 space before any exponentiation")

if __name__ == '__main__':
    main()