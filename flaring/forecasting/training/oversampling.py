import argparse
import re
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from data_loaders.SDOAIA_dataloader import AIA_GOESSequenceDataset
import yaml

"""
1. Load AIA+SXR training sequence dataset
2. Extracts the last-frame SXR value from each sequence
3. Compute class frequencies via log10(SXR) binning
4. Apply weightRandomSampler to favor underrepresented sxr events like(M/X class flares)
5. Plot original label distribution versus post-oversampling distribution
"""
### Example Code Run on the terminal
# python /home/aliso/flaring/forecasting/training/oversampling.py \
#   --config /path/to/config.yaml \
#   --output_plot /desired/path/oversampling_plot.png

def compute_weights_by_last_sxr(dataset, num_bins=10):
    """
    Compute sampling weights based on the last SXR value in each sequence
    """
    last_sxr = []
    for _, sxr_seq in dataset:
        #last_sxr.append(np.bincount(sxr_seq[:, -1])) ## get the last sxr value
        last_sxr.append(float(sxr_seq[-1]))  ## use this instead because sxq_seq  is a 1D tensor, not a 2D array
    log_sxr = np.log10(np.array(last_sxr)+1e-8)
    bins = np.linspace(log_sxr.min(), log_sxr.max(), num_bins+1)
    bin_indie = np.bincount(bin_indices, minlength=num_bins+2)
    weights = 1./(class_counts[bin_indices]+1e-6)
    return log_sxr, weights, bin_indie

## plot distributions
def plot_dist(log_sxr, weights, sampled_indices, output_path="oversampling_dist.png"):
    plt.figure(figsize=(12, 8))
    ## Subplots
    plt.subplot(1,2,1)
    plt.hist(log_sxr, bins=20, color="blue")
    plt.title("Original log10(SXR) Distribution")
    plt.xlabel("log10(SXR)")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    sampled_log_sxr = log_sxr[sampled_indices]
    plt.hist(sampled_log_sxr, bins=20, color="orange")
    plt.title("Oversampled log10(SXR) Distribution")
    plt.xlabel("log10(SXR)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved distribution plot to {output_path}")

### Create a function to resolve variables config.
def resolve_config_variables(config_dict):
    """
    Resolve ${variables} reference inside YAML config
    """
    variables={}
    for key, value in config_dict.items():
        if isinstance(value, str) and not value.startswith("$"):
            variables[key]=value

    def substitute_value(value, variables):
        if isinstance(value, str):
            pattern = r"\\$\\{([^]+)\\}"
            for match in re.finditer(pattern, value):
            var_name = match.group(1)
            if var_name in variables:
                value = value.replace(f"${{{var_name}}}", variables[var_name])
        return value

    def recursive_substitute(obj, variables):
        if isinstance(obj, dict):
            return {k: recursive_substitute(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_substitute(item, variables) for item in obj]
        else:
            return substitute_value(obj, variables)
    return recursive_substitute(config_dict, variables)

def main():
    parser=argparse.ArgumentParser(description = "Oversampling analysis for SXR label imbalance")
    # parser.add_argument("--aia_dir", type=str, required=True)
    # parser.add_argument("--sxr_dir", type=str, required=True)
    # parser.add_argument("--sxr_norm_path", type=str, required =True)
    # parser.add_argument("--sequence_length", type=int, default=12)
    # parser.add_argument("--stride", type=int, default=1)
    # parser.add_argument("--output_plot", type=str, default="oversampling_dist.png")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--output_plot", type=str, default ="oversampling_dist.png", help="Path to output plot")

    args = parser.parse_args()

    ## Load and resolve config
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = resolve_config_variables(config)

    ## Get path
    aia_train_dir = config["data"]["path"]["aia"] + "/train"
    sxr_train_dir = config["data"]["path"]["sxr"] + "/train"
    sxr_norm_path = config["data"]["path"]["sxr_norm"]

    ## Load normalization
    sxr_norm = np.load(args.sxr_norm_path)
    sxr_transform = lambda x: (np.log10(x+1e-8) - sxr_norm[0])/sxr_norm[1]

    ## Get sequence parameters
    seq_length = config["training"]["sequence"]["input_length"]
    stride = config["training"]["sequence"]["stride"]

    ## Load dataset
    dataset = AIA_GOESSequenceDataset(
        aia_dir = args.aia_dir,
        sxr_dir = args.sxr_dir,
        sequence_length = args.sequence_length,
        stride = args.stride,
        sxr_transform = sxr_transform,
    )

    ## Compute weights
    log_sxr, weights, _ = compute_weights_by_last_sxr(dataset)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    sampled_indices = list(sampler)

    ## Make distribution plot
    plot_dist(log_sxr, weights, sampled_indices, output_path=args.output_plot)

if __name__ == "__main__":
    main()





