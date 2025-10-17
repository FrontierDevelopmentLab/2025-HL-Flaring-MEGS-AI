
import numpy as np
from pathlib import Path
import glob
import os

def compute_sxr_norm(sxr_dir):
    """
    Compute mean and standard deviation of log10-transformed SXR values.
    Handles both single SXR values and SXR A/B arrays.

    Args:
        sxr_dir (str): Path to directory containing SXR .npy files.

    Returns:
        dict or tuple: If SXR A/B data detected, returns {'a': (mean_a, std_a), 'b': (mean_b, std_b)}.
                      If single SXR data, returns (mean, std) for backward compatibility.
    """
    sxr_dir = Path(sxr_dir).resolve()
    print(f"Checking SXR directory: {sxr_dir}")
    if not sxr_dir.is_dir():
        raise FileNotFoundError(f"SXR directory does not exist or is not a directory: {sxr_dir}")

    # Use glob for case-insensitive matching
    sxr_files = sorted(glob.glob(os.path.join(sxr_dir, "*.npy")))
    print(f"Found {len(sxr_files)} SXR files in {sxr_dir}")
    if len(sxr_files) == 0:
        print(f"No files matching '*.npy' found. Listing directory contents:")
        print(os.listdir(sxr_dir)[:10])  # Show first 10 files
        raise ValueError(f"No SXR files found in {sxr_dir}")

    sxr_a_values = []
    sxr_b_values = []
    single_sxr_values = []
    data_format_detected = None

    for f in sxr_files:
        try:
            sxr_data = np.load(f)
            
            if sxr_data.size == 1:
                # Single SXR value
                sxr_val = float(np.atleast_1d(sxr_data).flatten()[0])
                if not np.isfinite(sxr_val) or sxr_val < 0:
                    print(f"Skipping invalid SXR value in {f}: {sxr_val}")
                    continue
                single_sxr_values.append(np.log10(sxr_val))
                if data_format_detected is None:
                    data_format_detected = "single"
                    
            elif sxr_data.size == 2:
                # SXR A/B array [SXR-A, SXR-B]
                sxr_a_val = float(sxr_data[0])
                sxr_b_val = float(sxr_data[1])
                
                if not np.isfinite(sxr_a_val) or sxr_a_val < 0:
                    print(f"Skipping invalid SXR-A value in {f}: {sxr_a_val}")
                else:
                    sxr_a_values.append(np.log10(sxr_a_val))
                    
                if not np.isfinite(sxr_b_val) or sxr_b_val < 0:
                    print(f"Skipping invalid SXR-B value in {f}: {sxr_b_val}")
                else:
                    sxr_b_values.append(np.log10(sxr_b_val))
                    
                if data_format_detected is None:
                    data_format_detected = "dual"
                    
            else:
                print(f"Skipping file {f} with unexpected data size: {sxr_data.size}")
                continue
                
        except Exception as e:
            print(f"Failed to load SXR file {f}: {e}")
            continue

    # Determine which format we're working with
    if data_format_detected == "single":
        if len(single_sxr_values) == 0:
            raise ValueError(f"No valid single SXR values found in {sxr_dir}")
        
        single_sxr_values = np.array(single_sxr_values)
        mean = np.mean(single_sxr_values)
        std = np.std(single_sxr_values)
        print(f"Computed single SXR normalization: mean={mean}, std={std}")
        return mean, std
        
    elif data_format_detected == "dual":
        if len(sxr_a_values) == 0 or len(sxr_b_values) == 0:
            raise ValueError(f"No valid SXR A/B values found in {sxr_dir}")
        
        sxr_a_values = np.array(sxr_a_values)
        sxr_b_values = np.array(sxr_b_values)
        
        mean_a = np.mean(sxr_a_values)
        std_a = np.std(sxr_a_values)
        mean_b = np.mean(sxr_b_values)
        std_b = np.std(sxr_b_values)
        
        print(f"Computed SXR-A normalization: mean={mean_a}, std={std_a}")
        print(f"Computed SXR-B normalization: mean={mean_b}, std={std_b}")
        
        return {
            'a': (mean_a, std_a),
            'b': (mean_b, std_b)
        }
    
    else:
        raise ValueError(f"No valid SXR data found in {sxr_dir}")

if __name__ == "__main__":
    # Update this path to your real data SXR directory
    sxr_dir = "/mnt/data/PAPER_DATA_A_B/SXR/train"  # Replace with actual path
    output_dir = "/mnt/data/PAPER_DATA_A_B/SXR"  # Directory to save normalization files
    
    print("Computing SXR normalization...")
    sxr_norm = compute_sxr_norm(sxr_dir)
    
    if isinstance(sxr_norm, dict):
        # SXR A/B data detected - save separate files
        print("SXR A/B data detected. Saving separate normalization files...")
        
        # Save SXR-A normalization
        sxr_a_norm = np.array([sxr_norm['a'][0], sxr_norm['a'][1]])
        sxr_a_path = os.path.join(output_dir, "normalized_sxr_a.npy")
        np.save(sxr_a_path, sxr_a_norm)
        print(f"Saved SXR-A normalization to {sxr_a_path}")
        
        # Save SXR-B normalization
        sxr_b_norm = np.array([sxr_norm['b'][0], sxr_norm['b'][1]])
        sxr_b_path = os.path.join(output_dir, "normalized_sxr_b.npy")
        np.save(sxr_b_path, sxr_b_norm)
        print(f"Saved SXR-B normalization to {sxr_b_path}")
        
        print("Use these paths in your config:")
        print(f"  sxr_norm_paths:")
        print(f"    a: \"{sxr_a_path}\"")
        print(f"    b: \"{sxr_b_path}\"")
        
    else:
        # Single SXR data detected - save single file
        print("Single SXR data detected. Saving single normalization file...")
        sxr_norm_array = np.array([sxr_norm[0], sxr_norm[1]])
        sxr_path = os.path.join(output_dir, "normalized_sxr.npy")
        np.save(sxr_path, sxr_norm_array)
        print(f"Saved SXR normalization to {sxr_path}")
        
        print("Use this path in your config:")
        print(f"  sxr_norm_path: \"{sxr_path}\"")