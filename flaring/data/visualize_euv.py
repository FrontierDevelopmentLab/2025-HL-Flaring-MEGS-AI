import os
import glob
import numpy as np
from datetime import datetime
from skimage.transform import resize
from matplotlib import colormaps as cm
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image
import sunpy.visualization.colormaps
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings("ignore")

data_dir = "/mnt/data2/AIA_processed"
output_base = "/mnt/data2/frames"
wavelengths = ['94', '131', '171', '193', '211', '304']

## Ensure output folders exist
for wl in wavelengths:
    os.makedirs(os.path.join(output_base, wl), exist_ok=True)

## Load and sort all files by time
npy_files = sorted(glob.glob(f"{data_dir}/*.npy"))
print(f"Found {len(npy_files)} .npy files.")

## Function to remove corrupted files with os.remove
def remove_np_file(filepath):
    """
    Remove a .npy file if it is corrupted or cannot be loaded.
    Parameters
    ----------
    filepath

    Returns
    -------

    """
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading file '{filepath}': {e}")
        print("File appears to be corrupted. Removing it.")
        try:
            os.remove(filepath)
            print(f"Deleted corrupted file: {filepath}")
        except Exception as delete_error:
            print(f"Failed to delete file: {delete_error}")
        return None

## Process single file
def process_file(fpath):
    """
    Process a single .npy file, visualize the AIA frames, and save them as images.
    Parameters
    ----------
    fpath

    Returns
    -------

    """
    file = np.load(fpath)
    fig,ax = plt.subplots(1,1,figsize = (10, 10), dpi=300)
    for ch in range(len(wavelengths)):
        wl = wavelengths[ch]
        #out_path = os.path.join(output_base, wl, f"aia_{wl}_{fpath.split('.')[0]}")
        # Apply colormap and convert to RGB
        cmap = plt.get_cmap(f'sdoaia{wl}')
        plt.imshow(file[ch, :, :], cmap = cmap, origin ="lower", vmax =1, vmin = -1)
        plt.title(fpath.split('.')[0].split('/')[4], fontsize = 15)
        plt.savefig(os.path.join(output_base, str(wl), fpath.split('.')[0].split('/')[4])+ ".jpg") ## Modified the code logic
    plt.close()
## Main function
if __name__ == "__main__":
    #for i in tqdm(range(53314, len(npy_files))):
    #    npy_files[i] = remove_np_file(npy_files[i])
    with Pool(processes=90) as pool:
        # Remove corrupted files
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files), desc="Saving AIA frames", ncols=100))