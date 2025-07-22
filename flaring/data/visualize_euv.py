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

corrupted_files = []
def load_npy(filepath):
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"Skipping corrupt file: {filepath} — {e}")
        corrupted_files.append((filepath, str(e)))

def save_corrupted_files_to_csv(csv_path=output_base+"/corrupted_files.csv"):
    if corrupted_files:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filepath", "Error"])
            writer.writerows(corrupted_files)
        print(f"\nSaved {len(corrupted_files)} corrupted file entries to {csv_path}")
    else:
        print("\nNo corrupted files to save.")

## Process single file
def process_file(fpath):
    file = load_npy(fpath)
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
    save_corrupted_files_to_csv()
## Main function
if __name__ == "__main__":
    with Pool(processes=90) as pool:
        list(tqdm(pool.imap(process_file, npy_files), total=len(npy_files), desc="Saving AIA frames", ncols=100))