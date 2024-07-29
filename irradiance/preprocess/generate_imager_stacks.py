import argparse
import logging
import os
import sys
from os.path import exists
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from irradiance.data.utils import loadMapStack, str2bool


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

def load_map_stack(imager_stack):
    """
    Function that loads a stack of EUV images that is considered concurrent

    Parameters
    ----------
    imager_stack : 
        list with filenames to load into stack

    Returns
    -------
    np.array
        returns numpy array with stacks
    """    
    # Extract filename from index_imager_i (remove imager_path)
    filename = (imager_stack[0].replace(imager_path, '')).split('_')[1]
    # Replace .fits by .npy
    filename = filename.replace('.fits', '.npy')

    output_file = matches_stacks + '/AIA_' + filename
    # print(output_file)

    if exists(output_file):
        LOG.info(f'{filename} exists.')
        imager_stack = np.load(output_file)
        return imager_stack
    else:
        # if imager_extension_path is not None:
        #     calibration ='aiapy'
        # else:
        #     calibration = 'auto'
        calibration = 'aiapy'
        imager_stack = loadMapStack(imager_stack, resolution=imager_resolution, remove_nans=True,
                                    map_reproject=imager_reproject, aia_preprocessing=True, calibration=calibration,
                                    apply_norm=False, percentile_clip=0.25)
        # Save stack
        np.save(output_file, imager_stack)
        data = np.asarray(imager_stack)
        imager_stack = None

        return data


def parse_args():
    # Commands 
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-imager_path', dest='imager_path', type=str,
                   default="/mnt/disks/observational_data/AIA",
                   help='imager_path')
    p.add_argument('-imager_resolution', dest='imager_resolution', type=int,
                   default=256, help='Resolution of the output images')
    p.add_argument('-imager_reproject', dest='imager_reproject', type=str2bool, default=False,
                   help='Use reprojection from heliographic map (remove off-limb).')
    p.add_argument('-imager_stats', dest='imager_stats', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256/stats.npz",
                   help='Stats.')
    p.add_argument('-matches_csv', dest='matches_csv', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                   help='matches_table')
    p.add_argument('-matches_output', dest='matches_output', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256/matches_eve_aia.csv",
                   help='Updated matches')
    p.add_argument('-matches_stacks', dest='matches_stacks', type=str,
                   default="/mnt/disks/preprocessed_data/AIA/256",
                   help='Stacks for matches')
    args = p.parse_args()
    return args


if __name__ == "__main__":
    # Parser
    args = parse_args()
    imager_path = args.imager_path
    imager_resolution = args.imager_resolution
    imager_stats = args.imager_stats
    imager_reproject = args.imager_reproject  
    matches_file = args.matches_csv
    matches_stacks = args.matches_stacks
    matches_output = args.matches_output

    stats = {}

    # Load indices
    matches = pd.read_csv(matches_file)

    # Extract filenames for stacks
    imager_files = []
    imager_columns = [col for col in matches.columns if 'AIA' in col]

    for index, row in tqdm(matches.iterrows()):
        imager_files.append(row[imager_columns].tolist())  # (channel, files)

    # Path for outputs
    os.makedirs(matches_stacks, exist_ok=True)

    # Extract filename from index_imager_i (remove imager_path)
    converted_file_paths = [matches_stacks + '/AIA_' +
                            ((imager_files[i][0].replace(imager_path, '')).split('_')[1]).replace('.fits', '.npy')
                            for i in range(len(imager_files))]

    # Stacks
    print('Saving stacks')
    data = np.stack(process_map(load_map_stack, imager_files, chunksize=5,
                                total=len(imager_files)))
    imager_min = np.min(data, axis=(0, 2, 3), keepdims=False)
    imager_max = np.max(data, axis=(0, 2, 3), keepdims=False)
    imager_mean = np.mean(data, axis=(0, 2, 3), keepdims=False)
    imager_std = np.stack([np.std(data[:, wl, :, :], keepdims=False) for wl in range(data.shape[1])])
    data = None

    stats['AIA'] = {'mean': imager_mean, 'std': imager_std, 'min': imager_min, 'max': imager_max}

    print('Saving Matches')
    np.savez(imager_stats, **stats)
    
    matches['aia_stack'] = converted_file_paths
    matches.to_csv(matches_output, index=False)
