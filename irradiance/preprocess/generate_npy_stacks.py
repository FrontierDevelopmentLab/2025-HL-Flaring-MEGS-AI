import argparse
import logging
import os
from glob import glob
from multiprocessing import Pool
from os.path import exists

import numpy as np
from tqdm import tqdm

from irradiance.data.utils import loadMapStack, str2bool

# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


def load_map_stack(file_stack):
    # Extract filename from index_aia_i (remove aia_path)
    filename = os.path.basename(file_stack[0])
    # Replace .fits by .npy
    filename = filename.replace(extension, '.npy')
    output_file = os.path.join(output_path, filename)

    if exists(output_file):
        LOG.info(f'{filename} exists.')
        return output_file

    try:
        file_stack = loadMapStack(file_stack, resolution=resolution, remove_nans=True,
                                  map_reproject=map_reproject, aia_preprocessing=aia_preprocessing)
    except Exception as ex:
        LOG.error(str(ex))
        return None
    # Save stack
    np.save(output_file, file_stack)
    return output_file


def parse_args():
    # Commands
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # p.add_argument('-eve_path', type=str, default="/home/miraflorista/sw-irradiance/data/EVE/EVE.json",
    #            help='eve_path')
    p.add_argument('-data_path', dest='data_path', type=str,
                   default="/mnt/nerf-data/nerf_data/stereo_2014_04_converted", help='data_path')
    p.add_argument('-output_path', dest='output_path', type=str, default="/mnt/converted_data/test_set/stereo_a",
                   help='out_path')
    p.add_argument('-map_reproject', dest='map_reproject', type=str2bool, default=False,
                   help='Use reprojection from heliographic map (remove off-limb).')
    p.add_argument('-aia_preprocessing', type=str2bool, default=False,
                   help='Use AIA preprocessing.')
    p.add_argument('-resolution', dest='resolution', type=int, default=256, help='Resolution of the output images')
    p.add_argument('-extension', type=str, default='_A.fits', help='Extension of the data files')
    args = p.parse_args()
    return args


if __name__ == "__main__":
    # Parser
    args = parse_args()
    # eve_path = args.eve_path
    data_path = args.data_path
    output_path = args.output_path
    resolution = args.resolution
    map_reproject = args.map_reproject
    aia_preprocessing = args.aia_preprocessing
    extension = args.extension

    # Extract filenames for stacks
    subfolders = os.listdir(data_path)
    data_files = [sorted(glob(os.path.join(data_path, dir, f'*{extension}'))) for dir in subfolders]

    # find intersecting files
    basenames = [[os.path.basename(f) for f in files] for files in data_files]
    # find valid files: [(2, 3, 4), (1, 3, 4)] ---> [(3, 4), (3, 4)]
    intersecting_basenames = set(basenames[0]).intersection(*basenames)
    data_files = [[f for f in files if os.path.basename(f) in intersecting_basenames] for files in data_files]
    data_files = np.array(data_files).transpose()

    # Path for output
    os.makedirs(output_path, exist_ok=True)

    # Stacks
    with Pool(os.cpu_count()) as p:
        converted_file_paths = [r for r in tqdm(p.imap(load_map_stack, data_files), total=len(data_files)) if
                                r is not None]
