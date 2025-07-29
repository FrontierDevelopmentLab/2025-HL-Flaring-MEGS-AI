import collections.abc
import shutil

import pandas as pd

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# Now import hyper
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from itipy.data.dataset import StackDataset, get_intersecting_files, AIADataset
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, MapToDataEditor, BrightestPixelPatchEditor
import os
from multiprocessing import Pool
from tqdm import tqdm

# Configuration for all wavelengths to process
wavelengths = [94, 131, 171, 193, 211, 304]
base_input_folder = '/mnt/data2/SDO-AIA'
output_folder = '/mnt/data2/AIA_processed'
os.makedirs(output_folder, exist_ok=True)

sdo_norms = {
    '94': ImageNormalize(vmin=0, vmax=np.float32(16.560747), stretch=AsinhStretch(0.005), clip=True),
    '131': ImageNormalize(vmin=0, vmax=np.float32(75.84181), stretch=AsinhStretch(0.005), clip=True),
    '171': ImageNormalize(vmin=0, vmax=np.float32(1536.1443), stretch=AsinhStretch(0.005), clip=True),
    '193': ImageNormalize(vmin=0, vmax=np.float32(2288.1), stretch=AsinhStretch(0.005), clip=True),
    '211': ImageNormalize(vmin=0, vmax=np.float32(1163.9178), stretch=AsinhStretch(0.005), clip=True),
    '304': ImageNormalize(vmin=0, vmax=np.float32(401.82352), stretch=AsinhStretch(0.001), clip=True),
}


class SDODataset_flaring(StackDataset):
    """
    Dataset for SDO data

    Args:
        data: Data
        patch_shape (tuple): Patch shape
        wavelengths (list): List of wavelengths
        resolution (int): Resolution
        ext (str): File extension
        **kwargs: Additional arguments
    """

    def __init__(self, data, patch_shape=None, wavelengths=None, resolution=2048, ext='.fits', **kwargs):
        wavelengths = [171, 193, 211, 304, 6173, ] if wavelengths is None else wavelengths
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)
        ds_mapping = {94: AIADataset, 131: AIADataset, 171: AIADataset, 193: AIADataset, 211: AIADataset,
                      304: AIADataset}
        data_sets = [ds_mapping[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext)
                     for wl_id, files in zip(wavelengths, paths)]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


aia_dataset = SDODataset_flaring(data=base_input_folder, wavelengths=wavelengths, resolution=512)

def save_sample(i):
    data = aia_dataset[i]
    file_path = os.path.join(output_folder, aia_dataset.getId(i)) + '.npy'
    if os.path.exists(file_path):
        return  # Skip if file already exists
    np.save(file_path, data)

with Pool(processes=90) as pool:
    list(tqdm(pool.imap(save_sample, range(len(aia_dataset))), total=len(aia_dataset)))