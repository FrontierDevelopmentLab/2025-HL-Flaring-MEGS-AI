import collections.abc

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
# Now import hyper
import numpy as np
from astropy.visualization import ImageNormalize, AsinhStretch
from itipy.data.dataset import StackDataset, get_intersecting_files, AIADataset
from itipy.data.editor import BrightestPixelPatchEditor, sdo_norms
import os
from multiprocessing import Pool
from tqdm import tqdm

# Configuration for all wavelengths to process
# Load configuration from environment or use defaults
import os
import json

def load_config():
    """Load configuration from environment or use defaults."""
    if 'PIPELINE_CONFIG' in os.environ:
        try:
            config = json.loads(os.environ['PIPELINE_CONFIG'])
            return config
        except:
            pass
    
    # Default configuration
    return {
        'iti': {
            'wavelengths': [94, 131, 171, 193, 211, 304],
            'input_folder': '/mnt/data/AUGUST/SDO-AIA-timespan',
            'output_folder': '/mnt/data/AUGUST/AIA_ITI'
        }
    }

config = load_config()
wavelengths = config['iti']['wavelengths']
base_input_folder = config['iti']['input_folder']
output_folder = config['iti']['output_folder']
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

    def __init__(self, data, patch_shape=None, wavelengths=None, resolution=2048, ext='.fits', allow_errors=False, **kwargs):
        wavelengths = [171, 193, 211, 304, 6173, ] if wavelengths is None else wavelengths
        if isinstance(data, list):
            paths = data
        else:
            paths = get_intersecting_files(data, wavelengths, ext=ext, **kwargs)
        ds_mapping = {94: AIADataset, 131: AIADataset, 171: AIADataset, 193: AIADataset, 211: AIADataset,
                      304: AIADataset, 335: AIADataset}
        data_sets = [ds_mapping[wl_id](files, wavelength=wl_id, resolution=resolution, ext=ext, allow_errors=allow_errors)
                     for wl_id, files in zip(wavelengths, paths)]
        super().__init__(data_sets, **kwargs)
        if patch_shape is not None:
            self.addEditor(BrightestPixelPatchEditor(patch_shape))


# Check if we need to process anything before loading the dataset
def check_existing_files():
    """Check how many files already exist without loading the full dataset"""
    # Get file list from the base folder to estimate total samples
    from itipy.data.dataset import get_intersecting_files
    files = get_intersecting_files(base_input_folder, wavelengths, ext='.fits')
    if not files or len(files) == 0:
        return 0, 0
    
    # Count existing output files - need to check for each wavelength combination
    existing_count = 0
    total_expected = len(files[0])  # All wavelength lists should have same length
    
    # Check each time step (index across all wavelengths)
    for i in range(total_expected):
        # Check if output file exists for this time step
        # The output filename should be based on the first wavelength's filename
        first_wl_file = files[0][i]  # Use first wavelength as reference
        base_name = os.path.splitext(os.path.basename(first_wl_file))[0]
        # Remove wavelength suffix if present (e.g., "_171" from filename)
        if '_' in base_name:
            base_name = '_'.join(base_name.split('_')[:-1])
        output_path = os.path.join(output_folder, base_name) + '.npy'
        
        if os.path.exists(output_path):
            existing_count += 1
    
    return existing_count, total_expected

# Check existing files first
existing_files, total_expected = check_existing_files()
print(f"Found {existing_files} existing files out of {total_expected} expected files")

if existing_files >= total_expected:
    print("All files already processed. Nothing to do.")
else:
    print(f"Need to process {total_expected - existing_files} remaining files")
    
    # Only load the dataset if we need to process files
    aia_dataset = SDODataset_flaring(data=base_input_folder, wavelengths=wavelengths, resolution=512, allow_errors=True)
    
    # Filter out indices that already have processed files
    def get_unprocessed_indices():
        unprocessed = []
        for i in range(len(aia_dataset)):
            file_path = os.path.join(output_folder, aia_dataset.getId(i)) + '.npy'
            if not os.path.exists(file_path):
                unprocessed.append(i)
        return unprocessed

    def save_sample(i):
        try:
            data = aia_dataset[i]
            file_path = os.path.join(output_folder, aia_dataset.getId(i)) + '.npy'
            np.save(file_path, data)
        except Exception as e:
            print(f"Warning: Could not process sample {i} (ID: {aia_dataset.getId(i)}): {e}")
            return  # Skip this sample and continue with the next one

    # Get only unprocessed indices
    unprocessed_indices = get_unprocessed_indices()
    print(f"Processing {len(unprocessed_indices)} unprocessed samples")

    if unprocessed_indices:
        with Pool(processes=os.cpu_count()) as pool:
            list(tqdm(pool.imap(save_sample, unprocessed_indices), total=len(unprocessed_indices)))
            print("AIA data processing completed.")
    else:
        print("All samples already processed. Nothing to do.")