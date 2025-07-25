import collections.abc
import shutil

import pandas as pd
from tqdm import tqdm

# hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from itipy.data.dataset import get_intersecting_files
from astropy.io import fits

# Configuration for all wavelengths to process
wavelengths = [94, 131, 171, 193, 211, 304]
base_input_folder = '/mnt/data2/SDO-AIA'

aia_files = get_intersecting_files(base_input_folder, wavelengths)

aia_header = pd.DataFrame({'DATE-OBS': [], 'WAVELNTH': [], 'FILENAME': []})
hdu = fits.open(aia_files[0])
filename = pd.to_datetime(aia_files[0][0].split('/')[5].split('.')[0])

# loop through the files and extract headers
for i in tqdm(range(len(aia_files[0]))):
    hdu = fits.open(aia_files[0][i])
    header = hdu[1].header
    date_obs = header['DATE-OBS']
    wavelength = header['WAVELNTH']
    filename = pd.to_datetime(aia_files[0][i].split('/')[5].split('.')[0])
    aia_header = pd.concat([aia_header, pd.DataFrame({'DATE-OBS': [date_obs], 'WAVELNTH': [wavelength], 'FILENAME': [filename]})], ignore_index=True)


import pandas as pd
from astropy.io import fits
from multiprocessing import Pool
from tqdm import tqdm
import os

# Function to process a single file
def process_fits_file(file_path):
    try:
        with fits.open(file_path) as hdu:
            header = hdu[1].header
            date_obs = header['DATE-OBS']
            wavelength = header['WAVELNTH']
            filename = pd.to_datetime(os.path.basename(file_path).split('.')[0])
            return {'DATE-OBS': date_obs, 'WAVELNTH': wavelength, 'FILENAME': filename}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


from multiprocessing import freeze_support
#freeze_support()  # Required for Windows

file_list = aia_files[0]  # List of FITS file paths

with Pool(processes=os.cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_fits_file, file_list), total=len(file_list)))

# Filter out None results (in case of failed files)
results = [r for r in results if r is not None]

# Convert to DataFrame
aia_header = pd.DataFrame(results)

# Convert to timezone-naive
aia_header['DATE-OBS'] = aia_header['DATE-OBS'].dt.tz_localize(None)

#add a column for date difference between DATE-OBS and FILENAME
aia_header['DATE_DIFF'] = (pd.to_datetime(aia_header['FILENAME']) - pd.to_datetime(aia_header['DATE-OBS'])).dt.total_seconds()

# remove rows where DATE_DIFF is greater than plus or minus 60 seconds in a list
files_to_remove = aia_header[(aia_header['DATE_DIFF'] <= -60) | (aia_header['DATE_DIFF'] >= 60)]

print([os.path.join(base_input_folder, pd.to_datetime(files['FILENAME'].values[0]).strftime('%Y-%m-%dT%H:%M:%S')+".fits") for files in files_to_remove])

print(f"Number of files to remove: {len(files_to_remove)}")
for names in files_to_remove['FILENAME'].to_numpy():
    file_path = os.path.join(base_input_folder, "193/" + pd.to_datetime(names).strftime('%Y-%m-%dT%H:%M:%S') +".fits")
    if os.path.exists(file_path):
        os.makedirs("/mnt/data2/SDO-AIA_bad/removed_files/193", exist_ok=True)
        shutil.move(file_path, "/mnt/data2/SDO-AIA_bad/removed_files/193/")
        print(f"Removed file: {file_path}")
    else:
        print(f"File not found: {file_path}")