import os
import pandas as pd
import shutil
from datetime import datetime

aia_data_dir = "/mnt/data/ML-Ready/AIA-Data/"
sxr_data_dir = "/mnt/data/ML-Ready/GOES-18-SXR-B/"
flares_event_dir = "/mnt/data/ML-Ready/flares_event_dir/"
non_flares_event_dir = "/mnt/data/ML-Ready/non_flares_event_dir/"
mixed_data_dir = "/mnt/data/ML-Ready/mixed_data/"
flare_events_csv = "/mnt/data/flare_list/flare_events_2023-07-01_2023-08-15.csv"

# Create train, val, test subdirectories under flaring and non-flaring
for base_dir in [flares_event_dir, non_flares_event_dir, mixed_data_dir]:
    os.makedirs(os.path.join(base_dir, "AIA"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "SXR"), exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, "AIA", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "SXR", split), exist_ok=True)

# Load flare events
flare_event = pd.read_csv(flare_events_csv)

# Create list of flare event time ranges
flaring_eve_list = []
for i, row in flare_event.iterrows():
    start_time = pd.to_datetime(row['event_starttime'])
    end_time = pd.to_datetime(row['event_endtime'])
    flaring_eve_list.append((start_time, end_time))

# Define date ranges for splits
train_range = (datetime(2023, 7, 1,0,0,0), datetime(2023, 7, 25,23,59,59))
val_range = (datetime(2023, 7, 27,0,0,0), datetime(2023, 7, 31,23,59,59))
test_range = (datetime(2023, 8, 1,0,0,0), datetime(2023, 8, 15,23,59,59))

# Get list of files in data_dir
data_list = os.listdir(aia_data_dir)

for file in data_list:
    try:
        aia_time = pd.to_datetime(file.split(".")[0])
    except ValueError:
        print(f"Skipping file {file}: Invalid timestamp format")
        continue

    # Determine if the file is during a flare event
    is_flaring = any(start <= aia_time <= end for start, end in flaring_eve_list)
    base_dir = flares_event_dir if is_flaring else non_flares_event_dir

    # Determine split based on date
    if train_range[0] <= aia_time <= train_range[1]:
        split_dir = "train"
    elif val_range[0] <= aia_time <= val_range[1]:
        split_dir = "val"
    elif test_range[0] <= aia_time <= test_range[1]:
        split_dir = "test"
    else:
        print(f"Skipping file {file}: Outside date range")
        continue

    # Copy file to appropriate directory
    src_aia = os.path.join(aia_data_dir, file)
    src_sxr = os.path.join(sxr_data_dir, file)
    dst_aia = os.path.join(base_dir, "AIA", split_dir, file)
    dst_sxr = os.path.join(base_dir, "SXR", split_dir, file)

    if not os.path.exists(dst_aia):
        shutil.copy(src_aia, dst_aia)
        print(f"Copied {file} to {dst_aia} and {dst_sxr}")
    else:
        print(f"File {dst_aia} already exists, skipping copy.")
    if not os.path.exists(dst_sxr):
        shutil.copy(src_sxr, dst_sxr)
    else:
        print(f"File {dst_sxr} already exists, skipping copy.")

