import os
import pandas as pd
import shutil
from datetime import datetime


data_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data"
flares_event_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/flares_event_dir"
non_flares_event_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/non_flares_event_dir"
flare_events_csv = "/mnt/data/flare_list/flare_events_2023-07-01_2023-08-15.csv"

os.makedirs(flares_event_dir, exist_ok=True)
os.makedirs(non_flares_event_dir, exist_ok=True)

flare_event = pd.read_csv(flare_events_csv)


flaring_eve_list = []
for i, row in flare_event.iterrows():
    start_time = pd.to_datetime(row['event_starttime'])
    end_time = pd.to_datetime(row['event_endtime'])
    flaring_eve_list.append((start_time, end_time))

data_list = os.listdir(data_dir)

for file in data_list:
    try:
        aia_time = pd.to_datetime(file.split(".")[0])
    except ValueError:
        print(f"Skipping file {file}: Invalid timestamp format")
        continue

    # Check if the file's time falls within any flare event
    is_flaring = any(start <= aia_time <= end for start, end in flaring_eve_list)
    if is_flaring:
        src = os.path.join(data_dir, file)
        dst = os.path.join(flares_event_dir, file)
        shutil.copy(src, dst)
        print(f"Copied {file} to {flares_event_dir}")
    else:
        src = os.path.join(data_dir, file)
        dst = os.path.join(non_flares_event_dir, file)
        shutil.copy(src, dst)
        print(f"Copied {file} to {non_flares_event_dir}")

train_range = (datetime(2023, 7, 1), datetime(2023, 7, 20))
val_range = (datetime(2023, 7, 21), datetime(2023, 8, 5))
test_range = (datetime(2023, 8, 6), datetime(2023, 8, 15))

print(train_range[0],train_range[1])
# Create train, val, test subdirectories under flaring and non-flaring
for base_dir in [flares_event_dir, non_flares_event_dir]:
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    # Get list of files in the current directory (flaring or non-flaring)
    file_list = os.listdir(base_dir)


    for file in file_list:
        try:
            aia_time = pd.to_datetime(file.split(".")[0])
        except ValueError:
            print(f"Skipping file {file} in {base_dir}: Invalid timestamp format")
            continue

        # Determine split based on date
        if train_range[0] <= aia_time <= train_range[1]:
            split_dir = "train"
        elif val_range[0] <= aia_time <= val_range[1]:
            split_dir = "val"
        elif test_range[0] <= aia_time <= test_range[1]:
            split_dir = "test"
        else:
            print(f"Skipping file {file} in {base_dir}: Outside date range")
            continue

        # Move file to appropriate split directory
        src = os.path.join(base_dir, file)
        dst = os.path.join(base_dir, split_dir, file)
        shutil.move(src, dst)
        print(f"Moved {file} to {base_dir}/{split_dir}")