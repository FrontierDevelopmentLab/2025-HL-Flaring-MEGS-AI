import os
import pandas as pd
from sunpy.net import Fido
from sunpy.net import attrs as a
import numpy as np
import matplotlib.pyplot as plt
import shutil

## Separate the data folder into flare and quiet data based on the dates
#/mnt/data/flare_list/flare_events_2023-07-01_2023-08-15.csv
#/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data
## which dates to use for flaring, non-flaring
data_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data"
flares_event_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/flares_event_dir"
non_flares_event_dir = "/mnt/data/ML-Ready-Data-No-Intensity-Cut/non_flares_event_dir"
flare_events_csv = "/mnt/data/flare_list/flare_events_2023-07-01_2023-08-15.csv"

flare_event = pd.read_csv(flare_events_csv)

flare_event.head()

os.makedirs(flares_event_dir, exist_ok=True)
os.makedirs(non_flares_event_dir, exist_ok=True)

flaring_eve_list = []
non_flaring_eve_list = []

for i, row in flare_event.iterrows():
    start_time = pd.to_datetime(row['event_starttime'])
    end_time = pd.to_datetime(row['event_endtime'])

    # Create a date range for the flare event
    date_range = pd.date_range(start=start_time, end=end_time, freq='1h')

    # Add each date in the range to the respective list
    #(start_time, end_time)
    flaring_eve_list.append((start_time, end_time))
#print(flaring_eve_list)

## Get the list of file names from the data directory
data_list = os.listdir(data_dir)

# create for loop to iterate through the list of files in the data directory
for file in data_list:
    aia_time = pd.to_datetime(file.split(".")[0])
    print(aia_time)
    # Check if the file's time falls within any flare event
    is_flaring = any(start <= aia_time <= end for start, end in flaring_eve_list)
    if is_flaring:
        # Move to flaring directory
        src = os.path.join(data_dir, file)
        dst = os.path.join(flares_event_dir, file)
        shutil.copy(src, dst)
    else:
        # Move to non-flaring directory
        src = os.path.join(data_dir, file)
        dst = os.path.join(non_flares_event_dir, file)
        shutil.copy(src, dst)

