import numpy as np
import os


aia = os.listdir("/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data")


target_dates = ["2023-07-07", "2023-07-11", "2023-07-20", "2023-08-01"]

aia_dict = {}
aia_dict[0] = []
aia_dict[1] = []
aia_dict[2] = []
aia_dict[3] = []
aia_dict[4] = []
aia_dict[5] = []

for i, file in enumerate(aia):
    if file.split("T")[0] in target_dates:
        aia_data = np.load("/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data/"+file)
        aia_dict[0].append(aia_data[0].flatten())
        aia_dict[1].append(aia_data[1].flatten())
        aia_dict[2].append(aia_data[2].flatten())
        aia_dict[3].append(aia_data[3].flatten())
        aia_dict[4].append(aia_data[4].flatten())
        aia_dict[5].append(aia_data[5].flatten())
    print(f"Processed {i+1}/{len(aia)} files", end='\r')

def percentile(data, perc):
    return np.percentile(data, perc)

percentile_dict = {0:[percentile(aia_dict[0], 95), percentile(aia_dict[0], 99.5)],1: [percentile(aia_dict[1], 95), percentile(aia_dict[1], 99.5)], 2: [percentile(aia_dict[2], 95), percentile(aia_dict[2], 99.5)], 3: [percentile(aia_dict[3], 95), percentile(aia_dict[3], 99.5)], 4: [percentile(aia_dict[4], 95), percentile(aia_dict[4], 99.5)], 5: [percentile(aia_dict[5], 95), percentile(aia_dict[5], 99.5)]}

print(percentile_dict)

# target_dates = ["2023-07-07", "2023-07-11", "2023-07-20", "2023-08-01"]
# percentiles_to_check = [90, 95, 99, 99.5, 99.9, 99.99, 100]
#
# for file in aia_files:
#     if any(date in file for date in target_dates):
#         filepath = os.path.join(input_dir, file)
#         data = np.load(filepath)
#
#         # Flatten and clean
#         data_flat = data.flatten()
#         data_flat = data_flat[np.isfinite(data_flat)]  # remove NaNs/Infs
#
#         print(f"\nFile: {file}")
#         print(f"  Shape: {data.shape}")
#         for p in percentiles_to_check:
#             val = np.percentile(data_flat, p)
#             print(f"  {p:>6.2f}th percentile: {val:.2f}")


# aia_hist = {}
# for c, c_files in zip(sdo_channels, aia_files):
#     c_files = c_files[::len(c_files) // 100]
#     with Pool(12) as p:
#         data = [np.ravel(m) for m in tqdm(p.imap_unordered(getAIAData, c_files), total=len(c_files))]
#         data = np.concatenate(data)
#     threshold = np.nanmedian(data) + np.nanstd(data)
#     data[data > threshold] = np.nan
#     aia_hist[c] = [np.nanmean(data), np.nanstd(data)]