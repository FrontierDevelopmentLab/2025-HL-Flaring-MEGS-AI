import numpy as np
import os


aia = os.listdir("/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data")


target_dates = ["2023-07-11","2023-07-15","2023-07-16", "2023-07-18" "2023-07-20", "2023-07-26", "2023-07-30", "2023-08-01", "2023-08-02", "2023-08-07", ]

aia_dict = {}
aia_dict[0] = []
aia_dict[1] = []
aia_dict[2] = []
aia_dict[3] = []
aia_dict[4] = []
aia_dict[5] = []

count = 0
for i, file in enumerate(aia):
    if file.split("T")[0] in target_dates:
        aia_data = np.load("/mnt/data/ML-Ready-Data-No-Intensity-Cut/AIA-Data/"+file)
        aia_dict[0].append(aia_data[0].flatten())
        aia_dict[1].append(aia_data[1].flatten())
        aia_dict[2].append(aia_data[2].flatten())
        aia_dict[3].append(aia_data[3].flatten())
        aia_dict[4].append(aia_data[4].flatten())
        aia_dict[5].append(aia_data[5].flatten())
        count = count + 1
        print("Flares: " + str(count) + "\n")
    print(f"\nProcessed {i+1}/{len(aia)} files", end='\r')

def percentile(data, perc):
    return np.percentile(data, perc)

percentile_dict = {0:[percentile(aia_dict[0], 95), percentile(aia_dict[0], 99.5)],1: [percentile(aia_dict[1], 95), percentile(aia_dict[1], 99.5)], 2: [percentile(aia_dict[2], 95), percentile(aia_dict[2], 99.5)], 3: [percentile(aia_dict[3], 95), percentile(aia_dict[3], 99.5)], 4: [percentile(aia_dict[4], 95), percentile(aia_dict[4], 99.5)], 5: [percentile(aia_dict[5], 95), percentile(aia_dict[5], 99.5)]}

print(percentile_dict)
#{0: [np.float32(5.0747647), np.float32(16.560747)], 1: [np.float32(24.491392), np.float32(75.84181)], 2: [np.float32(607.3201), np.float32(1536.1443)], 3: [np.float32(1021.83466), np.float32(2288.1)], 4: [np.float32(480.13672), np.float32(1163.9178)], 5: [np.float32(144.44502), np.float32(401.82352)]}
