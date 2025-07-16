#


#
# data_dir = "/mnt/data/ML-Ready/AIA-Data"
# flares_event_dir = "/mnt/data/ML-Ready/flares_event_dir"
# non_flares_event_dir = "/mnt/data/ML-Ready/non_flares_event_dir"
# flare_events_csv = "/mnt/data/flare_list/flare_events_2023-07-01_2023-08-15.csv"
#
# train_range = (datetime(2023, 7, 1), datetime(2023, 7, 25))
# val_range = (datetime(2023, 7, 27), datetime(2023, 7, 30))
# test_range = (datetime(2023, 8, 1), datetime(2023, 8, 15))
#
# os.makedirs(flares_event_dir, exist_ok=True)
# os.makedirs(non_flares_event_dir, exist_ok=True)
#
# os.makedirs(os.path.join(flares_event_dir, "train"), exist_ok=True)
# os.makedirs(os.path.join(flares_event_dir, "val"), exist_ok=True)
# os.makedirs(os.path.join(flares_event_dir, "test"), exist_ok=True)
#
# os.makedirs(os.path.join(non_flares_event_dir, "train"), exist_ok=True)
# os.makedirs(os.path.join(non_flares_event_dir, "val"), exist_ok=True)
# os.makedirs(os.path.join(non_flares_event_dir, "test"), exist_ok=True)
#
#
# flare_event = pd.read_csv(flare_events_csv)
# print(f"Found {len(flare_event)} flare events")
# flaring_eve_list = []
# for i, row in flare_event.iterrows():
#     start_time = pd.to_datetime(row['event_starttime'])
#     end_time = pd.to_datetime(row['event_endtime'])
#     flaring_eve_list.append((start_time, end_time))
#
# data_list = os.listdir(data_dir)
# print(f"Found {len(data_list)} files in {data_dir}")
# for file in data_list:
#     try:
#         aia_time = pd.to_datetime(file.split(".")[0])
#     except ValueError:
#         print(f"Skipping file {file}: Invalid timestamp format")
#         continue
#
#     # Check if the file's time falls within any flare event
#     is_flaring = any(start <= aia_time <= end for start, end in flaring_eve_list)
#     if is_flaring:
#         src = os.path.join(data_dir, file)
#         dst = os.path.join(flares_event_dir, file)
#
#         if train_range[0] <= aia_time <= train_range[1]:
#             dst = os.path.join(flares_event_dir, "train")
#             shutil.copy(src, dst)
#         elif val_range[0] <= aia_time <= val_range[1]:
#             dst = os.path.join(flares_event_dir, "val")
#             shutil.copy(src, dst)
#         elif test_range[0] <= aia_time <= test_range[1]:
#             dst = os.path.join(flares_event_dir, "test")
#             shutil.copy(src, dst)
#         else:
#             print(f"Skipping {file}: Time {aia_time} not in any defined range")
#             continue
#         print(f"Copied {file} to {dst}")
#     else:
#         print("Skipping non-flaring event file:", file)
    # else:
    #     src = os.path.join(data_dir, file)
    #     dst = os.path.join(non_flares_event_dir, file)
    #     print(aia_time)
    #     print(train_range[0], train_range[1])
    #     if train_range[0] <= aia_time <= train_range[1]:
    #         split_dir = "train"
    #     elif val_range[0] <= aia_time <= val_range[1]:
    #         split_dir = "val"
    #     elif test_range[0] <= aia_time <= test_range[1]:
    #         split_dir = "test"
    #     dst = os.path.join(flares_event_dir, split_dir)
    #    # shutil.copy(src, dst)
    #     print(f"Copied {file} to {dst}")



        # Move file to appropriate split directory
        src = os.path.join(base_dir, file)
        dst = os.path.join(base_dir, split_dir, file)
        shutil.move(src, dst)
        print(f"Moved {file} to {base_dir}/{split_dir}")
