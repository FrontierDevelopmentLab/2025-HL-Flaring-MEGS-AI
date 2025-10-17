import os
import pandas as pd
import shutil
import argparse
from datetime import datetime
from pathlib import Path

def split_data(input_folder, output_dir, data_type, flare_events_csv=None, repartition=False, 
               train_start=None, train_end=None, val_start=None, val_end=None, 
               test_start=None, test_end=None, use_buffer_strategy=False, copy_files=False):
    """
    Split data from input folder into train/val/test based on custom date ranges or month.
    Optionally use flare events for additional classification.
    
    Args:
        input_folder (str): Path to the input folder containing data files
        output_dir (str): Path to the output directory where split data will be saved
        data_type (str): Type of data ('aia' or 'sxr')
        flare_events_csv (str, optional): Path to the flare events CSV file
        repartition (bool): If True, treat input_folder as already partitioned (has train/val/test subdirs)
        train_start (str, optional): Start date for training data (format: 'YYYY-MM-DD')
        train_end (str, optional): End date for training data (format: 'YYYY-MM-DD')
        val_start (str, optional): Start date for validation data (format: 'YYYY-MM-DD')
        val_end (str, optional): End date for validation data (format: 'YYYY-MM-DD')
        test_start (str, optional): Start date for test data (format: 'YYYY-MM-DD')
        test_end (str, optional): End date for test data (format: 'YYYY-MM-DD')
        use_buffer_strategy (bool): If True, use predefined buffer strategy with specific date ranges
        copy_files (bool): If True, copy files instead of moving them (default: False)
    """
    
    # Validate input folder
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Validate data type
    if data_type.lower() not in ['aia', 'sxr']:
        raise ValueError("data_type must be 'aia' or 'sxr'")
    
    # Parse and validate date ranges
    date_ranges = {}
    custom_dates = False
    buffer_strategy_ranges = None
    
    if use_buffer_strategy:
        print("Using predefined buffer strategy...")
        
        # Define the buffer strategy date ranges (multiple periods per split)
        buffer_strategy_ranges = {
            'train': [
                (pd.to_datetime("2012-01-01").replace(hour=0, minute=0, second=0, microsecond=0),
                 pd.to_datetime("2023-02-27").replace(hour=23, minute=59, second=59, microsecond=999999)),
                (pd.to_datetime("2023-07-01").replace(hour=0, minute=0, second=0, microsecond=0),
                 pd.to_datetime("2023-07-20").replace(hour=23, minute=59, second=59, microsecond=999999))
            ],
            'val': [
                (pd.to_datetime("2023-03-01").replace(hour=0, minute=0, second=0, microsecond=0),
                 pd.to_datetime("2023-06-29").replace(hour=23, minute=59, second=59, microsecond=999999)),
                (pd.to_datetime("2023-07-22").replace(hour=0, minute=0, second=0, microsecond=0),
                 pd.to_datetime("2023-07-30").replace(hour=23, minute=59, second=59, microsecond=999999))
            ],
            'test': [
                (pd.to_datetime("2023-08-01").replace(hour=0, minute=0, second=0, microsecond=0),
                 pd.to_datetime("2025-09-27").replace(hour=23, minute=59, second=59, microsecond=999999))
            ]
        }
        #print the buffer strategy date ranges
        print(buffer_strategy_ranges)
    elif any([train_start, train_end, val_start, val_end, test_start, test_end]):
        custom_dates = True
        print("Using custom date ranges for data splitting...")
        
        # Parse date strings to datetime objects
        try:
            if train_start and train_end:
                # Start date at beginning of day, end date at end of day
                train_start_dt = pd.to_datetime(train_start).replace(hour=0, minute=0, second=0, microsecond=0)
                train_end_dt = pd.to_datetime(train_end).replace(hour=23, minute=59, second=59, microsecond=999999)
                date_ranges['train'] = (train_start_dt, train_end_dt)
                print(f"Training data: {train_start} 00:00:00 to {train_end} 23:59:59")
            if val_start and val_end:
                val_start_dt = pd.to_datetime(val_start).replace(hour=0, minute=0, second=0, microsecond=0)
                val_end_dt = pd.to_datetime(val_end).replace(hour=23, minute=59, second=59, microsecond=999999)
                date_ranges['val'] = (val_start_dt, val_end_dt)
                print(f"Validation data: {val_start} 00:00:00 to {val_end} 23:59:59")
            if test_start and test_end:
                test_start_dt = pd.to_datetime(test_start).replace(hour=0, minute=0, second=0, microsecond=0)
                test_end_dt = pd.to_datetime(test_end).replace(hour=23, minute=59, second=59, microsecond=999999)
                date_ranges['test'] = (test_start_dt, test_end_dt)
                print(f"Test data: {test_start} 00:00:00 to {test_end} 23:59:59")
        except Exception as e:
            raise ValueError(f"Invalid date format. Please use 'YYYY-MM-DD' format. Error: {e}")
        
        # Validate that date ranges don't overlap
        ranges = list(date_ranges.values())
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                range1, range2 = ranges[i], ranges[j]
                if not (range1[1] < range2[0] or range2[1] < range1[0]):
                    raise ValueError(f"Date ranges cannot overlap. Found overlap between ranges: {range1[0].strftime('%Y-%m-%d')} to {range1[1].strftime('%Y-%m-%d')} and {range2[0].strftime('%Y-%m-%d')} to {range2[1].strftime('%Y-%m-%d')}")
    else:
        print("Using default month-based splitting...")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    print(f"Processing {data_type.upper()} data from: {input_folder}")
    print(f"Output directory: {output_dir}")
    
    # Get list of files in input folder
    if repartition:
        # For repartitioning, collect files from train/val/test subdirectories
        data_list = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(input_folder, split)
            if os.path.exists(split_dir):
                split_files = os.listdir(split_dir)
                # Add split information to each file for tracking
                for file in split_files:
                    data_list.append((file, split))
                print(f"Found {len(split_files)} files in {split}/ directory")
        print(f"Total files to repartition: {len(data_list)}")
    else:
        # For normal splitting, get files directly from input folder
        data_list = os.listdir(input_folder)
        print(f"Found {len(data_list)} files to process")
    
    moved_count = 0
    skipped_count = 0

    for file_info in data_list:
        if repartition:
            file, original_split = file_info
        else:
            file = file_info
            original_split = None
            
        try:
            # Extract timestamp from filename (assuming format like "2012-01-01T00:00:00.npy")
            file_time = pd.to_datetime(file.split(".")[0])
        except ValueError:
            print(f"Skipping file {file}: Invalid timestamp format")
            skipped_count += 1
            continue
        
        # Determine split based on custom date ranges or month
        new_split_dir = None
        
        if buffer_strategy_ranges:
            # Use buffer strategy with multiple periods per split
            for split_name, periods in buffer_strategy_ranges.items():
                for start_date, end_date in periods:
                    if start_date <= file_time <= end_date:
                        new_split_dir = split_name
                        break
                if new_split_dir:
                    break
        elif custom_dates:
            for split_name, (start_date, end_date) in date_ranges.items():
                if start_date <= file_time <= end_date:
                    new_split_dir = split_name
                    break
        
        print(f"File {file} assigned to split {new_split_dir}")

        # Check if file was assigned to a split
        if new_split_dir is None:
            if custom_dates or buffer_strategy_ranges:
                print(f"Skipping file {file}: No matching date range (file time: {file_time.strftime('%Y-%m-%d')})")
            else:
                # Use default month-based splitting
                month = file_time.month
                
                if month in [4, 5, 6, 7, 9, 10, 11, 12]:
                    new_split_dir = "train"
                elif month in [1,2,3]:
                    new_split_dir = "val"
                elif month == 8:
                    new_split_dir = "test"
                else:
                    print(f"Skipping file {file}: Unexpected month {month}")
                    skipped_count += 1
                    continue
            
            # If still no split assigned, skip the file
            if new_split_dir is None:
                skipped_count += 1
                continue
        
        # Determine source and destination paths
        if repartition:
            src_path = os.path.join(input_folder, original_split, file)
        else:
            src_path = os.path.join(input_folder, file)
        
        dst_path = os.path.join(output_dir, new_split_dir, file)
        
        # Skip if file is already in the correct split and we're repartitioning
        if repartition and original_split == new_split_dir and os.path.exists(dst_path):
            print(f"File {file} already in correct split ({new_split_dir}), skipping.")
            skipped_count += 1
            continue
        
        if not os.path.exists(dst_path):
            try:
                if copy_files:
                    shutil.copy2(src_path, dst_path)
                    action = "Copied"
                else:
                    shutil.move(src_path, dst_path)
                    action = "Moved"
                moved_count += 1
            except Exception as e:
                print(f"Error {action.lower()}ing {file}: {e}")
                skipped_count += 1
        else:
            action = "copy" if copy_files else "move"
            print(f"File {dst_path} already exists, skipping {action}.")
            skipped_count += 1
    
    action = "copied" if copy_files else "moved"
    print(f"\nProcessing complete!")
    print(f"Files {action}: {moved_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Total files processed: {moved_count + skipped_count}")
    
    # Check for overlapping files between splits
    print(f"\nChecking for overlapping files between splits...")
    overlap_found = False
    
    # Get all files in each split directory
    split_files = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            split_files[split] = set(os.listdir(split_dir))
        else:
            split_files[split] = set()
    
    # Check for overlaps between each pair of splits
    splits = list(split_files.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            split1, split2 = splits[i], splits[j]
            overlap = split_files[split1] & split_files[split2]
            
            if overlap:
                overlap_found = True
                print(f"WARNING: Found {len(overlap)} overlapping files between {split1} and {split2}:")
                for file in sorted(overlap):
                    print(f"  - {file}")
            else:
                print(f"✓ No overlap between {split1} and {split2}")
    
    if not overlap_found:
        print("✓ No overlapping files found between any splits - data integrity verified!")
    else:
        print(f"\n⚠️  WARNING: Overlapping files detected! Please review the splitting logic.")
    
    # Summary of files per split
    print(f"\nFinal split summary:")
    for split in ["train", "val", "test"]:
        file_count = len(split_files[split])
        print(f"  {split}: {file_count} files")

def main():
    parser = argparse.ArgumentParser(description='Split AIA or SXR data into train/val/test sets based on custom date ranges or month')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to the input folder containing data files (or partitioned folder for repartition)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to the output directory where split data will be saved')
    parser.add_argument('--data_type', type=str, choices=['aia', 'sxr'], required=True,
                       help='Type of data: "aia" or "sxr"')
    parser.add_argument('--flare_events_csv', type=str, default=None,
                       help='Path to the flare events CSV file (optional)')
    parser.add_argument('--repartition', action='store_true',
                       help='Repartition an already partitioned folder (input_folder should have train/val/test subdirs)')
    
    # Custom date range arguments
    parser.add_argument('--train_start', type=str, default=None,
                       help='Start date for training data (format: YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default=None,
                       help='End date for training data (format: YYYY-MM-DD)')
    parser.add_argument('--val_start', type=str, default=None,
                       help='Start date for validation data (format: YYYY-MM-DD)')
    parser.add_argument('--val_end', type=str, default=None,
                       help='End date for validation data (format: YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default=None,
                       help='Start date for test data (format: YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default=None,
                       help='End date for test data (format: YYYY-MM-DD)')
    parser.add_argument('--use_buffer_strategy', action='store_true',
                       help='Use predefined buffer strategy with specific date ranges and buffer zones')
    parser.add_argument('--copy_files', action='store_true',
                       help='Copy files instead of moving them (keeps original files intact)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_folder = os.path.abspath(args.input_folder)
    output_dir = os.path.abspath(args.output_dir)
    flare_events_csv = os.path.abspath(args.flare_events_csv) if args.flare_events_csv else None
    
    # Validate repartition mode
    if args.repartition:
        # Check if input folder has train/val/test subdirectories
        expected_dirs = ['train', 'val', 'test']
        missing_dirs = []
        for dir_name in expected_dirs:
            if not os.path.exists(os.path.join(input_folder, dir_name)):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"Warning: Input folder is missing expected subdirectories: {missing_dirs}")
            print("Continuing with available directories...")
    
    split_data(input_folder, output_dir, args.data_type, flare_events_csv, args.repartition,
               args.train_start, args.train_end, args.val_start, args.val_end, 
               args.test_start, args.test_end, args.use_buffer_strategy, args.copy_files)

if __name__ == "__main__":
    main()
