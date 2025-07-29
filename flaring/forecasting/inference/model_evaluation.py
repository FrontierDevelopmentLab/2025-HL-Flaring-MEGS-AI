import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import torch
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix,r2_score,accuracy_score

def convert_tensor_strings(value):
    """Convert string representations of tensors to floats"""
    if isinstance(value, str) and value.startswith('tensor('):
        # Extract numeric value from string like "tensor([-5.2153])"
        match = re.search(r'[-+]?\d*\.\d+|\d+', value)
        if match:
            return float(match.group())
    return value

def load_and_convert_data(csv_path):
    """Load CSV and convert tensor strings to floats"""
    df = pd.read_csv(csv_path)

    # Convert tensor strings to numeric values
    for col in ['Ground_Truth', 'Prediction']:
        if col in df.columns:
            df[col] = df[col].apply(convert_tensor_strings)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    return df
def calculate_tss(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    tss = sensitivity + specificity - 1
    return  tss

def calculate_metrics(csv_file_path,output_dir):
    data = load_and_convert_data(csv_file_path)
    y_pred = (data['Prediction']).values
    y_true = (data['Ground_Truth']).values


    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    mae = mean_absolute_error(y_true , y_pred)
    r_squared = r2_score(y_true, y_pred)
   # acc_score = accuracy_score(y_true,y_pred)
    #TSS = calculate_tss()

    plot_regression(y_true , y_pred ,output_dir )
    #plot_tss_thresholds(y_true, y_pred , output_dir)

    print("RMSE :", rmse)
    print("MAE :", mae)
    print("R2 Score :",r_squared )
   # print("Accuracy :", acc_score )

def plot_regression(y_true, y_pred, output_dir):
    """Create regression visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')

    # Residual plot
    residuals = y_pred - y_true
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_plots3.png'))
    plt.close()

"""
def plot_tss_thresholds(y_true, y_pred, output_dir):
    Plot TSS across different percentile thresholds
    thresholds = np.arange(80, 99, 2)
    tss_values = [calculate_tss(y_true, y_pred, t) for t in thresholds]

    plt.figure(figsize=(8, 5))
    plt.plot( tss_values, marker='o')
    plt.xlabel('Percentile Threshold')
    plt.ylabel('TSS Score')
    plt.title('TSS Across Different Thresholds')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'tss_thresholds.png'))
    plt.close()
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                    help='Path to csv')

    parser.add_argument('--output_dir' , type = str, required=True, help='path to save plots')

    args = parser.parse_args()

    result = calculate_metrics(args.csv_path,args.output_dir)



