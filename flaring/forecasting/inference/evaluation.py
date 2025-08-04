import os
import glob
from multiprocessing import Pool
from datetime import timedelta, datetime

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from scipy.ndimage import zoom
from matplotlib.colors import AsinhNorm, LogNorm
import seaborn as sns


class SolarFlareEvaluator:
    def __init__(self,
                 csv_path=None,
                 aia_dir=None,
                 weight_path=None,
                 baseline_csv_path=None,
                 output_dir="./solar_flare_evaluation"):
        """
        Initialize the solar flare evaluation system with baseline comparison.

        Args:
            csv_path (str): Path to main model prediction results CSV
            aia_dir (str): Path to AIA image data directory
            weight_path (str): Path to main model attention weights directory
            baseline_csv_path (str): Path to baseline model prediction results CSV
            output_dir (str): Base output directory for results
        """
        # Set paths
        self.csv_path = csv_path
        self.aia_dir = aia_dir
        self.weight_path = weight_path
        self.baseline_csv_path = baseline_csv_path
        self.output_dir = output_dir

        # Create output directory structure
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.frames_dir = os.path.join(output_dir, "movie_frames")
        self.comparison_dir = os.path.join(output_dir, "baseline_comparison")

        for dir_path in [self.metrics_dir, self.plots_dir, self.frames_dir, self.comparison_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data holders
        self.df = None
        self.baseline_df = None
        self.sxr_df = None
        self.y_true = None
        self.y_pred = None
        self.y_baseline = None

    def load_data(self):
        """Load and prepare all required data including baseline"""
        # Load main model prediction data
        if self.csv_path and os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            self._clean_prediction_data(self.df)
            self.y_true = self.df['groundtruth'].values
            self.y_pred = self.df['predictions'].values
            print(f"Loaded main model data with {len(self.df)} records")

        # Load baseline model prediction data
        if self.baseline_csv_path and os.path.exists(self.baseline_csv_path):
            self.baseline_df = pd.read_csv(self.baseline_csv_path)
            self._clean_prediction_data(self.baseline_df)
            self.y_baseline = self.baseline_df['predictions'].values
            print(f"Loaded baseline model data with {len(self.baseline_df)} records")

            # Ensure same length as main model data
            if len(self.y_baseline) != len(self.y_pred):
                print("Warning: Baseline and main model have different number of predictions")
                min_len = min(len(self.y_baseline), len(self.y_pred))
                self.y_baseline = self.y_baseline[:min_len]
                self.y_pred = self.y_pred[:min_len]
                self.y_true = self.y_true[:min_len]


    def _clean_prediction_data(self, df):
        """Clean and prepare prediction data"""
        for col in ['groundtruth', 'predictions']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

    def calculate_metrics(self):
        """Calculate and save performance metrics for both models"""
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No prediction data available. Load data first.")

        # Calculate metrics for main model
        main_metrics = {
            'Model': 'ViT',
            'MSE': mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'RMSE': np.sqrt(mean_squared_error(np.log10(self.y_true), np.log10(self.y_pred))),
            'MAE': mean_absolute_error(np.log10(self.y_true), np.log10(self.y_pred)),
            'R2': r2_score(np.log10(self.y_true), np.log10(self.y_pred)),
            'Pearson_Corr': np.corrcoef(np.log10(self.y_true), np.log10(self.y_pred))[0, 1],
        }

        # Calculate metrics for each flare class
        flare_classes = {
            'Quiet': (0, 1e-6),  # Below 1e-6
            'C': (1e-6, 1e-5),  # 1e-6 to 1e-5
            'M': (1e-5, 1e-4),  # 1e-5 to 1e-4
            'X': (1e-4, np.inf)  # Above 1e-4
        }

        flare_class_metrics = []

        for class_name, (lower_bound, upper_bound) in flare_classes.items():
            # Create mask for current flare class
            if upper_bound == np.inf:
                mask = self.y_true >= lower_bound
            else:
                mask = (self.y_true >= lower_bound) & (self.y_true < upper_bound)

            # Skip if no samples in this class
            if not np.any(mask):
                print(f"Warning: No samples found for flare class {class_name}")
                continue

            # Get true and predicted values for this class
            y_true_class = self.y_true[mask]
            y_pred_class = self.y_pred[mask]

            # Calculate metrics for this flare class
            class_metrics = {
                'Model': f'ViT_{class_name}',
                'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_pred_class))),
                'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_pred_class)),
                'R2': r2_score(np.log10(y_true_class), np.log10(y_pred_class)),
                'Sample_Count': len(y_true_class),
                'Pearson_Corr': np.corrcoef(np.log10(y_true_class), np.log10(y_pred_class))[0, 1],
            }

            flare_class_metrics.append(class_metrics)

            # If baseline exists, calculate baseline metrics for this class too
            if self.y_baseline is not None:
                y_baseline_class = self.y_baseline[mask]

                baseline_class_metrics = {
                    'Model': f'Baseline_{class_name}',
                    'MSE': mean_squared_error(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'RMSE': np.sqrt(mean_squared_error(np.log10(y_true_class), np.log10(y_baseline_class))),
                    'MAE': mean_absolute_error(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'R2': r2_score(np.log10(y_true_class), np.log10(y_baseline_class)),
                    'Sample_Count': len(y_true_class),
                    'Pearson_Corr': np.corrcoef(np.log10(y_true_class), np.log10(y_baseline_class))[0, 1],
                }

                flare_class_metrics.append(baseline_class_metrics)

                # Calculate improvement for this class
                improvement_class_metrics = {
                    'Model': f'Improvement_{class_name} (%)',
                    'MSE': ((baseline_class_metrics['MSE'] - class_metrics['MSE']) / baseline_class_metrics[
                        'MSE']) * 100,
                    'RMSE': ((baseline_class_metrics['RMSE'] - class_metrics['RMSE']) / baseline_class_metrics[
                        'RMSE']) * 100,
                    'MAE': ((baseline_class_metrics['MAE'] - class_metrics['MAE']) / baseline_class_metrics[
                        'MAE']) * 100,
                    'R2': ((class_metrics['R2'] - baseline_class_metrics['R2']) / abs(
                        baseline_class_metrics['R2'])) * 100,
                    'Sample_Count': len(y_true_class),
                    'Pearson_Corr': (
                        np.corrcoef(np.log10(y_true_class), np.log10(y_baseline_class))[0, 1] -
                        np.corrcoef(np.log10(y_true_class), np.log10(y_pred_class))[0, 1]
                    ) * 100
                }

                flare_class_metrics.append(improvement_class_metrics)

        metrics_list = [main_metrics] + flare_class_metrics

        # Calculate metrics for baseline model if available
        if self.y_baseline is not None:
            baseline_metrics = {
                'Model': 'Baseline',
                'MSE': mean_squared_error(np.log10(self.y_true), np.log10(self.y_baseline)),
                'RMSE': np.sqrt(mean_squared_error(np.log10(self.y_true), np.log10(self.y_baseline))),
                'MAE': mean_absolute_error(np.log10(self.y_true), np.log10(self.y_baseline)),
                'R2': r2_score(np.log10(self.y_true), np.log10(self.y_baseline)),
                'Pearson_Corr': np.corrcoef(np.log10(self.y_true), np.log10(self.y_baseline))[0, 1]
            }
            metrics_list.append(baseline_metrics)

            # Calculate improvement metrics
            improvement_metrics = {
                'Model': 'Improvement_Overall (%)',
                'MSE': ((baseline_metrics['MSE'] - main_metrics['MSE']) / baseline_metrics['MSE']) * 100,
                'RMSE': ((baseline_metrics['RMSE'] - main_metrics['RMSE']) / baseline_metrics['RMSE']) * 100,
                'MAE': ((baseline_metrics['MAE'] - main_metrics['MAE']) / baseline_metrics['MAE']) * 100,
                'R2': ((main_metrics['R2'] - baseline_metrics['R2']) / abs(baseline_metrics['R2'])) * 100,
                'Pearson_Corr': ((baseline_metrics['Pearson_Corr'] - main_metrics['Pearson_Corr']) / abs(baseline_metrics['Pearson_Corr'])) * 100
            }
            metrics_list.append(improvement_metrics)

        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_list)
        metrics_path = os.path.join(self.metrics_dir, "performance_comparison.csv")
        metrics_df.to_csv(metrics_path, index=False)

        # Generate comparison plots
        self._plot_regression_comparison()
        if self.y_baseline is not None:
            self._plot_error_comparison()
            self._plot_metrics_comparison(metrics_list)

        return metrics_df

    def _calculate_tss(self, y_true, y_pred, threshold=None):
        """Calculate True Skill Statistic"""
        if threshold is None:
            threshold = np.median(y_true)

        y_true_bin = (y_true > threshold).astype(int)
        y_pred_bin = (y_pred > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity + specificity - 1

    def _plot_regression_comparison(self):
        """Generate regression comparison plot"""

        log_bins = np.logspace(np.log10(min(self.y_true)),
                               np.log10(max(self.y_true)), 100)
        shared_norm = LogNorm(vmin=1, vmax=None)  # Or specify vmax explicitly

        if self.y_baseline is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None

        fig.patch.set_facecolor('#1a1a2e')
        sns.set_palette("husl")
        ax1.set_facecolor('#2d2d44')
        if ax2 is not None:
            ax2.set_facecolor('#2d2d44')

        # Main model plot
        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val],
                 label='Perfect Prediction', color='red', linestyle='--', linewidth=2)

        h1 = ax1.hist2d(self.y_true, self.y_pred, bins=[log_bins, log_bins],
                        cmap='summer', norm=shared_norm)

        ax1.set_xlabel('Ground Truth Flux', color='white')
        ax1.set_ylabel('Predicted Flux', color='white')
        ax1.set_title('ViT Model Performance', color='white')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        for spine in ax1.spines.values():
            spine.set_color('white')

        # Baseline model plot if available
        if self.y_baseline is not None and ax2 is not None:
            h2 = ax2.hist2d(self.y_true, self.y_baseline, bins=[log_bins, log_bins],
                            cmap='summer', norm=shared_norm)

            min_val = min(min(self.y_true), min(self.y_baseline))
            max_val = max(max(self.y_true), max(self.y_baseline))
            ax2.plot([min_val, max_val], [min_val, max_val],
                     label='Perfect Prediction', color='red', linestyle='--', linewidth=2)

            ax2.set_xlabel('Ground Truth Flux', color='white')
            ax2.set_ylabel('Predicted Flux', color='white')
            ax2.set_title('Baseline Model Performance', color='white')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            for spine in ax2.spines.values():
                spine.set_color('white')

        # Colorbar (use h1[3] or h2[3], they’re identical)
        cbar = fig.colorbar(h1[3], ax=[ax1, ax2] if ax2 else ax1, orientation='vertical', pad=.02)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        cbar.set_label("Count (log scale)", color='white')

        #plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "regression_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved regression comparison plot to {plot_path}")

    def _plot_error_comparison(self):
        """Generate error comparison plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals comparison
        main_residuals = self.y_pred - self.y_true
        baseline_residuals = self.y_baseline - self.y_true

        ax1.scatter(self.y_true, main_residuals, alpha=0.5, label='Main Model', color='blue')
        ax1.scatter(self.y_true, baseline_residuals, alpha=0.5, label='Baseline Model', color='red')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Ground Truth')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Error histograms
        ax2.hist(np.abs(main_residuals), bins=30, alpha=0.7, label='Main Model', color='blue')
        ax2.hist(np.abs(baseline_residuals), bins=30, alpha=0.7, label='Baseline Model', color='red')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Prediction scatter comparison
        ax3.scatter(self.y_baseline, self.y_pred, alpha=0.5, color='purple')
        min_val = min(min(self.y_baseline), min(self.y_pred))
        max_val = max(max(self.y_baseline), max(self.y_pred))
        ax3.plot([min_val, max_val], [min_val, max_val], '--k', alpha=0.7)
        ax3.set_xlabel('Baseline Predictions')
        ax3.set_ylabel('Main Model Predictions')
        ax3.set_title('Model Predictions Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        # Error reduction plot
        error_improvement = np.abs(baseline_residuals) - np.abs(main_residuals)
        ax4.hist(error_improvement, bins=30, alpha=0.7, color='green')
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Error Reduction (Baseline - Main)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Error Improvement Distribution')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')

        plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "error_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved error comparison plot to {plot_path}")

    def _plot_metrics_comparison(self, metrics_list):
        """Generate metrics comparison bar plot"""
        main_metrics = metrics_list[0]
        baseline_metrics = metrics_list[1]

        metrics_names = ['RMSE', 'MSE', 'MAE', 'R2']
        main_values = [main_metrics[m] for m in metrics_names]
        baseline_values = [baseline_metrics[m] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, main_values, width, label='Main Model', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width / 2, baseline_values, width, label='Baseline Model', color='red', alpha=0.7)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        add_value_labels(bars1)
        add_value_labels(bars2)

        plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "metrics_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics comparison plot to {plot_path}")

    @staticmethod
    def init_worker(csv_data, baseline_csv_data):
        """Initialize each worker process with CSV data"""
        global csv_data_global, baseline_csv_data_global
        csv_data_global = csv_data
        baseline_csv_data_global = baseline_csv_data
        print(f"Worker {os.getpid()}: CSV data loaded")

    def load_csv_data(self):
        """Load and prepare CSV data for workers"""
        # Load main model CSV
        csv_data = pd.read_csv(self.csv_path)
        if 'timestamp' in csv_data.columns:
            csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'])

        # Load baseline CSV
        baseline_data = pd.read_csv(self.baseline_csv_path)
        if 'timestamp' in baseline_data.columns:
            baseline_data['timestamp'] = pd.to_datetime(baseline_data['timestamp'])

        return csv_data, baseline_data


    def load_aia_image(self, timestamp):
        """Load AIA image for given timestamp"""
        pattern = f"{self.aia_dir}/*{timestamp}*"
        files = glob.glob(pattern)
        if files:
            return np.load(files[0])
        return None

    def load_attention_map(self, timestamp):
        """Load attention map for given timestamp"""
        filepath = os.path.join(self.weight_path, f"{timestamp}")
        try:
            attention = np.loadtxt(filepath, delimiter=",")
            target_shape = [512, 512]
            zoom_factors = (target_shape[0] / attention.shape[0],
                            target_shape[1] / attention.shape[1])
            return zoom(attention, zoom_factors, order=1)
        except Exception as e:
            print(f"Could not load attention map for {timestamp}: {e}")
            return None

    def get_sxr_data_for_timestamp(self, timestamp, window_hours=12):
        """Get SXR data around the given timestamp from CSV files"""
        try:
            # Access global CSV data loaded in worker
            global csv_data_global, baseline_csv_data_global

            target_time = pd.to_datetime(timestamp)

            # Find matching row in main model CSV
            main_row = csv_data_global[csv_data_global['timestamp'] == target_time]
            baseline_row = baseline_csv_data_global[baseline_csv_data_global['timestamp'] == target_time]

            if main_row.empty:
                print(f"No main model data found for timestamp {timestamp}")
                return None, None, None

            if baseline_row.empty:
                print(f"No baseline data found for timestamp {timestamp}")
                baseline_pred = None
            else:
                baseline_pred = baseline_row.iloc[0]['predictions']

            # Extract data using correct column names
            current_data = {
                'groundtruth': main_row.iloc[0]['groundtruth'],
                'predictions': main_row.iloc[0]['predictions'],
                'baseline_predictions': baseline_row.iloc[0]['predictions'] if not baseline_row.empty else None,
                'timestamp': target_time
            }

            # Create window data (get surrounding timestamps within window_hours)
            time_window_start = target_time - pd.Timedelta(hours=window_hours / 2)
            time_window_end = target_time + pd.Timedelta(hours=window_hours / 2)

            # Filter data within window
            main_window = csv_data_global[
                (csv_data_global['timestamp'] >= time_window_start) &
                (csv_data_global['timestamp'] <= time_window_end)
                ].copy()

            baseline_window = baseline_csv_data_global[
                (baseline_csv_data_global['timestamp'] >= time_window_start) &
                (baseline_csv_data_global['timestamp'] <= time_window_end)
                ].copy()

            # Merge the windows for plotting
            window_data = main_window[['timestamp', 'groundtruth', 'predictions']].copy()

            if not baseline_window.empty:
                # Merge baseline predictions
                baseline_pred_col = baseline_window[['timestamp', 'predictions']].rename(
                    columns={'predictions': 'baseline_predictions'})
                window_data = window_data.merge(baseline_pred_col, on='timestamp', how='left')
            else:
                window_data['baseline_predictions'] = None

            return window_data, current_data, target_time

        except Exception as e:
            print(f"Could not get SXR data for timestamp {timestamp}: {e}")
            return None, None, None

    def generate_frame_worker(self, timestamp):
        """Worker function to generate a single frame"""
        try:
            print(f"Worker {os.getpid()}: Processing {timestamp}")

            # Load data
            aia_data = self.load_aia_image(timestamp)
            attention_data = self.load_attention_map(timestamp)

            if aia_data is None or attention_data is None:
                print(f"Worker {os.getpid()}: Skipping {timestamp} (missing data)")
                return None

            # Get SXR data from CSV
            sxr_window, sxr_current, target_time = self.get_sxr_data_for_timestamp(timestamp)

            # Generate frame
            save_path = os.path.join(self.frames_dir, f"{timestamp}.png")

            # Create figure
            fig = plt.figure(figsize=(22, 8))  # Made wider to accommodate three lines
            fig.patch.set_facecolor('#1a1a2e')
            gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.8], hspace=0.2, wspace=0.2)

            wavs = ['94', '131', '171', '193', '211', '304']
            att_max = np.percentile(attention_data, 100)
            att_min = np.percentile(attention_data, 0)
            att_norm = AsinhNorm(vmin=att_min, vmax=att_max, clip=False)

            # Plot AIA images with attention maps
            for i in range(6):
                row = i // 3
                col = i % 3
                ax = fig.add_subplot(gs[row, col])

                aia_img = aia_data[i]
                ax.imshow(aia_img, cmap="gray", origin='lower')
                ax.imshow(attention_data, cmap='hot', origin='lower', alpha=0.35, norm=att_norm)
                ax.set_title(f'AIA {wavs[i]} Å', fontsize=10, color='white')
                ax.axis('off')

            # Plot SXR data
            sxr_ax = fig.add_subplot(gs[:, 3])
            sxr_ax.set_facecolor('#2a2a3e')

            if sxr_window is not None and not sxr_window.empty:
                # Plot ground truth
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['groundtruth'], label='Ground Truth', linewidth=3, alpha=1, markersize=5, color = "#F78E69")

                # Plot new model predictions
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['predictions'], label='ViT Model', linewidth=3, alpha=1, markersize=5, color = "#C0B9DD")

                # Plot baseline predictions if available
                if 'baseline_predictions' in sxr_window.columns and sxr_window['baseline_predictions'].notna().any():
                    sxr_ax.plot(sxr_window['timestamp'], sxr_window['baseline_predictions'], label='Baseline Model', linewidth=3, alpha=1, markersize=5, color = "#94ECBE")

                # Mark current time
                if sxr_current is not None:
                    sxr_ax.axvline(target_time, color='white', linestyle='--',
                                   linewidth=2, alpha=0.8, label='Current Time')

                    # Create info text with all available values
                    info_lines = [
                        "Current Values:",
                        f"GT: {sxr_current['groundtruth']:.2e}",
                        f"ViT: {sxr_current['predictions']:.2e}"
                    ]
                    if sxr_current['baseline_predictions'] is not None:
                        info_lines.append(f"Base: {sxr_current['baseline_predictions']:.2e}")

                    info_text = "\n".join(info_lines)
                    sxr_ax.text(0.02, 0.98, info_text, transform=sxr_ax.transAxes,
                                fontsize=8, color='white', verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

                sxr_ax.set_ylabel('SXR Flux', fontsize=10, color='white')
                sxr_ax.set_xlabel('Time', fontsize=10, color='white')
                sxr_ax.set_title('SXR Data Comparison', fontsize=12, color='white')
                sxr_ax.legend(fontsize=8, loc='upper right')
                legend1 = sxr_ax.legend()
                legend1.get_frame().set_facecolor('black')
                legend1.get_frame().set_edgecolor('white')
                for text in legend1.get_texts():
                    text.set_color('white')
                sxr_ax.grid(True, alpha=0.3)
                sxr_ax.tick_params(axis='x', rotation=45, labelsize=8, colors='white')
                sxr_ax.tick_params(axis='y', labelsize=8, colors='white')
                for spine in sxr_ax.spines.values():
                    spine.set_color('white')
                try:
                    sxr_ax.set_yscale('log')
                except:
                    pass  # Skip log scale if data doesn't support it
            else:
                sxr_ax.text(0.5, 0.5, 'No SXR Data\nAvailable',
                            transform=sxr_ax.transAxes, fontsize=12, color='white',
                            horizontalalignment='center', verticalalignment='center')
                sxr_ax.set_title('SXR Data Comparison', fontsize=12, color='white')

            for spine in sxr_ax.spines.values():
                spine.set_color('white')

            plt.suptitle(f'Timestamp: {timestamp}', color='white', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, facecolor='#1a1a2e')
            plt.close()

            print(f"Worker {os.getpid()}: Completed {timestamp}")
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')  # Clean up any open figures
            return None

    def create_attention_movie(self, timestamps):
        """Generate attention visualization movie with baseline comparison"""
        print(f"Generated {len(timestamps)} timestamps to process")

        # Load CSV data once
        print("Loading CSV data...")
        csv_data, baseline_csv_data = self.load_csv_data()

        # Determine number of processes
        num_processes = min(os.cpu_count(), len(timestamps))  # Don't use more processes than timestamps
        num_processes = max(1, num_processes - 1)  # Leave one CPU free
        print(f"Using {num_processes} processes")

        # Process frames in parallel
        import time
        start_time = time.time()

        with Pool(processes=num_processes, initializer=self.init_worker,
                  initargs=(csv_data, baseline_csv_data)) as pool:
            # Use map to process all timestamps
            results = pool.map(self.generate_frame_worker, timestamps)

        # Filter out failed frames
        frame_paths = [path for path in results if path is not None]

        processing_time = time.time() - start_time
        print(f"Generated {len(frame_paths)} frames in {processing_time:.2f} seconds")
        if len(frame_paths) > 0:
            print(f"Average: {processing_time / len(frame_paths):.2f} seconds per frame")

        if not frame_paths:
            print("No frames were successfully generated. Cannot create movie.")
            return

        # Compile into video
        print("Creating movie...")
        video_start = time.time()

        # Sort frame paths by timestamp to ensure correct order
        frame_paths.sort(key=lambda x: os.path.basename(x))

        movie_path = os.path.join(self.output_dir, "AIA_video.mp4")
        with imageio.get_writer(movie_path, fps=30) as writer:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    image = imageio.imread(frame_path)
                    writer.append_data(image)

        video_time = time.time() - video_start
        total_time = time.time() - start_time

        print(f"Video creation took {video_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"✅ Movie saved to: {movie_path}")

        # Optional: Clean up frame files
        cleanup = input("Delete individual frame files? (y/n): ").lower().strip()
        if cleanup == 'y':
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            print("Frame files deleted")

    def run_full_evaluation(self, timestamps=None):
        """Run complete evaluation pipeline with baseline comparison"""
        print("=== Solar Flare Evaluation with Baseline Comparison ===")
        print(f"Output will be saved to: {self.output_dir}")

        # Load all data
        print("\nLoading data...")
        self.load_data()

        # Quantitative evaluation
        print("\nCalculating performance metrics...")
        metrics_df = self.calculate_metrics()

        print("\n=== Performance Metrics Comparison ===")
        print(metrics_df.to_string(index=False))

        # Visual evaluation if timestamps provided
        if timestamps:
            print("\nGenerating attention visualizations...")
            self.create_attention_movie(timestamps)

        print("\nEvaluation complete!")
        return metrics_df


# Example Usage
if __name__ == "__main__":
    # Example paths - replace with your actual paths
    example_csv = "/mnt/data/ML-Ready_clean/mixed_data/output/final-vit-results.csv"
    example_baseline_csv = "/home/griffingoodwin/2025-HL-Flaring-MEGS-AI/final-model-results.csv"
    example_aia = "/mnt/data/ML-Ready_clean/mixed_data/AIA/test/"
    example_weights = "/mnt/data/ML-Ready_clean/mixed_data/weights-finalepoch/"

    # Sample timestamps - Fixed the datetime generation
    start_time = datetime(2023, 8, 5)
    end_time = datetime(2023, 8, 10)
    interval = timedelta(minutes=1)  # Changed from minutes=60 to hours=1 for clarity
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time.strftime("%Y-%m-%dT%H:%M:%S"))
        current_time += interval

    # Initialize evaluator with baseline comparison
    evaluator = SolarFlareEvaluator(
        csv_path=example_csv,
        baseline_csv_path=example_baseline_csv,
        aia_dir=example_aia,
        weight_path=example_weights,
        output_dir="./solar_flare_comparison_results"
    )

    # Run complete evaluation with baseline comparison
    evaluator.run_full_evaluation(timestamps=timestamps)