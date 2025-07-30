import os
import glob
import yaml
from multiprocessing import Pool
from datetime import timedelta, datetime
import argparse
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from scipy.ndimage import zoom
from matplotlib.colors import AsinhNorm


class SolarFlareEvaluator:
    def __init__(self, config_path="eval_config.yaml"):
        """
        Initialize the solar flare evaluation system with configuration from YAML file.

        Args:
            config_path (str): Path to configuration YAML file
        """
        # Load and validate configuration
        self.config = self._load_and_validate_config(config_path)

        # Set core paths from config
        self.csv_path = self.config['prediction_csv']['full_csv_path']
        self.aia_dir = self.config['data']['aia']['test_dir']
        self.weight_path = self.config['data']['weights']['dir']
        self.baseline_csv_path = self.config['baseline']['csv_path']
        self.output_dir = self.config['output']['evaluation_results']

        # Visualization settings
        self.viz_settings = self.config['visualization']
        self.metrics_config = self.config['metrics']

        # Create directory structure
        self._create_output_dirs()

        # Initialize data holders
        self.df = None
        self.baseline_df = None
        self.y_true = None
        self.y_pred = None
        self.y_baseline = None

    def _load_and_validate_config(self, config_path):
        """Load and validate configuration file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Required paths check
        required_paths = [
            'prediction_csv.full_csv_path',
            'data.aia.test_dir',
            'data.weights.dir',
            'output.evaluation_results'
        ]

        for path in required_paths:
            keys = path.split('.')
            current = config
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config key: {path}")
                current = current[key]

        # Expand environment variables in paths
        return self._expand_config_paths(config)

    def _expand_config_paths(self, config):
        """Recursively expand paths in config"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self._expand_config_paths(value)
                elif isinstance(value, str):
                    config[key] = os.path.expandvars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, (dict, list)):
                    self._expand_config_paths(item)
                elif isinstance(item, str):
                    config[i] = os.path.expandvars(item)
        return config

    def _create_output_dirs(self):
        """Create all required output directories"""
        dirs = {
            'metrics_dir': os.path.join(self.output_dir, "metrics"),
            'plots_dir': os.path.join(self.output_dir, "plots"),
            'frames_dir': os.path.join(self.output_dir, "movie_frames"),
            'comparison_dir': os.path.join(self.output_dir, "baseline_comparison"),
            'movies_dir': os.path.join(self.output_dir, "movies")
        }

        # Create instance attributes
        for name, path in dirs.items():
            os.makedirs(path, exist_ok=True)
            setattr(self, name, path)

    def load_data(self):
        """Load and prepare all required data including baseline"""
        # Load main model prediction data
        if os.path.exists(self.csv_path):
            self.df = pd.read_csv(self.csv_path)
            self._clean_prediction_data(self.df)
            self.y_true = self.df['groundtruth'].values
            self.y_pred = self.df['predictions'].values
            print(f"Loaded main model data with {len(self.df)} records")

        # Load baseline model prediction data if configured
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
            'Model': 'Main',
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'R2': r2_score(self.y_true, self.y_pred),
            'TSS': self._calculate_tss(self.y_true, self.y_pred,
                                       threshold=self.metrics_config['tss_threshold'])
        }

        metrics_list = [main_metrics]

        # Calculate metrics for baseline model if available
        if self.y_baseline is not None:
            baseline_metrics = {
                'Model': 'Baseline',
                'MSE': mean_squared_error(self.y_true, self.y_baseline),
                'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_baseline)),
                'MAE': mean_absolute_error(self.y_true, self.y_baseline),
                'R2': r2_score(self.y_true, self.y_baseline),
                'TSS': self._calculate_tss(self.y_true, self.y_baseline,
                                           threshold=self.metrics_config['tss_threshold'])
            }
            metrics_list.append(baseline_metrics)

            # Calculate improvement metrics
            improvement_metrics = {
                'Model': 'Improvement (%)',
                'MSE': ((baseline_metrics['MSE'] - main_metrics['MSE']) / baseline_metrics['MSE']) * 100,
                'RMSE': ((baseline_metrics['RMSE'] - main_metrics['RMSE']) / baseline_metrics['RMSE']) * 100,
                'MAE': ((baseline_metrics['MAE'] - main_metrics['MAE']) / baseline_metrics['MAE']) * 100,
                'R2': ((main_metrics['R2'] - baseline_metrics['R2']) / abs(baseline_metrics['R2'])) * 100,
                'TSS': ((main_metrics['TSS'] - baseline_metrics['TSS']) / abs(baseline_metrics['TSS'])) * 100
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
            threshold = self.metrics_config['tss_threshold']

        y_true_bin = (y_true > threshold).astype(int)
        y_pred_bin = (y_pred > threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity + specificity - 1

    def _plot_regression_comparison(self):
        """Generate regression comparison plot using config settings"""
        colors = self.viz_settings['colors']
        styles = self.viz_settings['styles']
        font_sizes = self.viz_settings['plot_settings']['font_sizes']

        if self.y_baseline is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None

        # Main model plot
        ax1.scatter(self.y_true, self.y_pred, alpha=styles['alpha'],
                    label='Main Model', color=colors['main_model'])
        ax1.plot([min(self.y_true), max(self.y_true)],
                 [min(self.y_true), max(self.y_true)],
                 '--k', label='Perfect Prediction', alpha=styles['alpha'])

        # Add regression line
        coeffs = np.polyfit(self.y_true, self.y_pred, 1)
        regression_line = np.poly1d(coeffs)
        ax1.plot(self.y_true, regression_line(self.y_true),
                 '-', color=colors['main_model'],
                 label='Regression Line', alpha=styles['alpha'])

        ax1.set_xlabel('Ground Truth Flux', fontsize=font_sizes['axis'])
        ax1.set_ylabel('Predicted Flux', fontsize=font_sizes['axis'])
        ax1.set_title('Main Model Performance', fontsize=font_sizes['title'])
        ax1.legend(fontsize=font_sizes['legend'])
        ax1.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Baseline model plot if available
        if self.y_baseline is not None and ax2 is not None:
            ax2.scatter(self.y_true, self.y_baseline, alpha=styles['alpha'],
                        label='Baseline Model', color=colors['baseline_model'])
            ax2.plot([min(self.y_true), max(self.y_true)],
                     [min(self.y_true), max(self.y_true)],
                     '--k', label='Perfect Prediction', alpha=styles['alpha'])

            coeffs_baseline = np.polyfit(self.y_true, self.y_baseline, 1)
            regression_line_baseline = np.poly1d(coeffs_baseline)
            ax2.plot(self.y_true, regression_line_baseline(self.y_true),
                     '-', color=colors['baseline_model'],
                     label='Regression Line', alpha=styles['alpha'])

            ax2.set_xlabel('Ground Truth Flux', fontsize=font_sizes['axis'])
            ax2.set_ylabel('Predicted Flux', fontsize=font_sizes['axis'])
            ax2.set_title('Baseline Model Performance', fontsize=font_sizes['title'])
            ax2.legend(fontsize=font_sizes['legend'])
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])

        plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "regression_comparison.png")
        plt.savefig(plot_path, dpi=self.viz_settings['plot_settings']['dpi'],
                    bbox_inches='tight', facecolor=self.viz_settings['plot_settings']['face_color'])
        plt.close()
        print(f"Saved regression comparison plot to {plot_path}")

    def _plot_error_comparison(self):
        """Generate error comparison plots"""
        colors = self.viz_settings['colors']
        styles = self.viz_settings['styles']
        font_sizes = self.viz_settings['plot_settings']['font_sizes']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals comparison
        main_residuals = self.y_pred - self.y_true
        baseline_residuals = self.y_baseline - self.y_true

        ax1.scatter(self.y_true, main_residuals, alpha=styles['alpha'],
                    label='Main Model', color=colors['main_model'])
        ax1.scatter(self.y_true, baseline_residuals, alpha=styles['alpha'],
                    label='Baseline Model', color=colors['baseline_model'])
        ax1.axhline(y=0, color='k', linestyle='--', alpha=styles['alpha'])
        ax1.set_xlabel('Ground Truth', fontsize=font_sizes['axis'])
        ax1.set_ylabel('Residuals', fontsize=font_sizes['axis'])
        ax1.set_title('Residuals vs Ground Truth', fontsize=font_sizes['title'])
        ax1.legend(fontsize=font_sizes['legend'])
        ax1.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # Error histograms
        ax2.hist(np.abs(main_residuals), bins=30, alpha=styles['alpha'],
                 label='Main Model', color=colors['main_model'])
        ax2.hist(np.abs(baseline_residuals), bins=30, alpha=styles['alpha'],
                 label='Baseline Model', color=colors['baseline_model'])
        ax2.set_xlabel('Absolute Error', fontsize=font_sizes['axis'])
        ax2.set_ylabel('Frequency', fontsize=font_sizes['axis'])
        ax2.set_title('Error Distribution', fontsize=font_sizes['title'])
        ax2.legend(fontsize=font_sizes['legend'])
        ax2.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])

        # Prediction scatter comparison
        ax3.scatter(self.y_baseline, self.y_pred, alpha=styles['alpha'],
                    color=colors.get('comparison', 'purple'))
        min_val = min(min(self.y_baseline), min(self.y_pred))
        max_val = max(max(self.y_baseline), max(self.y_pred))
        ax3.plot([min_val, max_val], [min_val, max_val], '--k', alpha=styles['alpha'])
        ax3.set_xlabel('Baseline Predictions', fontsize=font_sizes['axis'])
        ax3.set_ylabel('Main Model Predictions', fontsize=font_sizes['axis'])
        ax3.set_title('Model Predictions Comparison', fontsize=font_sizes['title'])
        ax3.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])
        ax3.set_xscale('log')
        ax3.set_yscale('log')

        # Error reduction plot
        error_improvement = np.abs(baseline_residuals) - np.abs(main_residuals)
        ax4.hist(error_improvement, bins=30, alpha=styles['alpha'],
                 color=colors.get('improvement', 'green'))
        ax4.axvline(x=0, color='k', linestyle='--', alpha=styles['alpha'])
        ax4.set_xlabel('Error Reduction (Baseline - Main)', fontsize=font_sizes['axis'])
        ax4.set_ylabel('Frequency', fontsize=font_sizes['axis'])
        ax4.set_title('Error Improvement Distribution', fontsize=font_sizes['title'])
        ax4.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])
        ax4.set_xscale('log')
        ax4.set_yscale('log')

        plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "error_comparison.png")
        plt.savefig(plot_path, dpi=self.viz_settings['plot_settings']['dpi'],
                    bbox_inches='tight', facecolor=self.viz_settings['plot_settings']['face_color'])
        plt.close()
        print(f"Saved error comparison plot to {plot_path}")

    def _plot_metrics_comparison(self, metrics_list):
        """Generate metrics comparison bar plot"""
        colors = self.viz_settings['colors']
        styles = self.viz_settings['styles']
        font_sizes = self.viz_settings['plot_settings']['font_sizes']

        main_metrics = metrics_list[0]
        baseline_metrics = metrics_list[1]

        metrics_names = ['RMSE', 'MSE', 'MAE', 'R2', 'TSS']
        main_values = [main_metrics[m] for m in metrics_names]
        baseline_values = [baseline_metrics[m] for m in metrics_names]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, main_values, width,
                       label='Main Model', color=colors['main_model'], alpha=styles['alpha'])
        bars2 = ax.bar(x + width/2, baseline_values, width,
                       label='Baseline Model', color=colors['baseline_model'], alpha=styles['alpha'])

        ax.set_xlabel('Metrics', fontsize=font_sizes['axis'])
        ax.set_ylabel('Values', fontsize=font_sizes['axis'])
        ax.set_title('Performance Metrics Comparison', fontsize=font_sizes['title'])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=font_sizes['legend'])
        ax.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])

        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=font_sizes['legend']-2)

        add_value_labels(bars1)
        add_value_labels(bars2)

        plt.tight_layout()
        plot_path = os.path.join(self.comparison_dir, "metrics_comparison.png")
        plt.savefig(plot_path, dpi=self.viz_settings['plot_settings']['dpi'],
                    bbox_inches='tight', facecolor=self.viz_settings['plot_settings']['face_color'])
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
        baseline_data = pd.read_csv(self.baseline_csv_path) if self.baseline_csv_path else None
        if baseline_data is not None and 'timestamp' in baseline_data.columns:
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
            target_shape = self.config['data']['weights']['target_shape']
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
            baseline_row = baseline_csv_data_global[baseline_csv_data_global['timestamp'] == target_time] if baseline_csv_data_global is not None else None

            if main_row.empty:
                print(f"No main model data found for timestamp {timestamp}")
                return None, None, None

            # Extract data using correct column names
            current_data = {
                'groundtruth': main_row.iloc[0]['groundtruth'],
                'predictions': main_row.iloc[0]['predictions'],
                'timestamp': target_time
            }

            # Add baseline predictions if available
            if baseline_row is not None and not baseline_row.empty:
                current_data['baseline_predictions'] = baseline_row.iloc[0]['predictions']
            else:
                current_data['baseline_predictions'] = None

            # Create window data (get surrounding timestamps within window_hours)
            time_window_start = target_time - pd.Timedelta(hours=window_hours/2)
            time_window_end = target_time + pd.Timedelta(hours=window_hours/2)

            # Filter data within window
            main_window = csv_data_global[
                (csv_data_global['timestamp'] >= time_window_start) &
                (csv_data_global['timestamp'] <= time_window_end)
                ].copy()

            baseline_window = baseline_csv_data_global[
                (baseline_csv_data_global['timestamp'] >= time_window_start) &
                (baseline_csv_data_global['timestamp'] <= time_window_end)
                ].copy() if baseline_csv_data_global is not None else None

            # Merge the windows for plotting
            window_data = main_window[['timestamp', 'groundtruth', 'predictions']].copy()

            if baseline_window is not None and not baseline_window.empty:
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
        """Worker function to generate a single frame using config settings"""
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

            # Generate frame path
            save_path = os.path.join(self.frames_dir, f"{timestamp}.png")

            # Create figure with config settings
            fig = plt.figure(figsize=self.viz_settings['plot_settings']['figure_size'])
            fig.patch.set_facecolor(self.viz_settings['plot_settings']['face_color'])
            gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 2.8],
                                  hspace=0.2, wspace=0.2)

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
                ax.imshow(attention_data, cmap='hot', origin='lower',
                          alpha=0.35, norm=att_norm)
                ax.set_title(f'AIA {wavs[i]} Å',
                             fontsize=self.viz_settings['plot_settings']['font_sizes']['axis'],
                             color='white')
                ax.axis('off')

            # Plot SXR data with configurable colors
            sxr_ax = fig.add_subplot(gs[:, 3])
            sxr_ax.set_facecolor('#2a2a3e')

            if sxr_window is not None and not sxr_window.empty:
                # Plot using config colors
                colors = self.viz_settings['colors']
                styles = self.viz_settings['styles']
                font_sizes = self.viz_settings['plot_settings']['font_sizes']

                # Plot ground truth
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['groundtruth'],
                            linestyle='-', marker='o',
                            color=colors['ground_truth'],
                            linewidth=styles['line_width'],
                            markersize=styles['marker_size'],
                            alpha=styles['alpha'],
                            label='Ground Truth')

                # Plot main model predictions
                sxr_ax.plot(sxr_window['timestamp'], sxr_window['predictions'],
                            linestyle='-', marker='o',
                            color=colors['main_model'],
                            linewidth=styles['line_width'],
                            markersize=styles['marker_size'],
                            alpha=styles['alpha'],
                            label='New Model')

                # Plot baseline predictions if available
                if 'baseline_predictions' in sxr_window.columns and sxr_window['baseline_predictions'].notna().any():
                    sxr_ax.plot(sxr_window['timestamp'], sxr_window['baseline_predictions'],
                                linestyle='-', marker='o',
                                color=colors['baseline_model'],
                                linewidth=styles['line_width'],
                                markersize=styles['marker_size'],
                                alpha=styles['alpha'],
                                label='Baseline Model')

                # Mark current time
                if sxr_current is not None:
                    sxr_ax.axvline(target_time,
                                   color=colors['current_time'],
                                   linestyle='--',
                                   linewidth=styles['line_width'],
                                   alpha=styles['alpha'],
                                   label='Current Time')

                    # Create info text with all available values
                    info_lines = [
                        "Current Values:",
                        f"GT: {sxr_current['groundtruth']:.2e}",
                        f"New: {sxr_current['predictions']:.2e}"
                    ]
                    if sxr_current['baseline_predictions'] is not None:
                        info_lines.append(f"Base: {sxr_current['baseline_predictions']:.2e}")

                    info_text = "\n".join(info_lines)
                    sxr_ax.text(0.02, 0.98, info_text, transform=sxr_ax.transAxes,
                                fontsize=font_sizes['legend']-2,
                                color='white',
                                verticalalignment='top',
                                bbox=dict(boxstyle='round',
                                          facecolor='black',
                                          alpha=0.7))

                sxr_ax.set_ylabel('SXR Flux', fontsize=font_sizes['axis'], color='white')
                sxr_ax.set_xlabel('Time', fontsize=font_sizes['axis'], color='white')
                sxr_ax.set_title('SXR Data Comparison', fontsize=font_sizes['title'], color='white')
                sxr_ax.legend(fontsize=font_sizes['legend'], loc='upper right')
                sxr_ax.grid(True, alpha=self.viz_settings['plot_settings']['grid_alpha'])
                sxr_ax.tick_params(axis='x', rotation=45, labelsize=font_sizes['tick'], colors='white')
                sxr_ax.tick_params(axis='y', labelsize=font_sizes['tick'], colors='white')
                try:
                    sxr_ax.set_yscale('log')
                except:
                    pass  # Skip log scale if data doesn't support it
            else:
                sxr_ax.text(0.5, 0.5, 'No SXR Data\nAvailable',
                            transform=sxr_ax.transAxes,
                            fontsize=self.viz_settings['plot_settings']['font_sizes']['title'],
                            color='white',
                            horizontalalignment='center',
                            verticalalignment='center')
                sxr_ax.set_title('SXR Data Comparison',
                                 fontsize=self.viz_settings['plot_settings']['font_sizes']['title'],
                                 color='white')

            for spine in sxr_ax.spines.values():
                spine.set_color('white')

            plt.suptitle(f'Timestamp: {timestamp}',
                         color='white',
                         fontsize=self.viz_settings['plot_settings']['font_sizes']['title'])
            plt.tight_layout()
            plt.savefig(save_path,
                        dpi=self.viz_settings['plot_settings']['dpi'],
                        facecolor=self.viz_settings['plot_settings']['face_color'])
            plt.close()

            print(f"Worker {os.getpid()}: Completed {timestamp}")
            return save_path

        except Exception as e:
            print(f"Worker {os.getpid()}: Error processing {timestamp}: {e}")
            plt.close('all')
            return None

    def create_attention_movie(self, timestamps):
        """Generate attention visualization movie using config settings"""
        print(f"Processing {len(timestamps)} timestamps")

        # Load CSV data
        csv_data, baseline_csv_data = self.load_csv_data()

        # Process frames in parallel
        num_processes = min(os.cpu_count(), len(timestamps))
        num_processes = max(1, num_processes - 1)

        with Pool(processes=num_processes,
                  initializer=self.init_worker,
                  initargs=(csv_data, baseline_csv_data)) as pool:
            results = pool.map(self.generate_frame_worker, timestamps)

        # Compile frames into movie
        frame_paths = sorted([p for p in results if p is not None])
        if not frame_paths:
            print("No frames generated - skipping movie creation")
            return

        movie_path = os.path.join(self.movies_dir, "AIA_attention_movie.mp4")
        with imageio.get_writer(
                movie_path,
                fps=self.config['output']['movies']['fps'],
                quality=self.config['output']['movies'].get('quality', 8)
        ) as writer:
            for path in frame_paths:
                writer.append_data(imageio.imread(path))

        # Cleanup if configured
        if self.config['output']['movies']['cleanup_frames']:
            for path in frame_paths:
                os.remove(path)
            print("Cleaned up frame files")

    def _generate_timestamps_from_config(self):
        """Generate timestamps based on config settings"""
        start_time = datetime.strptime(
            self.config['timestamp']['start_time'],
            "%Y-%m-%dT%H:%M:%S"
        )
        end_time = datetime.strptime(
            self.config['timestamp']['end_time'],
            "%Y-%m-%dT%H:%M:%S"
        )
        cadence = int(self.config['timestamp']['cadence'])

        timestamps = []
        current_time = start_time
        while current_time <= end_time:
            timestamps.append(current_time.strftime("%Y-%m-%dT%H:%M:%S"))
            current_time += timedelta(minutes=cadence)

        return timestamps

    def run_full_evaluation(self, timestamps=None):
        """Run complete evaluation pipeline"""
        print("=== Solar Flare Evaluation ===")
        print(f"Configuration: {self.config}")

        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = self._generate_timestamps_from_config()

        # Load data and calculate metrics
        self.load_data()
        metrics = self.calculate_metrics()
        print("\nPerformance Metrics:\n", metrics.to_string(index=False))

        # Generate visualizations
        if timestamps:
            self.create_attention_movie(timestamps)

        print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solar Flare Evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation config YAML')
    args = parser.parse_args()

    evaluator = SolarFlareEvaluator(config_path=args.config)
    evaluator.run_full_evaluation()