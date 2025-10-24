#!/usr/bin/env python3
"""
Data Processing Pipeline Orchestrator

This script orchestrates the four main data processing steps:
1. EUV data cleaning (euv_data_cleaning.py) - removes bad AIA files
2. ITI data processing (iti_data_processing.py) - processes the good data
3. SXR data processing (sxr_data_processing.py) - processes and combines GOES X-ray data
4. Data alignment (align_data.py) - concatenates GOES data and checks for missing data

Each step can be skipped if it's already completed.

Configuration:
- Use --config to specify a custom configuration file (YAML or Python)
- Use --show-config to display current configuration
- Use --create-template to create a YAML configuration template
"""

import os
import sys
import subprocess
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from pipeline_config import PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataProcessingPipeline:
    def __init__(self, base_dir=None, config=None):
        """
        Initialize the data processing pipeline.
        
        Args:
            base_dir: Base directory for the project. If None, uses current script's directory.
            config: PipelineConfig instance. If None, uses default configuration.
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        # Load configuration
        self.config = config if config is not None else PipelineConfig()
        
        # Define script paths
        self.scripts = {
            'euv_cleaning': self.base_dir / 'euv_data_cleaning.py',
            'iti_processing': self.base_dir / 'iti_data_processing.py',
            'sxr_processing': self.base_dir / 'sxr_data_processing.py',
            'align_data': self.base_dir / 'align_data.py'
        }
        
        # Define step names and descriptions
        self.steps = {
            'euv_cleaning': {
                'name': 'EUV Data Cleaning',
                'description': 'Remove bad AIA files based on timestamp validation',
                'output_check': self._check_euv_cleaning_complete
            },
            'iti_processing': {
                'name': 'ITI Data Processing',
                'description': 'Process good AIA data using ITI methods',
                'output_check': self._check_iti_processing_complete
            },
            'sxr_processing': {
                'name': 'SXR Data Processing',
                'description': 'Process and combine GOES X-ray data from multiple satellites',
                'output_check': self._check_sxr_processing_complete
            },
            'align_data': {
                'name': 'Data Alignment',
                'description': 'Concatenate GOES data and check for missing data',
                'output_check': self._check_align_data_complete
            }
        }
    
    def _check_euv_cleaning_complete(self):
        """
        Check if EUV data cleaning is complete by looking for the bad files directory.
        """
        bad_files_dir = Path(self.config.get_path('euv', 'bad_files_dir'))
        if bad_files_dir.exists():
            # Check if any files were moved (indicating cleaning was done)
            wavelengths = self.config.get_path('euv', 'wavelengths')
            wavelength_dirs = [bad_files_dir / str(wl) for wl in wavelengths]
            return any(d.exists() and any(d.iterdir()) for d in wavelength_dirs)
        return False
    
    def _check_iti_processing_complete(self):
        """
        Check if ITI data processing is complete by looking for processed files.
        """
        output_dir = Path(self.config.get_path('iti', 'output_folder'))
        if output_dir.exists():
            # Check if there are processed .npy files
            npy_files = list(output_dir.glob('*.npy'))
            return len(npy_files) > 0
        return False
    
    def _check_sxr_processing_complete(self):
        """
        Check if SXR data processing is complete by looking for combined GOES files.
        """
        output_dir = Path(self.config.get_path('sxr', 'output_dir'))
        if output_dir.exists():
            # Check if there are combined GOES CSV files
            csv_files = list(output_dir.glob('combined_g*_avg1m_*.csv'))
            return len(csv_files) > 0
        return False
    
    def _check_align_data_complete(self):
        """
        Check if data alignment is complete by looking for output directories.
        """
        output_dirs = [
            Path(self.config.get_path('alignment', 'output_sxr_a_dir')),
            Path(self.config.get_path('alignment', 'output_sxr_b_dir'))
        ]
        return all(d.exists() and any(d.iterdir()) for d in output_dirs)
    
    def run_script(self, script_name, step_info):
        """
        Run a single processing script.
        
        Args:
            script_name: Name of the script to run
            step_info: Dictionary containing step information
            
        Returns:
            bool: True if successful, False otherwise
        """
        script_path = self.scripts[script_name]
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        logger.info(f"Starting {step_info['name']}...")
        logger.info(f"Description: {step_info['description']}")
        logger.info(f"Running: {script_path}")
        
        # Create environment variables for configuration
        env = os.environ.copy()
        env.update({
            'PIPELINE_CONFIG': json.dumps(self.config.config),
            'BASE_DATA_DIR': self.config.config['base_data_dir']
        })
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                env=env
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✓ {step_info['name']} completed successfully in {duration:.2f} seconds")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"✗ {step_info['name']} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            logger.error(f"✗ {step_info['name']} failed with exception: {e}")
            logger.error(f"Duration: {duration:.2f} seconds")
            return False
    
    def run_pipeline(self, force_rerun=False, specific_steps=None):
        """
        Run the data processing pipeline.
        
        Args:
            force_rerun: If True, run all steps regardless of completion status
            specific_steps: List of step names to run. If None, run all steps.
        """
        logger.info("=" * 80)
        logger.info("Starting Data Processing Pipeline")
        logger.info("=" * 80)
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Force rerun: {force_rerun}")
        if specific_steps:
            logger.info(f"Running specific steps: {', '.join(specific_steps)}")
        else:
            logger.info("Running all steps")
        logger.info("=" * 80)
        
        pipeline_start_time = time.time()
        successful_steps = 0
        failed_steps = 0
        
        # Filter steps if specific steps are requested
        steps_to_run = self.steps
        if specific_steps:
            steps_to_run = {k: v for k, v in self.steps.items() if k in specific_steps}
            if not steps_to_run:
                logger.error("No valid steps specified. Available steps: " + ", ".join(self.steps.keys()))
                return False
        
        for step_name, step_info in steps_to_run.items():
            logger.info(f"\n--- Step: {step_info['name']} ---")
            
            # Check if step is already complete
            if not force_rerun and step_info['output_check']():
                logger.info(f"✓ {step_info['name']} already completed - skipping")
                successful_steps += 1
                continue
            
            # Run the step
            if self.run_script(step_name, step_info):
                successful_steps += 1
            else:
                failed_steps += 1
                logger.error(f"Pipeline stopped due to failure in {step_info['name']}")
                break
        
        pipeline_end_time = time.time()
        total_duration = pipeline_end_time - pipeline_start_time
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Successful steps: {successful_steps}")
        logger.info(f"Failed steps: {failed_steps}")
        
        if failed_steps == 0:
            logger.info("✓ All steps completed successfully!")
        else:
            logger.error("✗ Pipeline completed with errors")
        
        logger.info("=" * 80)
        
        return failed_steps == 0

def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Processing Pipeline Orchestrator')
    parser.add_argument('--force', action='store_true', 
                       help='Force rerun all steps regardless of completion status')
    parser.add_argument('--base-dir', type=str, 
                       help='Base directory for the project (default: script directory)')
    parser.add_argument('--config', type=str,
                       help='Path to custom configuration file (YAML or Python)')
    parser.add_argument('--show-config', action='store_true',
                       help='Display current configuration and exit')
    parser.add_argument('--create-template', action='store_true',
                       help='Create a YAML configuration template file and exit')
    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration paths and exit')
    parser.add_argument('--steps', type=str,
                       help='Comma-separated list of steps to run (e.g., euv_cleaning,iti_processing). Available steps: euv_cleaning, iti_processing, sxr_processing, align_data. Can also be configured in the config file under steps.run_steps')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_template:
        config = PipelineConfig()
        config.save_config_template()
        return
    
    # Load configuration
    config = PipelineConfig(args.config)
    
    if args.show_config:
        config.print_config()
        return
    
    if args.validate:
        is_valid, missing_paths = config.validate_paths()
        if is_valid:
            print("✓ All required paths exist")
        else:
            print("✗ Missing required paths:")
            for path in missing_paths:
                print(f"  - {path}")
        return
    
    # Parse steps argument - command line takes precedence over config
    specific_steps = None
    if args.steps:
        specific_steps = [step.strip() for step in args.steps.split(',')]
        # Validate step names
        valid_steps = ['euv_cleaning', 'iti_processing', 'sxr_processing', 'align_data']
        invalid_steps = [step for step in specific_steps if step not in valid_steps]
        if invalid_steps:
            logger.error(f"Invalid step names: {', '.join(invalid_steps)}")
            logger.error(f"Valid steps are: {', '.join(valid_steps)}")
            sys.exit(1)
    else:
        # Get steps from configuration
        config_steps = config.get_steps()
        if config_steps:
            specific_steps = config_steps
            logger.info(f"Using steps from configuration: {', '.join(specific_steps)}")
    
    # Create pipeline instance
    pipeline = DataProcessingPipeline(args.base_dir, config)
    
    # Validate paths before running (only for the steps that will be run)
    is_valid, missing_paths = config.validate_paths(specific_steps)
    if not is_valid:
        logger.error("Configuration validation failed. Missing required paths:")
        for path in missing_paths:
            logger.error(f"  - {path}")
        logger.error("Use --validate to check configuration")
        sys.exit(1)
    
    # Create necessary directories (only for the steps that will be run)
    config.create_directories(specific_steps)
    
    # Run the pipeline
    success = pipeline.run_pipeline(force_rerun=args.force, specific_steps=specific_steps)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
