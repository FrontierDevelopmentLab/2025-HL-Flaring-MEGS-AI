#!/usr/bin/env python3
"""
Pipeline Configuration Management

This module handles loading and managing configuration for the data processing pipeline.
It supports both YAML and Python configuration files.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List


class PipelineConfig:
    """Configuration manager for the data processing pipeline."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration.
        
        Args:
            config_path: Path to configuration file (YAML or Python). If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            return self._load_from_file(self.config_path)
        else:
            return self._get_default_config()
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or Python file."""
        config_path = Path(config_path)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.py':
            # Load Python configuration
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            return config_module.CONFIG
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'base_data_dir': '/mnt/data',
            'euv': {
                'input_folder': '/mnt/data/AUGUST/SDO-AIA-timespan',
                'bad_files_dir': '/mnt/data/AUGUST/SDO-AIA_bad',
                'wavelengths': [94, 131, 171, 193, 211, 304]
            },
            'iti': {
                'input_folder': '/mnt/data/AUGUST/SDO-AIA-timespan',
                'output_folder': '/mnt/data/AUGUST/AIA_ITI',
                'wavelengths': [94, 131, 171, 193, 211, 304]
            },
            'sxr': {
                'input_dir': '/mnt/data/AUGUST/GOES-timespan',
                'output_dir': '/mnt/data/AUGUST/combined'
            },
            'alignment': {
                'goes_data_dir': '/mnt/data/AUGUST/combined',
                'aia_processed_dir': '/mnt/data/AUGUST/AIA_ITI',
                'output_sxr_a_dir': '/mnt/data/AUGUST/GOES-SXR-A',
                'output_sxr_b_dir': '/mnt/data/AUGUST/GOES-SXR-B',
                'aia_missing_dir': '/mnt/data/AUGUST/AIA_ITI_MISSING'
            },
            'processing': {
                'max_processes': None,
                'batch_size_multiplier': 4,
                'min_batch_size': 1
            },
            'steps': {
                'run_steps': []
            }
        }
    
    def get_path(self, section: str, key: str) -> str:
        """
        Get a path from configuration.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            
        Returns:
            Path value
        """
        return self.config[section][key]
    
    def get_steps(self) -> List[str]:
        """
        Get the list of steps to run from configuration.
        
        Returns:
            List of step names. Empty list means run all steps.
        """
        steps = self.config.get('steps', {}).get('run_steps', [])
        if steps is None:
            return []
        return steps
    
    def print_config(self):
        """Print the current configuration."""
        print("Current Pipeline Configuration:")
        print("=" * 50)
        print(json.dumps(self.config, indent=2, default=str))
    
    def validate_paths(self, specific_steps: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate that all required paths exist for the specified steps.
        
        Args:
            specific_steps: List of step names to validate. If None, validates all steps.
        
        Returns:
            Tuple of (is_valid, missing_paths)
        """
        missing_paths = []
        
        # Check base data directory (always required)
        base_dir = Path(self.config['base_data_dir'])
        if not base_dir.exists():
            missing_paths.append(f"base_data_dir: {base_dir}")
        
        # If no specific steps provided, check all steps
        if specific_steps is None:
            specific_steps = ['euv_cleaning', 'iti_processing', 'sxr_processing', 'align_data']
        
        # Check EUV paths (for euv_cleaning step)
        if 'euv_cleaning' in specific_steps:
            euv_input = Path(self.config['euv']['input_folder'])
            if not euv_input.exists():
                missing_paths.append(f"euv.input_folder: {euv_input}")
        
        # Check ITI paths (for iti_processing step)
        if 'iti_processing' in specific_steps:
            iti_input = Path(self.config['iti']['input_folder'])
            if not iti_input.exists():
                missing_paths.append(f"iti.input_folder: {iti_input}")
        
        # Check SXR paths (for sxr_processing step)
        if 'sxr_processing' in specific_steps:
            sxr_input = Path(self.config['sxr']['input_dir'])
            if not sxr_input.exists():
                missing_paths.append(f"sxr.input_dir: {sxr_input}")
        
        # Check alignment paths (for align_data step)
        if 'align_data' in specific_steps:
            alignment_config = self.config['alignment']
            for key, path in alignment_config.items():
                if path and path != 'na':  # Skip 'na' values
                    if not Path(path).exists():
                        missing_paths.append(f"alignment.{key}: {path}")
        
        return len(missing_paths) == 0, missing_paths
    
    def create_directories(self, specific_steps: List[str] = None):
        """Create necessary output directories for the specified steps."""
        directories_to_create = []
        
        # If no specific steps provided, create all directories
        if specific_steps is None:
            specific_steps = ['euv_cleaning', 'iti_processing', 'sxr_processing', 'align_data']
        
        # Add directories based on steps that will be run
        if 'euv_cleaning' in specific_steps:
            directories_to_create.append(self.config['euv']['bad_files_dir'])
        
        if 'iti_processing' in specific_steps:
            directories_to_create.append(self.config['iti']['output_folder'])
        
        if 'sxr_processing' in specific_steps:
            directories_to_create.append(self.config['sxr']['output_dir'])
        
        if 'align_data' in specific_steps:
            directories_to_create.extend([
                self.config['alignment']['output_sxr_a_dir'],
                self.config['alignment']['output_sxr_b_dir'],
                self.config['alignment']['aia_missing_dir']
            ])
        
        for directory in directories_to_create:
            if directory and directory != 'na':  # Skip 'na' values
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_config_template(self, output_path: str = None):
        """Save a configuration template to file."""
        if output_path is None:
            output_path = Path(__file__).parent / 'pipeline_config_template.yaml'
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration template saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline Configuration Manager')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--validate', action='store_true', help='Validate configuration paths')
    parser.add_argument('--create-template', action='store_true', help='Create configuration template')
    
    args = parser.parse_args()
    
    config = PipelineConfig(args.config)
    
    if args.show:
        config.print_config()
    
    if args.validate:
        is_valid, missing = config.validate_paths()
        if is_valid:
            print("✓ All required paths exist")
        else:
            print("✗ Missing required paths:")
            for path in missing:
                print(f"  - {path}")
    
    if args.create_template:
        config.save_config_template()
