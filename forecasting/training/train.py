
import argparse
import os
from datetime import datetime
import re
from multiprocessing import Pool, cpu_count
from functools import partial

import yaml
import wandb
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import MSELoss, HuberLoss
from pathlib import Path
import sys
# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.data_loaders.SDOAIA_dataloader import AIA_GOESDataModule
from forecasting.models.vision_transformer_custom import ViT
from forecasting.models.linear_and_hybrid import LinearIrradianceModel, HybridIrradianceModel
from forecasting.models.vit_patch_model import ViT as ViTPatch
from forecasting.models.vit_patch_model_uncertainty import ViTUncertainty
from forecasting.models import FusionViTHybrid
from forecasting.models.CNN_Patch import CNNPatch
from forecasting.models.vit_patch_model_local import ViTLocal
from callback import ImagePredictionLogger_SXR, AttentionMapCallback

from pytorch_lightning.callbacks import Callback

from forecasting.models.FastSpectralNet import FastViTFlaringModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_DEBUG"] = "WARN"
# Shared memory optimizations
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads

def print_gpu_memory(stage=""):
    """Print GPU memory usage for monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory {stage} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    else:
        print(f"No GPU available for memory monitoring {stage}")

def resolve_config_variables(config_dict):
    """Recursively resolve ${variable} references within the config"""

    # Extract variables defined at root level (like base_data_dir, base_checkpoint_dir)
    variables = {}
    for key, value in config_dict.items():
        if isinstance(value, str) and not value.startswith('${'):
            variables[key] = value

    def substitute_value(value, variables):
        if isinstance(value, str):
            # Replace ${var_name} with actual values
            pattern = r'\$\{([^}]+)\}'
            for match in re.finditer(pattern, value):
                var_name = match.group(1)
                if var_name in variables:
                    value = value.replace(f'${{{var_name}}}', variables[var_name])
        return value

    def recursive_substitute(obj, variables):
        if isinstance(obj, dict):
            return {k: recursive_substitute(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_substitute(item, variables) for item in obj]
        else:
            return substitute_value(obj, variables)

    return recursive_substitute(config_dict, variables)


# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML.')
args = parser.parse_args()

# Load config with variable substitution
with open(args.config, 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.SafeLoader)

# Resolve variables like ${base_data_dir}
config_data = resolve_config_variables(config_data)

# GPU Memory Isolation for Multi-GPU Systems
gpu_id = config_data.get('gpu_id', 0)
if gpu_id != -1:  # Only if using GPU
    # Set CUDA device visibility to only the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES to GPU {gpu_id}")
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Cleared CUDA cache for GPU {gpu_id}")
        
    # Set memory allocation strategy for better isolation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"
    
    # Disable memory sharing between processes
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"GPU Memory Isolation configured for GPU {gpu_id}")
else:
    print("Using CPU - no GPU memory isolation needed")

# Debug: Print resolved paths
print("Resolved paths:")
print(f"AIA dir: {config_data['data']['aia_dir']}")
print(f"SXR dir: {config_data['data']['sxr_dir']}")
print(f"Checkpoints dir: {config_data['data']['checkpoints_dir']}")

# Load normalization parameters
if 'sxr_norm_path' in config_data['data']:
    # Single normalization file (backward compatibility)
    sxr_norm = np.load(config_data['data']['sxr_norm_path'])
elif 'sxr_norm_paths' in config_data['data']:
    # Separate normalization files for each channel
    sxr_norm = {
        'a': np.load(config_data['data']['sxr_norm_paths']['a']),
        'b': np.load(config_data['data']['sxr_norm_paths']['b'])
    }
else:
    raise ValueError("Either 'sxr_norm_path' or 'sxr_norm_paths' must be specified in config")

n = 0

torch.manual_seed(config_data['megsai']['seed'])
np.random.seed(config_data['megsai']['seed'])

training_wavelengths = config_data['wavelengths']


# Validate SXR channel configuration
sxr_channels = config_data.get('sxr_channels', ['a', 'b'])
if sxr_channels == ['a']:
    raise ValueError("Cannot train on SXR-A only. Must include SXR-B or train on SXR-B only.")
if 'b' not in sxr_channels:
    raise ValueError("SXR-B channel must be included in sxr_channels.")

# DataModule
data_loader = AIA_GOESDataModule(
    aia_train_dir= config_data['data']['aia_dir']+"/train",
    aia_val_dir=config_data['data']['aia_dir']+"/val",
    aia_test_dir=config_data['data']['aia_dir']+"/test",
    sxr_train_dir=config_data['data']['sxr_dir']+"/train",
    sxr_val_dir=config_data['data']['sxr_dir']+"/val",
    sxr_test_dir=config_data['data']['sxr_dir']+"/test",
    batch_size=config_data['batch_size'],
    num_workers=min(8, os.cpu_count()),  # Limit workers to prevent shm issues
    sxr_norm=sxr_norm,
    wavelengths=training_wavelengths,
    oversample=config_data['oversample'],
    balance_strategy=config_data['balance_strategy'],
    sxr_channels=sxr_channels,
)
data_loader.setup()

# Monitor memory after data loading
print_gpu_memory("after data loading")

# Logger
#wb_name = f"{instrument}_{n}" if len(combined_parameters) > 1 else "aia_sxr_model"
wandb_logger = WandbLogger(
    entity=config_data['wandb']['entity'],
    project=config_data['wandb']['project'],
    job_type=config_data['wandb']['job_type'],
    tags=config_data['wandb']['tags'],
    name=config_data['wandb']['wb_name'],
    notes=config_data['wandb']['notes'],
    config=config_data['megsai']
)

# Logging callback
total_n_valid = len(data_loader.val_ds)
plot_data = [data_loader.val_ds[i] for i in range(0, total_n_valid, max(1, total_n_valid // 4))]
plot_samples = plot_data  # Keep as list of ((aia, sxr), target)
#sxr_callback = SXRPredictionLogger(plot_samples)

sxr_plot_callback = ImagePredictionLogger_SXR(plot_samples, sxr_norm)
# Attention map callback - get patch size from config
patch_size = config_data.get('vit_custom', {}).get('patch_size', 8)
attention = AttentionMapCallback(patch_size=patch_size)


class PTHCheckpointCallback(Callback):
    def __init__(self, dirpath, monitor='val_total_loss', mode='min', save_top_k=1, filename_prefix="model"):
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_prefix = filename_prefix
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        is_better = (self.mode == 'min' and current_score < self.best_score) or \
                    (self.mode == 'max' and current_score > self.best_score)

        if is_better:
            self.best_score = current_score
            # Save as .pth file
            filename = f"{self.filename_prefix}-epoch={trainer.current_epoch:02d}-{self.monitor}={current_score:.4f}.pth"
            filepath = os.path.join(self.dirpath, filename)

            torch.save({
                'model': pl_module,
                'epoch': trainer.current_epoch,
                'optimizer_state_dict': trainer.optimizers[0].state_dict(),
                'loss': current_score,
            }, filepath)




# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=config_data['data']['checkpoints_dir'],
    monitor='val_total_loss',
    mode='min',
    save_top_k=10,
    filename=f"{config_data['wandb']['wb_name']}-{{epoch:02d}}-{{val_total_loss:.4f}}"
)

pth_callback = PTHCheckpointCallback(
    dirpath=config_data['data']['checkpoints_dir'],
    monitor='val_total_loss',
    mode='min',
    save_top_k=1,
    filename_prefix=config_data['wandb']['wb_name']
)

def process_batch(batch_data, sxr_norm, c_threshold, m_threshold, x_threshold, sxr_channels=['a', 'b']):
    """Process a single batch and return counts for different flare classes."""
    from forecasting.models.vit_patch_model_local import unnormalize_sxr
    
    batch, batch_idx = batch_data
    _, sxr = batch  # sxr is [B, num_channels] based on sxr_channels config
    
    # Unnormalize the SXR values
    sxr_un = torch.zeros_like(sxr)
    for i in range(len(sxr_channels)):
        if isinstance(sxr_norm, dict):
            # Use separate normalization for each channel
            sxr_un[:, i] = unnormalize_sxr(sxr[:, i], sxr_norm, channel_idx=i)
        else:
            # Use single normalization (backward compatibility)
            sxr_un[:, i] = unnormalize_sxr(sxr[:, i], sxr_norm)
    
    # Use the last channel (typically 'b') for flare classification as it's typically more sensitive
    # If only one channel, use that one
    channel_idx = len(sxr_channels) - 1  # Use last channel (typically 'b')
    sxr_un_flat = sxr_un[:, channel_idx].view(-1).cpu().numpy()
    
    total = len(sxr_un_flat)
    quiet_count = ((sxr_un_flat < c_threshold)).sum()
    c_count = ((sxr_un_flat >= c_threshold) & (sxr_un_flat < m_threshold)).sum()
    m_count = ((sxr_un_flat >= m_threshold) & (sxr_un_flat < x_threshold)).sum()
    x_count = ((sxr_un_flat >= x_threshold)).sum()
    
    return {
        'total': total,
        'quiet_count': quiet_count,
        'c_count': c_count,
        'm_count': m_count,
        'x_count': x_count,
        'batch_idx': batch_idx
    }

def get_base_weights(data_loader, sxr_norm, sxr_channels=['a', 'b']):
    print("Calculating base weights from DataModule...")
    
    # Thresholds for SXR classes
    c_threshold = 1e-6
    m_threshold = 1e-5
    x_threshold = 1e-4

    from forecasting.models.vit_patch_model_local import unnormalize_sxr
    
    quiet_count = 0
    c_count = 0
    m_count = 0
    x_count = 0
    total = 0
    
    # Use the train_dataloader which already exists
    train_loader = data_loader.train_dataloader()
    print(f"Processing {len(train_loader)} batches...")
    
    for batch_idx, (aia_batch, sxr_batch) in enumerate(train_loader):
        if batch_idx % 50 == 0:
            print(f"Processed {batch_idx}/{len(train_loader)} batches...")
            
        # Unnormalize the SXR batch - sxr_batch is [B, num_channels]
        sxr_un = torch.zeros_like(sxr_batch)
        for i in range(len(sxr_channels)):
            if isinstance(sxr_norm, dict):
                # Use separate normalization for each channel
                sxr_un[:, i] = unnormalize_sxr(sxr_batch[:, i], sxr_norm, channel_idx=i)
            else:
                # Use single normalization (backward compatibility)
                sxr_un[:, i] = unnormalize_sxr(sxr_batch[:, i], sxr_norm)
        
        # Use the last channel (typically 'b') for flare classification as it's typically more sensitive
        channel_idx = len(sxr_channels) - 1  # Use last channel (typically 'b')
        sxr_un_flat = sxr_un[:, channel_idx].view(-1).cpu().numpy()
        
        batch_total = len(sxr_un_flat)
        batch_quiet = ((sxr_un_flat < c_threshold)).sum()
        batch_c = ((sxr_un_flat >= c_threshold) & (sxr_un_flat < m_threshold)).sum()
        batch_m = ((sxr_un_flat >= m_threshold) & (sxr_un_flat < x_threshold)).sum()
        batch_x = ((sxr_un_flat >= x_threshold)).sum()
        
        total += batch_total
        quiet_count += batch_quiet
        c_count += batch_c
        m_count += batch_m
        x_count += batch_x

    # Avoid division by zero
    quiet_count = max(quiet_count, 1)
    c_count = max(c_count, 1)
    m_count = max(m_count, 1)
    x_count = max(x_count, 1)

    # Inverse frequency weighting
    quiet_weight = total / (quiet_count)
    c_weight = total / (c_count)
    m_weight = total / m_count
    x_weight = total / x_count

    print("Base weights calculated")
    print(f"Total samples: {total}")
    print(f"Quiet samples: {quiet_count}, weight: {quiet_weight:.4f}")
    print(f"C samples: {c_count}, weight: {c_weight:.4f}")
    print(f"M samples: {m_count}, weight: {m_weight:.4f}")
    print(f"X samples: {x_count}, weight: {x_weight:.4f}")
    
    return {
        'quiet': quiet_weight,
        'c_class': c_weight,
        'm_class': m_weight,
        'x_class': x_weight
    }

# Model
if config_data['selected_model'] == 'linear':
    model = LinearIrradianceModel(
        d_input= len(config_data['wavelengths']),
        d_output=1,
        lr= config_data['megsai']['lr'],
        loss_func=HuberLoss(),
        weight_decay=config_data['megsai']['weight_decay'],
        cosine_restart_T0=config_data['megsai']['cosine_restart_T0'],
        cosine_restart_Tmult=config_data['megsai']['cosine_restart_Tmult'],
        cosine_eta_min=config_data['megsai']['cosine_eta_min']
    )
elif config_data['selected_model'] == 'hybrid':
    model = HybridIrradianceModel(
        d_input= len(config_data['wavelengths']),
        d_output=1,
        cnn_model=config_data['megsai']['cnn_model'],
        ln_model=True,
        cnn_dp=config_data['megsai']['cnn_dp'],
        lr=config_data['megsai']['lr'],
        weight_decay=config_data['megsai']['weight_decay'],
        cosine_restart_T0=config_data['megsai']['cosine_restart_T0'],
        cosine_restart_Tmult=config_data['megsai']['cosine_restart_Tmult'],
        cosine_eta_min=config_data['megsai']['cosine_eta_min']
    )
elif config_data['selected_model'] == 'CNNPatch':
    model = CNNPatch(model_kwargs=config_data['cnn_patch'], sxr_norm = sxr_norm)

elif config_data['selected_model'] == 'ViT':
    base_weights = get_base_weights(data_loader, sxr_norm) if config_data.get('calculate_base_weights', True) else None
    model = ViT(model_kwargs=config_data['vit_custom'], sxr_norm = sxr_norm, base_weights=base_weights)

elif config_data['selected_model'] == 'ViTPatch':
    # Calculate base weights only if configured to do so
    base_weights = get_base_weights(data_loader, sxr_norm) if config_data.get('calculate_base_weights', True) else None
    model = ViTPatch(model_kwargs=config_data['vit_custom'], sxr_norm = sxr_norm, base_weights=base_weights)

elif config_data['selected_model'] == 'ViTLocal':
    base_weights = get_base_weights(data_loader, sxr_norm, sxr_channels) if config_data.get('calculate_base_weights', True) else None
    model = ViTLocal(
        model_kwargs=config_data['vit_custom'], 
        sxr_norm=sxr_norm, 
        base_weights=base_weights,
        sxr_channels=sxr_channels
    )

elif config_data['selected_model'] == 'ViTUncertainty':
    base_weights = get_base_weights(data_loader, sxr_norm) if config_data.get('calculate_base_weights', True) else None
    model = ViTUncertainty(model_kwargs=config_data['vit_custom'], sxr_norm = sxr_norm, base_weights=base_weights)

elif config_data['selected_model'] == 'FusionViTHybrid':
    # Expect a 'fusion' section in YAML
    fusion_cfg = config_data.get('fusion', {})
    scalar_branch = fusion_cfg.get('scalar_branch', 'hybrid')
    scalar_kwargs = fusion_cfg.get('scalar_kwargs', {
        'd_input': len(config_data['wavelengths']),
        'd_output': 1,
        'cnn_model': config_data.get('megsai', {}).get('cnn_model', 'updated'),
        'cnn_dp': config_data.get('megsai', {}).get('cnn_dp', 0.75),
        'lr': fusion_cfg.get('lr', config_data.get('megsai', {}).get('lr', 1e-4)),
    })
    vit_kwargs = config_data.get('vit_custom', {})

    model = FusionViTHybrid(
        vit_kwargs=vit_kwargs,
        scalar_branch=scalar_branch,
        scalar_kwargs=scalar_kwargs,
        sxr_norm=sxr_norm,
        lr=fusion_cfg.get('lr', 1e-4),
        lambda_vit_to_target=fusion_cfg.get('lambda_vit_to_target', 0.3),
        lambda_scalar_to_target=fusion_cfg.get('lambda_scalar_to_target', 0.1),
        learnable_gate=fusion_cfg.get('learnable_gate', True),
        gate_init_bias=fusion_cfg.get('gate_init_bias', 5.0),
    )

else:
    raise NotImplementedError(f"Architecture {config_data['selected_model']} not supported.")

# Monitor memory after model creation
print_gpu_memory("after model creation")

# Set device based on config
gpu_id = config_data.get('gpu_id', 0)
if gpu_id == -1:
    accelerator = "cpu"
    devices = 1
    print("Using CPU for training")
else:
    if torch.cuda.is_available():
        accelerator = "gpu"
        # When CUDA_VISIBLE_DEVICES is set, PyTorch Lightning only sees GPU 0
        devices = [0]  # Always use device 0 since we've isolated to specific GPU
        print(f"Using GPU {gpu_id} for training (mapped to device 0 after CUDA_VISIBLE_DEVICES)")
    else:
        accelerator = "cpu"
        devices = 1
        print(f"GPU {gpu_id} not available, falling back to CPU")

# Trainer
if config_data['selected_model'] == 'ViT' or config_data['selected_model'] == 'ViTPatch' or config_data['selected_model'] == 'FusionViTHybrid':
    trainer = Trainer(
        default_root_dir=config_data['data']['checkpoints_dir'],
        accelerator=accelerator,
        devices=devices,
        max_epochs=config_data['epochs'],
        callbacks=[attention, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
    )
else:
    trainer = Trainer(
        default_root_dir=config_data['data']['checkpoints_dir'],
        accelerator=accelerator,
        devices=devices,
        max_epochs=config_data['epochs'],
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
    )
# Save checkpoint
trainer.fit(model, data_loader)

# Save final PyTorch checkpoint with model and state_dict
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_checkpoint_path = os.path.join(config_data['data']['checkpoints_dir'], f"{config_data['wandb']['wb_name']}-final-{timestamp}.pth")
torch.save({
    'model': model,
    'state_dict': model.state_dict()
}, final_checkpoint_path)
print(f"Saved final PyTorch checkpoint: {final_checkpoint_path}")
n += 1
# Finalize
wandb.finish()