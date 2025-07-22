
import argparse
import os
from datetime import datetime
import re

import yaml
import itertools
import wandb
import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import MSELoss
from SDOAIA_dataloader import AIA_GOESDataModule
from models.vision_transformer_custom import ViT
from models.linear_and_hybrid import LinearIrradianceModel, HybridIrradianceModel
from callback import ImagePredictionLogger_SXR
from pytorch_lightning.callbacks import Callback


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

# Debug: Print resolved paths
print("Resolved paths:")
print(f"AIA dir: {config_data['data']['aia_dir']}")
print(f"SXR dir: {config_data['data']['sxr_dir']}")
print(f"Checkpoints dir: {config_data['data']['checkpoints_dir']}")

# Debug: Print resolved paths
print("Resolved paths:")
print(f"AIA dir: {config_data['data']['aia_dir']}")
print(f"SXR dir: {config_data['data']['sxr_dir']}")
print(f"Checkpoints dir: {config_data['data']['checkpoints_dir']}")

sxr_norm = np.load(config_data['data']['sxr_norm_path'])

n = 0

torch.manual_seed(config_data['model']['seed'])
np.random.seed(config_data['model']['seed'])

# DataModule
data_loader = AIA_GOESDataModule(
    aia_train_dir= config_data['data']['aia_dir']+"/train",
    aia_val_dir=config_data['data']['aia_dir']+"/val",
    aia_test_dir=config_data['data']['aia_dir']+"/test",
    sxr_train_dir=config_data['data']['sxr_dir']+"/train",
    sxr_val_dir=config_data['data']['sxr_dir']+"/val",
    sxr_test_dir=config_data['data']['sxr_dir']+"/test",
    batch_size=config_data['model']['batch_size'],
    num_workers=os.cpu_count(),
    sxr_norm=sxr_norm,
)
data_loader.setup()

# Logger
#wb_name = f"{instrument}_{n}" if len(combined_parameters) > 1 else "aia_sxr_model"
wandb_logger = WandbLogger(
    entity=config_data['wandb']['entity'],
    project=config_data['wandb']['project'],
    job_type=config_data['wandb']['job_type'],
    tags=config_data['wandb']['tags'],
    name=config_data['wandb']['wb_name'],
    notes=config_data['wandb']['notes'],
    config=config_data['model']
)

# Logging callback
total_n_valid = len(data_loader.val_ds)
plot_data = [data_loader.val_ds[i] for i in range(0, total_n_valid, max(1, total_n_valid // 4))]
plot_samples = plot_data  # Keep as list of ((aia, sxr), target)
#sxr_callback = SXRPredictionLogger(plot_samples)

sxr_plot_callback = ImagePredictionLogger_SXR(plot_samples, sxr_norm)


class PTHCheckpointCallback(Callback):
    def __init__(self, dirpath, monitor='valid_loss', mode='min', save_top_k=1, filename_prefix="model"):
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
    monitor='valid_loss',
    mode='min',
    save_top_k=1,
    filename=f"{config_data['wandb']['wb_name']}-{{epoch:02d}}-{{valid_loss:.4f}}.pth"
)

pth_callback = PTHCheckpointCallback(
    dirpath=config_data['data']['checkpoints_dir'],
    monitor='valid_loss',
    mode='min',
    save_top_k=1,
    filename_prefix=config_data['wandb']['wb_name']
)

# Model
if config_data['selected_model'] == 'linear':
    model = LinearIrradianceModel(
        d_input=6,
        d_output=1,
        lr= config_data['model']['lr'],
        loss_func=MSELoss()
    )
elif config_data['selected_model'] == 'hybrid':
    model = HybridIrradianceModel(
        d_input=6,
        d_output=1,
        cnn_model=config_data['model']['cnn_model'],
        ln_model=True,
        cnn_dp=config_data['model']['cnn_dp'],
        lr=config_data['model']['lr'],
    )
elif config_data['selected_model'] == 'ViT':
    print("Using ViT")
#     model = ViT(embed_dim=config_data['vit']['embed_dim'], hidden_dim=config_data['vit']['hidden_dim'],
#                 num_channels=config_data['vit']['num_channels'],num_heads=config_data['vit']['num_heads'],
#                 num_layers=config_data['vit']['num_layers'], num_classes=config_data['vit']['num_classes'],
#                 patch_size=config_data['vit']['patch_size'], num_patches=config_data['vit']['num_patches'],
#                 dropout=config_data['vit']['dropout'], lr=config_data['vit']['lr'])
    model = ViT(model_kwargs=config_data['vit'])
else:
    raise NotImplementedError(f"Architecture {config_data['selected_model']} not supported.")

# Trainer
trainer = Trainer(
    default_root_dir=config_data['data']['checkpoints_dir'],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=config_data['model']['epochs'],
    callbacks=[sxr_plot_callback, pth_callback],
    logger=wandb_logger,
    log_every_n_steps=10
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