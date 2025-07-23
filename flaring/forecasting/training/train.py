import argparse
import os
import sys
from datetime import datetime
import re
import yaml
import wandb
import torch
import numpy as np
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from torch.nn import MSELoss
import torch.nn.functional as F
from torch import optim
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import torch.nn as nn

# Get the absolute path to the forecasting directory
current_dir = Path(__file__).parent  # training/
forecasting_dir = current_dir.parent  # forecasting/
sys.path.insert(0, str(forecasting_dir))

# Import models from other files
from models.vision_transformer_custom import ViT, VisionTransformer
from models.FastSpectralNet import FastViTFlaringModel

# Import data modules
from data_loaders.SDOAIA_dataloader import (
    AIA_GOESDataModule,
    AIA_GOESSequenceDataModule
)

# Import callbacks
from callback import ImagePredictionLogger_SXR, AttentionMapCallback



# --------------------------
# Utility Functions
# --------------------------

def resolve_config_variables(config_dict):
    """Recursively resolve ${variable} references within the config"""
    variables = {}
    for key, value in config_dict.items():
        if isinstance(value, str) and not value.startswith('${'):
            variables[key] = value

    def substitute_value(value, variables):
        if isinstance(value, str):
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

# --------------------------
# Model Wrappers
# --------------------------

class SequenceViTWrapper(LightningModule):
    """Wrapper for ViT to handle sequence input"""
    def __init__(self, model_kwargs: Dict[str, Any], sequence_length: int = 12):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length

        # Filter out training-specific parameters
        transformer_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in ['lr', 'optimizer', 'weight_decay', 'lr_scheduler', 'warmup_epochs']
        }
        self.model = ViT(transformer_kwargs)

        # Add temporal processing
        if sequence_length > 1:
            # Get output dimension from a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 512, 512, 6)
                dummy_output = self.model(dummy_input)
                output_dim = dummy_output.shape[-1]

            self.temporal_proj = nn.Linear(sequence_length, 1)
            self.feature_proj = nn.Linear(output_dim, output_dim)
        else:
            self.temporal_proj = None
            self.feature_proj = None

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Handle both sequence and single-frame cases
        if return_attention:
            # For attention visualization, always use single frame
            if self.sequence_length > 1 and x.dim() == 5:
                # Get first frame from sequence
                attn_frame = x[:, 0] if x.size(0) > 1 else x[0:1, 0]
                # Get attention weights
                _, attn = self.model(attn_frame, return_attention=True)
            else:
                # Single frame case
                _, attn = self.model(x, return_attention=True)

            # Get predictions for full input
            preds = self._forward_predictions(x)
            return preds, attn
        else:
            return self._forward_predictions(x)

    def _forward_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method to process predictions"""
        if self.sequence_length > 1 and x.dim() == 5:
            B, T, H, W, C = x.shape
            x = x.reshape(B*T, H, W, C)
            preds = self.model(x, return_attention=False)
            preds = preds.reshape(B, T, -1)

            if self.feature_proj is not None:
                preds = self.feature_proj(preds)
            return self.temporal_proj(preds.transpose(1, 2)).squeeze(-1)
        else:
            return self.model(x, return_attention=False)



    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def _calculate_loss(self, batch, mode="train"):
        imgs, sxr = batch
        preds = self(imgs)

        # Ensure predictions and targets have same shape
        if isinstance(preds, tuple):  # When return_attention=True
            preds = preds[0]  # Take only the predictions

        if preds.dim() > 1 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        loss = F.huber_loss(preds, sxr)
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

class SequenceFastViTWrapper(FastViTFlaringModel):
    """Extended FastViT for sequence input"""
    def __init__(self, *args, sequence_length: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        if sequence_length > 1:
            # Add temporal attention
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.hparams.embed_dim,
                num_heads=self.hparams.num_heads,
                batch_first=True
            )
            self.temporal_norm = nn.LayerNorm(self.hparams.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, H, W, C]
        if self.sequence_length > 1:
            B, T, H, W, C = x.shape
            # Process each frame
            x = x.reshape(B*T, H, W, C)
            x = super().forward(x)
            x = x.reshape(B, T, -1)

            # Temporal attention
            x = self.temporal_norm(x)
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)  # Aggregate temporal dimension
            return self.regression_head(x).squeeze(-1)
        else:
            return super().forward(x)

# --------------------------
# Training Script
# --------------------------

def train():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML')
    args = parser.parse_args()

    # Load and resolve config
    with open(args.config, 'r') as stream:
        config_data = yaml.safe_load(stream)
    config_data = resolve_config_variables(config_data)

    # Initialize data module
    use_sequences = config_data['training']['sequence']['enabled']
    sequence_length = config_data['training']['sequence']['length'] if use_sequences else 1

    if use_sequences:
        print("Training with Sequence Data")
        data_loader = AIA_GOESSequenceDataModule(
            aia_train_dir=os.path.join(config_data['data']['paths']['aia'], "train"),
            aia_val_dir=os.path.join(config_data['data']['paths']['aia'], "val"),
            aia_test_dir=os.path.join(config_data['data']['paths']['aia'], "test"),
            sxr_train_dir=os.path.join(config_data['data']['paths']['sxr'], "train"),
            sxr_val_dir=os.path.join(config_data['data']['paths']['sxr'], "val"),
            sxr_test_dir=os.path.join(config_data['data']['paths']['sxr'], "test"),
            batch_size=config_data['training']['batch_size'],
            num_workers=os.cpu_count(),
            sxr_norm=np.load(config_data['data']['paths']['sxr_norm']),
            sequence_length=sequence_length,
            stride=config_data['training']['sequence']['stride']
        )
    else:
        print("Training one to one regressor")
        data_loader = AIA_GOESDataModule(
            aia_train_dir=os.path.join(config_data['data']['paths']['aia'], "train"),
            aia_val_dir=os.path.join(config_data['data']['paths']['aia'], "val"),
            aia_test_dir=os.path.join(config_data['data']['paths']['aia'], "test"),
            sxr_train_dir=os.path.join(config_data['data']['paths']['sxr'], "train"),
            sxr_val_dir=os.path.join(config_data['data']['paths']['sxr'], "val"),
            sxr_test_dir=os.path.join(config_data['data']['paths']['sxr'], "test"),
            batch_size=config_data['training']['batch_size'],
            num_workers=os.cpu_count(),
            sxr_norm=np.load(config_data['data']['paths']['sxr_norm'])
        )
    data_loader.setup()

    # Initialize model
    model_type = config_data['model']['type']
    if model_type == 'ViT':
        vit_kwargs = {
            **config_data['ViT']['architecture'],
            **config_data['ViT']['training']
        }
        if use_sequences:
            model = SequenceViTWrapper(
                model_kwargs=vit_kwargs,
                sequence_length=sequence_length
            )
        else:
            model = ViT(vit_kwargs)
    elif model_type == 'FastViT':
        fastvit_kwargs = {
            **config_data['FastViT']['architecture'],
            **config_data['FastViT']['training']
        }
        model = SequenceFastViTWrapper(
            d_input=6,
            d_output=1,
            eve_norm=tuple(np.load(config_data['data']['paths']['sxr_norm'])),
            image_size=512,
            patch_size=32,
            sequence_length=sequence_length,
            **fastvit_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize logger and callbacks
    wandb_tags = config_data['wandb']['tags'].copy()
    wandb_tags[3] = model_type.lower()  # Update model type tag
    wandb_tags[4] = "sequence" if use_sequences else "single-frame"  # Update sequence tag

    wandb_logger = WandbLogger(
        entity=config_data['wandb']['entity'],
        project=config_data['wandb']['project'],
        name=f"{model_type.lower()}-seq{sequence_length}" if use_sequences else f"{model_type.lower()}-single",
        tags=wandb_tags,
        config=config_data
    )

    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=config_data['data']['paths']['checkpoints'],
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            filename=f"{model_type}-{{epoch:02d}}-{{val_loss:.4f}}"
        ),
        ImagePredictionLogger_SXR(
            data_samples=[data_loader.val_ds[i] for i in range(0, min(4, len(data_loader.val_ds)))],
            sxr_norm=np.load(config_data['data']['paths']['sxr_norm'])
        )
    ]

    if model_type == 'ViT':
        callbacks.append(AttentionMapCallback())

    # Initialize trainer
    trainer = Trainer(
        default_root_dir=config_data['data']['paths']['checkpoints'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config_data['training']['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10
    )

    # Train and save final model
    trainer.fit(model, data_loader)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        config_data['data']['paths']['checkpoints'],
        f"{model_type}-final-{timestamp}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to: {model_path}")

    wandb.finish()

if __name__ == "__main__":
    train()