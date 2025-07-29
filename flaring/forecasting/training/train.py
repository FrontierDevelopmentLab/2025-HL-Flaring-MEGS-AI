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
    AIA_GOESSequenceDataModule
)

# Import callbacks
from training.callback import ImagePredictionLogger_SXR, AttentionMapCallback

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
# Model Wrappers (Modified for Seq2Seq)
# --------------------------

class Seq2SeqViTWrapper(LightningModule):
    """Wrapper for ViT to handle sequence-to-sequence prediction"""
    def __init__(self, model_kwargs: Dict[str, Any],
                 input_sequence_length: int = 6,
                 output_sequence_length: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Filter out training-specific parameters
        transformer_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in ['lr', 'optimizer', 'weight_decay', 'lr_scheduler', 'warmup_epochs']
        }
        self.model = ViT(transformer_kwargs)

        # Add temporal processing for sequence output
        # Get output dimension from a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 512, 512, 6)
            dummy_output = self.model(dummy_input, return_attention=False)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]
            output_dim = dummy_output.shape[-1]

        # Temporal decoder for sequence output
        self.temporal_decoder = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_sequence_length)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            # For attention visualization, use first frame
            if x.dim() == 5:  # [B, T, H, W, C]
                attn_frame = x[:, 0] if x.size(0) > 1 else x[0:1, 0]
                _, attn = self.model(attn_frame, return_attention=True)
                preds = self._forward_predictions(x)
                return preds, attn
            else:
                _, attn = self.model(x, return_attention=True)
                preds = self._forward_predictions(x)
                return preds, attn
        else:
            return self._forward_predictions(x)

    def _forward_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Process input sequence and return output sequence"""
        B, T_in, H, W, C = x.shape

        # Process each frame independently
        x = x.reshape(B*T_in, H, W, C)
        features = self.model(x, return_attention=False)
        features = features.reshape(B, T_in, -1)

        # Average features across input sequence
        features = features.mean(dim=1)  # [B, D]

        # Decode to output sequence
        return self.temporal_decoder(features)  # [B, T_out]

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.model_kwargs.get('lr', 1e-4),
                                weight_decay=self.hparams.model_kwargs.get('weight_decay', 0.01))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.model_kwargs.get('warmup_epochs', 5),
            eta_min=1e-6
        )
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, targets = batch  # targets shape: [B, T_out]
        preds = self(imgs)  # preds shape: [B, T_out]

        # Ensure predictions and targets have same shape
        if isinstance(preds, tuple):  # When return_attention=True
            preds = preds[0]

        # Calculate loss for each timestep
        loss = F.huber_loss(preds, targets)

        # Log overall loss and per-timestep losses
        self.log(f"{mode}_loss", loss)
        for t in range(self.output_sequence_length):
            self.log(f"{mode}_loss_t{t}", F.huber_loss(preds[:, t], targets[:, t]))

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

class Seq2SeqFastViTWrapper(FastViTFlaringModel):
    """Extended FastViT for sequence-to-sequence prediction"""
    def __init__(self, *args,
                 input_sequence_length: int = 6,
                 output_sequence_length: int = 6,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Replace final regression head with sequence decoder
        if hasattr(self, 'regression_head'):
            self.regression_head = nn.Sequential(
                nn.Linear(self.hparams.embed_dim, self.hparams.embed_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hparams.embed_dim * 2, output_sequence_length)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T_in, H, W, C]
        B, T_in, H, W, C = x.shape
        x = x.reshape(B*T_in, H, W, C)
        x = super().forward_features(x)  # Get features before regression head
        x = x.reshape(B, T_in, -1)

        # Aggregate temporal information
        x = x.mean(dim=1)  # [B, D]

        # Decode to output sequence
        return self.regression_head(x)  # [B, T_out]

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

    # Initialize data module for seq2seq
    input_seq_length = config_data['training']['sequence']['input_length']
    output_seq_length = config_data['training']['sequence']['output_length']
    stride = config_data['training']['sequence']['stride']

    print(f"Training Seq2Seq model with input length {input_seq_length}, output length {output_seq_length}, stride {stride}")

    data_loader = AIA_GOESSequenceDataModule(
        aia_train_dir=os.path.join(config_data['data']['paths']['aia'], "train"),
        aia_val_dir=os.path.join(config_data['data']['paths']['aia'], "val"),
        aia_test_dir=os.path.join(config_data['data']['paths']['aia'], "test"),
        sxr_train_dir=os.path.join(config_data['data']['paths']['sxr'], "train"),
        sxr_val_dir=os.path.join(config_data['data']['paths']['sxr'], "val"),
        sxr_test_dir=os.path.join(config_data['data']['paths']['sxr'], "test"),
        sxr_norm=np.load(config_data['data']['paths']['sxr_norm']),
        sequence_length=input_seq_length,  # Using input length for sequence
        stride=stride,
        batch_size=config_data['training']['batch_size'],
        num_workers=os.cpu_count()
    )
    data_loader.setup()

    # Initialize model
    model_type = config_data['model']['type']
    if model_type == 'ViT':
        vit_kwargs = {
            **config_data['ViT']['architecture'],
            **config_data['ViT']['training']
        }
        model = Seq2SeqViTWrapper(
            model_kwargs=vit_kwargs,
            input_sequence_length=input_seq_length,
            output_sequence_length=output_seq_length
        )
    elif model_type == 'FastViT':
        fastvit_kwargs = {
            **config_data['FastViT']['architecture'],
            **config_data['FastViT']['training']
        }
        model = Seq2SeqFastViTWrapper(
            d_input=6,
            d_output=output_seq_length,
            eve_norm=tuple(np.load(config_data['data']['paths']['sxr_norm'])),
            image_size=512,
            patch_size=32,
            input_sequence_length=input_seq_length,
            output_sequence_length=output_seq_length,
            **fastvit_kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize logger and callbacks
    wandb_tags = config_data['wandb']['tags'].copy()
    wandb_tags[3] = model_type.lower()
    wandb_tags[4] = f"seq2seq_{input_seq_length}to{output_seq_length}"

    wandb_logger = WandbLogger(
        entity=config_data['wandb']['entity'],
        project=config_data['wandb']['project'],
        name=f"{model_type.lower()}-seq2seq-{input_seq_length}to{output_seq_length}",
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
            filename=f"{model_type}-seq2seq-{{epoch:02d}}-{{val_loss:.4f}}"
        ),
        ImagePredictionLogger_SXR(
            data_samples=[data_loader.val_ds[i] for i in range(0, min(4, len(data_loader.val_ds)))],
            sxr_norm=np.load(config_data['data']['paths']['sxr_norm']),
            log_every_n_epochs=1

        )
    ]

    if model_type == 'ViT':
        callbacks.append(AttentionMapCallback(
            log_every_n_epochs=1,
            num_samples=4,

        ))

    trainer = Trainer(
        default_root_dir=config_data['data']['paths']['checkpoints'],
        accelerator="gpu",
        devices=4,
        strategy="ddp_find_unused_parameters_false",
        precision="bf16-mixed",
        max_epochs=config_data['training']['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )

    # Memory debugging
    print("\n===== Memory Status Before Training =====")
    print(f"PyTorch sees {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
        print(f"Cached:    {torch.cuda.memory_reserved(i)/1e9:.2f}GB\n")

    torch.cuda.empty_cache()

    try:
        trainer.fit(model, data_loader)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n!!! OOM Error Occurred !!!")
            raise
        else:
            raise

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        config_data['data']['paths']['checkpoints'],
        f"{model_type}-seq2seq-final-{timestamp}.pt"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Saved final model to: {model_path}")

    wandb.finish()

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_DEBUG"] = "WARN"
    train()