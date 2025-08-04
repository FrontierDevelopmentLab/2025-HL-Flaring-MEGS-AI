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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.nn import MSELoss
import torch.nn.functional as F
from torch import optim
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as T

# Get the absolute path to the forecasting directory
current_dir = Path(__file__).parent
forecasting_dir = current_dir.parent
sys.path.insert(0, str(forecasting_dir))

# Import models
from models.vision_transformer_custom import ViT
from models.FastSpectralNet import FastViTFlaringModel

# Import data modules
from data_loaders.SDOAIA_dataloader import AIA_GOESSequenceDataModule

# Import callbacks
from training.callback import ImagePredictionLogger_SXR, AttentionMapCallback

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

class Seq2SeqViTWrapper(LightningModule):
    """Wrapper for ViT to handle sequence-to-sequence regression"""
    def __init__(self, model_kwargs: Dict[str, Any],
                 input_sequence_length: int = 6,
                 output_sequence_length: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        transformer_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in ['lr', 'optimizer', 'weight_decay', 'lr_scheduler', 'warmup_epochs', 'eve_norm']
        }
        self.model = ViT(transformer_kwargs)

        with torch.no_grad():
            num_channels = transformer_kwargs.get('num_channels', 6)
            dummy_input = torch.randn(1, num_channels, 512, 512)
           # print(f"Seq2SeqViTWrapper init: Dummy input shape={dummy_input.shape}")
            dummy_output = self.model(dummy_input, return_attention=False)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]
            output_dim = dummy_output.shape[-1]
        self.temporal_decoder = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_sequence_length)
        )
        #print(f"Initialized ViT with output_dim={output_dim}, temporal_decoder output={output_sequence_length}")

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_frame = x[:, 0] if x.size(0) > 1 else x[0:1, 0]
            _, attn = self.model(attn_frame, return_attention=True)
            preds = self._forward_predictions(x)
            return preds, attn
        return self._forward_predictions(x)

    def _forward_predictions(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.shape[1] != self.input_sequence_length or x.shape[2] != 6:
            raise ValueError(f"Expected input shape [B, {self.input_sequence_length}, 6, H, W], got {x.shape}")
        B, T_in, C, H, W = x.shape
       # print(f"Forward input shape: [B={B}, T={T_in}, C={C}, H={H}, W={W}]")
        x = x.reshape(B*T_in, C, H, W)
        features = self.model(x, return_attention=False)
        features = features.reshape(B, T_in, -1)
        features = features.mean(dim=1)
        preds = self.temporal_decoder(features)
       # print(f"Output shape: {preds.shape}")
        return preds

    def denormalize_sxr(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = torch.tensor(self.hparams.model_kwargs.get('eve_norm', (0.0, 1.0)), device=x.device)
        return torch.pow(10, x * std + mean) - 1e-8

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.model_kwargs.get('lr', 1e-4),
                                weight_decay=self.hparams.model_kwargs.get('weight_decay', 0.01))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.model_kwargs.get('warmup_epochs', 5),
            eta_min=1e-6
        )
       # print(f"Configured optimizer: AdamW, lr={self.hparams.model_kwargs.get('lr', 1e-4)}")
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, targets = batch
        #print(f"{mode.capitalize()} batch shapes: AIA={imgs.shape}, SXR={targets.shape}")
        preds = self(imgs)
        if isinstance(preds, tuple):
            preds = preds[0]
        loss = F.huber_loss(preds, targets)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        for t in range(self.output_sequence_length):
            self.log(f"{mode}_loss_t{t}", F.huber_loss(preds[:, t], targets[:, t]))
            self.log(f"{mode}_pred_t{t}", self.denormalize_sxr(preds[:, t]).mean())
       # print(f"{mode.capitalize()} loss: {loss.item():.4f}")
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config.yaml', required=True, help='Path to config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config_data = yaml.safe_load(stream)
    config_data = resolve_config_variables(config_data)
    #print("Loaded config:", config_data)

    for path in [config_data['data']['paths']['aia'], config_data['data']['paths']['sxr'], config_data['data']['paths']['sxr_norm']]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
    os.makedirs(config_data['data']['paths']['checkpoints'], exist_ok=True)
    print(f"Checkpoint directory: {config_data['data']['paths']['checkpoints']}")

    # Load and validate sxr_norm
    sxr_norm_path = config_data['data']['paths']['sxr_norm']
    sxr_norm = np.load(sxr_norm_path, allow_pickle=True)
   # print(f"Loaded sxr_norm from {sxr_norm_path}: type={type(sxr_norm)}, content={sxr_norm}")
    if isinstance(sxr_norm, (np.ndarray, list)) and len(sxr_norm) == 2:
        sxr_norm = tuple(float(x) for x in sxr_norm)
        print(f"Converted sxr_norm to tuple: {sxr_norm}")
    if not isinstance(sxr_norm, tuple) or len(sxr_norm) != 2:
        raise ValueError(f"sxr_norm must be a tuple of (mean, std), got type={type(sxr_norm)}, content={sxr_norm}")
    if not all(np.isfinite(sxr_norm)):
        raise ValueError(f"sxr_norm contains non-finite values: {sxr_norm}")
   # print(f"SXR normalization: mean={sxr_norm[0]:.4f}, std={sxr_norm[1]:.4f}")

    input_seq_length = config_data['training']['sequence']['input_length']
    output_seq_length = config_data['training']['sequence']['output_length']
    stride = config_data['training']['sequence']['stride']
   # print(f"Training Seq2Seq regression: input_length={input_seq_length}, output_length={output_seq_length}, stride={stride}")

    data_loader = AIA_GOESSequenceDataModule(
        aia_train_dir=os.path.join(config_data['data']['paths']['aia'], "train"),
        aia_val_dir=os.path.join(config_data['data']['paths']['aia'], "val"),
        aia_test_dir=os.path.join(config_data['data']['paths']['aia'], "test"),
        sxr_train_dir=os.path.join(config_data['data']['paths']['sxr'], "train"),
        sxr_val_dir=os.path.join(config_data['data']['paths']['sxr'], "val"),
        sxr_test_dir=os.path.join(config_data['data']['paths']['sxr'], "test"),
        sxr_norm=sxr_norm,
        sequence_length=input_seq_length,
        stride=stride,
        batch_size=config_data['training']['batch_size'],
        num_workers=os.cpu_count(),
        train_transforms=T.Compose(config_data['data'].get('transforms', {}).get('train', [])),
        val_transforms=T.Compose(config_data['data'].get('transforms', {}).get('val', []))
    )
    data_loader.setup()



    model_type = config_data['model']['type']
    if model_type == 'ViT':
        vit_kwargs = {
            **config_data['ViT']['architecture'],
            **config_data['ViT']['training'],
            'eve_norm': sxr_norm
        }
        model = Seq2SeqViTWrapper(
            model_kwargs=vit_kwargs,
            input_sequence_length=input_seq_length,
            output_sequence_length=output_seq_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    print(f"Initialized {model_type} model")

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
    #print(f"Initialized WandbLogger: project={config_data['wandb']['project']}")

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
            sxr_norm=sxr_norm
        ),
        AttentionMapCallback(
            log_every_n_epochs=1,
            num_samples=4,
            save_dir="attention_maps"
        ),
        EarlyStopping(monitor='val_loss', patience=5, mode='min')
    ]
    print("Initialized callbacks")

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
        accumulate_grad_batches=1,
        check_val_every_n_epoch=1
    )
    print("Initialized Trainer")

    print("\n===== Memory Status Before Training =====")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Allocated: {torch.cuda.memory_allocated(i)/1e9:.2f}GB")
        print(f"Cached:    {torch.cuda.memory_reserved(i)/1e9:.2f}GB")

    torch.cuda.empty_cache()

    # Test a single batch for debugging
    train_loader = data_loader.train_dataloader()
    batch = next(iter(train_loader))
   # print(f"Sample batch shapes: AIA={batch[0].shape}, SXR={batch[1].shape}")
    with torch.no_grad():
        model.eval()
        model.cuda()  # Move model to CUDA
        input_tensor = batch[0].cuda()
       # print(f"Input tensor device: {input_tensor.device}")
        print(f"Model device: {next(model.parameters()).device}")
        output = model(input_tensor)
       # print(f"Sample output shape: {output.shape}")
    model.train()
    model.to('cpu')  # Move back to CPU to avoid interfering with Trainer

    try:
        trainer.fit(model, data_loader)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n!!! OOM Error Occurred !!!")
            print("Try reducing batch_size or accumulate_grad_batches")
            raise
        else:
            raise

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