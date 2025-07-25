import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
import pytorch_lightning as pl
from typing import Tuple, Dict, Any, Optional


class ViT(pl.LightningModule):
    def __init__(self, model_kwargs: Dict[str, Any]):
        super().__init__()

        # Set default values for training parameters
        self.lr = model_kwargs.pop('lr', 1e-4)  # Default learning rate if not specified
        self.optimizer_name = model_kwargs.pop('optimizer', 'adam')
        self.weight_decay = model_kwargs.pop('weight_decay', 0.0)
        self.lr_scheduler = model_kwargs.pop('lr_scheduler', None)
        self.warmup_epochs = model_kwargs.pop('warmup_epochs', 0)

        # Save all hyperparameters (including the remaining model_kwargs)
        self.save_hyperparameters()

        # Initialize the VisionTransformer with remaining kwargs
        self.model = VisionTransformer(**model_kwargs)

    def forward(self, x: torch.Tensor, return_attention: bool = True) -> torch.Tensor:
        return self.model(x, return_attention=return_attention)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Create optimizer based on config
        if self.optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        else:  # Default to Adam
            optimizer = optim.Adam(self.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)

        # Configure learning rate scheduler
        if self.lr_scheduler == 'reduce_on_plateau':
            scheduler = {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=3
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif self.lr_scheduler == 'cosine':
            scheduler = {
                'scheduler': optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.max_epochs,
                    eta_min=self.lr/10
                ),
                'interval': 'epoch'
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer

    def _calculate_loss(self, batch: Tuple[torch.Tensor, torch.Tensor], mode: str = "train") -> torch.Tensor:
        imgs, sxr = batch
        preds = self.model(imgs)
        loss = F.huber_loss(torch.squeeze(preds), sxr)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log('learning_rate', self.lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._calculate_loss(batch, mode="test")


class VisionTransformer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_channels: int,
            num_heads: int,
            num_layers: int,
            num_classes: int,
            patch_size: int,
            num_patches: int,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)
        self.transformer_blocks = nn.ModuleList([
            AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :T + 1]

        # Apply Transformer blocks
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [T+1, B, embed_dim]

        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = block(x)

        # Perform prediction
        cls = x[0]

        if return_attention:
            # Stack attention weights: [n_layers, batch_size, num_heads, seq_len, seq_len]
            attention_weights = torch.stack(attention_weights)
            return self.mlp_head(cls), attention_weights

        return self.mlp_head(cls)


class AttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False
    ) -> torch.Tensor:
        inp_x = self.layer_norm_1(x)
        if return_attention:
            attn_output, attn_weights = self.attn(inp_x, inp_x, inp_x, average_attn_weights=False)
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x, attn_weights
        else:
            attn_output = self.attn(inp_x, inp_x, inp_x)[0]
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x


def img_to_patch(
        x: torch.Tensor,
        patch_size: int,
        flatten_channels: bool = True
) -> torch.Tensor:
    x = x.permute(0, 3, 1, 2)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x