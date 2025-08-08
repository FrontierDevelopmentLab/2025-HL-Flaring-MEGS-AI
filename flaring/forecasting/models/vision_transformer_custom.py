from collections import deque

import math
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

#norm = np.load("/mnt/data/ML-Ready_clean/mixed_data/SXR/normalized_sxr.npy")

def unnormalize_sxr(normalized_values, sxr_norm):
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8

class ViT(pl.LightningModule):
    def __init__(self, model_kwargs, sxr_norm):
        super().__init__()
        self.lr = model_kwargs['lr']
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('lr', None)
        self.model = VisionTransformer(**filtered_kwargs)
        self.adaptive_loss = SXRRegressionDynamicLoss(window_size=500)
        self.sxr_norm = sxr_norm

    def forward(self, x, return_attention=True):
        return self.model(x, return_attention=return_attention)

    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the cycle length after each restart
            eta_min=1e-7  # Minimum learning rate
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'learning_rate'
            }
        }

    # M/X Class Flare Detection Optimized Weights

    def _calculate_loss(self, batch, mode="train"):
        imgs, sxr = batch
        preds = self.model(imgs)
        preds_squeezed = torch.squeeze(preds)

        # Unnormalize
        sxr_un = unnormalize_sxr(sxr, self.sxr_norm)
        preds_squeezed_un = unnormalize_sxr(preds_squeezed, self.sxr_norm)

        # Use adaptive rare event loss
        loss, weights = self.adaptive_loss.calculate_loss(
            preds_squeezed, sxr, sxr_un, preds_squeezed_un
        )

        # Log adaptation info
        if mode == "train" and self.global_step % 200 == 0:
            multipliers = self.adaptive_loss.get_current_multipliers()
            for key, value in multipliers.items():
                self.log(f"adaptive/{key}", value)

            self.log("adaptive/avg_weight", weights.mean())
            self.log("adaptive/max_weight", weights.max())
        if mode == "val":
            multipliers = self.adaptive_loss.get_current_multipliers()
            for key, value in multipliers.items():
                self.log(f"val/{key}", value)
            self.log("val/avg_weight", weights.mean())
            self.log("val/max_weight", weights.max())

        # FIXED: Log current learning rate from optimizer
        if mode == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if mode == "val":
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")




class VisionTransformer(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_channels,
            num_heads,
            num_layers,
            num_classes,
            patch_size,
            num_patches,
            dropout=0.0,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding

        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size ** 2), embed_dim)

        self.transformer_blocks = nn.ModuleList([
            AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

    def forward(self, x, return_attention=False):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

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

        # Perform classification prediction
        cls = x[0]  # Take CLS token
        out = self.mlp_head(cls)

        if return_attention:
            return out, attention_weights
        else:
            return out


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network

        """
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

    def forward(self, x, return_attention=False):
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


def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Args:
        x: Tensor representing the image of shape [B, H, W, C]
        patch_size: Number of pixels per dimension of the patches (integer)
        flatten_channels: If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    x = x.permute(0, 3, 1, 2)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x


class SXRRegressionDynamicLoss:
    def __init__(self, window_size=500):
        self.c_threshold = 1e-6
        self.m_threshold = 1e-5
        self.x_threshold = 1e-4

        self.window_size = window_size
        self.quiet_errors = deque(maxlen=window_size)
        self.c_errors = deque(maxlen=window_size)
        self.m_errors = deque(maxlen=window_size)
        self.x_errors = deque(maxlen=window_size)

        self.base_weights = {
            'quiet': 1.0,
            'c_class': 2.0,
            'm_class': 5.0,
            'x_class': 10.0
        }

    def calculate_loss(self, preds_squeezed, sxr, sxr_un, preds_squeezed_un):
        base_loss = F.huber_loss(preds_squeezed, sxr, delta=1.0, reduction='none')
        weights = self._get_adaptive_weights(sxr_un, preds_squeezed_un, base_loss)
        self._update_tracking(sxr_un, preds_squeezed_un, base_loss)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        return loss, weights

    def _get_adaptive_weights(self, sxr_un, preds_squeezed_un, base_loss):
        device = sxr_un.device

        # Get continuous multipliers per class with custom params
        quiet_mult = self._get_performance_multiplier(
            self.quiet_errors, max_multiplier=1.5, min_multiplier=0.5, sensitivity=2.0, sxrclass = 'quiet'
        )
        c_mult = self._get_performance_multiplier(
            self.c_errors, max_multiplier=3.0, min_multiplier=0.7, sensitivity=2.5, sxrclass = 'c_class'
        )
        m_mult = self._get_performance_multiplier(
            self.m_errors, max_multiplier=7.0, min_multiplier=0.8, sensitivity=3.0, sxrclass = 'm_class'
        )
        x_mult = self._get_performance_multiplier(
            self.x_errors, max_multiplier=15.0, min_multiplier=0.9, sensitivity=4.0, sxrclass = 'x_class'
        )

        quiet_weight = self.base_weights['quiet'] * quiet_mult
        c_weight = self.base_weights['c_class'] * c_mult
        m_weight = self.base_weights['m_class'] * m_mult
        x_weight = self.base_weights['x_class'] * x_mult

        weights = torch.ones_like(sxr_un, device=device)
        weights = torch.where(sxr_un < self.c_threshold, quiet_weight, weights)
        weights = torch.where((sxr_un >= self.c_threshold) & (sxr_un < self.m_threshold), c_weight, weights)
        weights = torch.where((sxr_un >= self.m_threshold) & (sxr_un < self.x_threshold), m_weight, weights)
        weights = torch.where(sxr_un >= self.x_threshold, x_weight, weights)

        # Normalize so mean weight ~1.0 (optional, helps stability)
        mean_weight = torch.mean(weights)
        weights = weights / (mean_weight + 1e-8)

        # Clamp extreme weights
        weights = torch.clamp(weights, min=0.1, max=40.0)

        # Save for logging
        self.current_multipliers = {
            'quiet_mult': quiet_mult,
            'c_mult': c_mult,
            'm_mult': m_mult,
            'x_mult': x_mult,
            'quiet_weight': quiet_weight,
            'c_weight': c_weight,
            'm_weight': m_weight,
            'x_weight': x_weight
        }

        return weights

    def _get_performance_multiplier(self, error_history, max_multiplier=10.0, min_multiplier=0.5, sensitivity=3.0, sxrclass='quiet'):
        """Class-dependent performance multiplier"""

        class_params = {
            'quiet': {'min_samples': 200, 'recent_window': 100},
            'c_class': {'min_samples': 200, 'recent_window': 100},
            'm_class': {'min_samples': 200, 'recent_window': 100},
            'x_class': {'min_samples': 200, 'recent_window': 100}
        }

        if len(error_history) < class_params[sxrclass]['min_samples']:
            return 1.0

        recent_window = class_params[sxrclass]['recent_window']
        recent = np.mean(list(error_history)[-recent_window:])
        overall = np.mean(list(error_history))

        # if overall < 1e-10:
        #     return 1.0

        ratio = recent / overall
        multiplier = np.exp(sensitivity * (ratio - 1))
        return np.clip(multiplier, min_multiplier, max_multiplier)

    def _update_tracking(self, sxr_un, preds_squeezed_un, base_loss):
        try:
            sxr_np = sxr_un.detach().cpu().numpy()
            preds_np = preds_squeezed_un.detach().cpu().numpy()
            log_error = np.abs(np.log10(np.maximum(sxr_np, 1e-12)) - np.log10(np.maximum(preds_np, 1e-12)))

            quiet_mask = sxr_np < self.c_threshold
            if quiet_mask.sum() > 0:
                self.quiet_errors.append(float(np.mean(log_error[quiet_mask])))

            c_mask = (sxr_np >= self.c_threshold) & (sxr_np < self.m_threshold)
            if c_mask.sum() > 0:
                self.c_errors.append(float(np.mean(log_error[c_mask])))

            m_mask = (sxr_np >= self.m_threshold) & (sxr_np < self.x_threshold)
            if m_mask.sum() > 0:
                self.m_errors.append(float(np.mean(log_error[m_mask])))

            x_mask = sxr_np >= self.x_threshold
            if x_mask.sum() > 0:
                self.x_errors.append(float(np.mean(log_error[x_mask])))

        except Exception:
            pass

    def get_current_multipliers(self):
        """Get current performance multipliers for logging"""
        return {
            'quiet_mult': self._get_performance_multiplier(self.quiet_errors,sxrclass='quiet'),
            'c_mult': self._get_performance_multiplier(self.c_errors,sxrclass='c_class'),
            'm_mult': self._get_performance_multiplier(self.m_errors,sxrclass='m_class'),
            'x_mult': self._get_performance_multiplier(self.x_errors,sxrclass='x_class'),
            'quiet_count': len(self.quiet_errors),
            'c_count': len(self.c_errors),
            'm_count': len(self.m_errors),
            'x_count': len(self.x_errors),
            'quiet_error': np.mean(self.quiet_errors) if self.quiet_errors else 0.0,
            'c_error': np.mean(self.c_errors) if self.c_errors else 0.0,
            'm_error': np.mean(self.m_errors) if self.m_errors else 0.0,
            'x_error': np.mean(self.x_errors) if self.x_errors else 0.0,
            'quiet_weight': self.current_multipliers['quiet_weight'],
            'c_weight': self.current_multipliers['c_weight'],
            'm_weight': self.current_multipliers['m_weight'],
            'x_weight': self.current_multipliers['x_weight']
        }