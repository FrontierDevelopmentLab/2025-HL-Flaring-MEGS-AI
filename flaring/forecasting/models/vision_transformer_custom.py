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

norm = np.load("/mnt/data/ML-Ready/mixed_data/SXR/normalized_sxr.npy")

def unnormalize_sxr(normalized_values, sxr_norm):
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8

class ViT(pl.LightningModule):
    def __init__(self, model_kwargs):
        super().__init__()
        self.lr = model_kwargs['lr']
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('lr', None)
        self.model = VisionTransformer(**filtered_kwargs)
        self.dynamic_loss = SXRRegressionDynamicLoss(window_size=50)

    def forward(self, x, return_attention=True):
        return self.model(x, return_attention=return_attention)

    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01,  # Add weight decay
            betas=(0.9, 0.95)  # Better betas for ViT
        )

        # Option 1: ReduceLROnPlateau (your current approach, fixed)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,  # Increased patience
        #     min_lr=1e-7,  # Set minimum LR
        # )

        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss',
        #         'interval': 'epoch',
        #         'frequency': 1,
        #         'strict': True,
        #         'name': 'learning_rate'  # This helps with logging
        #     }
        # }

        # Option 2: Cosine Annealing with Warmup (recommended)
        # Uncomment this section and comment out the above return statement to use


        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
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
        sxr_un = unnormalize_sxr(sxr, norm)
        preds_squeezed_un = unnormalize_sxr(preds_squeezed, norm)

        # Use regression-focused dynamic loss
        loss, weights = self.dynamic_loss.calculate_loss(
            preds_squeezed, sxr, sxr_un, preds_squeezed_un,
            epoch=self.current_epoch
        )

        # Log performance info
        if mode == "train" and self.global_step % 200 == 0:
            perf_info = self.dynamic_loss.get_performance_info()
            for key, value in perf_info['range_multipliers'].items():
                self.log(f"regression/{key}", value)
            if 'recent_relative_error' in perf_info:
                self.log("regression/relative_error", perf_info['recent_relative_error'])
            if 'recent_bias' in perf_info:
                self.log("regression/bias", perf_info['recent_bias'])
        # Calculate standard metrics (on normalized values)
        mae = F.l1_loss(preds_squeezed, sxr)
        mse = F.mse_loss(preds_squeezed, sxr)

        # Log all metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{mode}_mae", mae, sync_dist=True)
        self.log(f"{mode}_mse", mse, sync_dist=True)
        #self.log(f"{mode}_avg_weight", total_weights.mean(), sync_dist=True)
        #self.log(f"{mode}_max_weight", total_weights.max(), sync_dist=True)
        #self.log(f"{mode}_mx_flare_ratio", (sxr_un >= m_class_threshold).float().mean(), sync_dist=True)  # FIXED: Use unnormalized

        # FIXED: Log current learning rate from optimizer
        if mode == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    # Optional: Add learning rate warmup
    def on_train_epoch_start(self):
        # Warmup for first 5 epochs
        if self.current_epoch < 5:
            warmup_lr = self.lr * (self.current_epoch + 1) / 5
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group['lr'] = warmup_lr



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
        x: Tensor representing the image of shape [B, C, H, W]
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
    """
    Dynamic weighted loss for SXR flux regression from EUV images
    Focuses on prediction accuracy across the full flux range with adaptive emphasis
    """

    def __init__(self, window_size=50):
        # SXR flux ranges for context (but we're predicting continuous values)
        self.quiet_level = 1e-9  # Typical background
        self.c_threshold = 1e-6  # C-class reference
        self.m_threshold = 1e-5  # M-class reference
        self.x_threshold = 1e-4  # X-class reference

        # Performance tracking for different flux ranges
        self.window_size = window_size
        self.quiet_errors = deque(maxlen=window_size)  # < 1e-6
        self.moderate_errors = deque(maxlen=window_size)  # 1e-6 to 1e-5
        self.high_errors = deque(maxlen=window_size)  # 1e-5 to 1e-4
        self.extreme_errors = deque(maxlen=window_size)  # > 1e-4

        # Relative error tracking (more important for regression)
        self.relative_errors = deque(maxlen=window_size)

        # Trend tracking (are we consistently under/over predicting?)
        self.bias_tracking = deque(maxlen=window_size)

        self.epoch = 0

    def calculate_loss(self, preds_squeezed, sxr, sxr_un, preds_squeezed_un, epoch=None):
        """
        Calculate adaptive regression loss for SXR prediction

        Args:
            preds_squeezed: Normalized predictions
            sxr: Normalized targets
            sxr_un: Unnormalized targets (actual SXR flux)
            preds_squeezed_un: Unnormalized predictions (actual SXR flux)
            epoch: Current epoch
        """
        if epoch is not None:
            self.epoch = epoch

        # Base regression loss
        base_loss = F.huber_loss(preds_squeezed, sxr, delta=1.0, reduction='none')

        # Calculate adaptive weights based on flux ranges and recent performance
        adaptive_weights = self._calculate_regression_weights(
            sxr_un, preds_squeezed_un, base_loss
        )

        # Update performance tracking
        self._update_regression_tracking(sxr_un, preds_squeezed_un, base_loss)

        # Apply weights
        weighted_loss = base_loss * adaptive_weights
        loss = weighted_loss.mean()

        return loss, adaptive_weights

    def _calculate_regression_weights(self, sxr_un, preds_squeezed_un, base_loss):
        """Calculate weights for regression focusing on flux magnitude and recent performance"""

        # 1. Flux-magnitude based weighting (logarithmic importance)
        flux_weights = self._get_flux_magnitude_weights(sxr_un)

        # 2. Error-magnitude based weighting (focus on large errors)
        error_weights = self._get_error_magnitude_weights(sxr_un, preds_squeezed_un)

        # 3. Adaptive range-based weighting (based on recent performance per flux range)
        range_weights = self._get_adaptive_range_weights(sxr_un)

        # 4. Bias correction weights (fix systematic under/over prediction)
        bias_weights = self._get_bias_correction_weights(sxr_un, preds_squeezed_un)

        # 5. Curriculum weighting (gradually focus on harder examples)
        curriculum_weights = self._get_curriculum_weights(base_loss)

        # Combine all weights
        total_weights = (flux_weights *
                         error_weights *
                         range_weights *
                         bias_weights *
                         curriculum_weights)

        # Clamp to reasonable range
        total_weights = torch.clamp(total_weights, min=0.5, max=20.0)

        return total_weights

    def _get_flux_magnitude_weights(self, sxr_un):
        """Weight based on flux magnitude - higher flux = higher importance"""

        # Logarithmic weighting: higher flux values get more attention
        # But not too extreme to avoid ignoring quiet periods
        log_flux = torch.log10(sxr_un + 1e-10)  # Add small constant to avoid log(0)

        # Normalize to reasonable range
        # quiet (~1e-9) → log ≈ -9 → weight ≈ 1.0
        # C-class (1e-6) → log ≈ -6 → weight ≈ 1.3
        # M-class (1e-5) → log ≈ -5 → weight ≈ 1.4
        # X-class (1e-4) → log ≈ -4 → weight ≈ 1.5
        normalized_log = (log_flux + 9) / 5  # Shift and scale
        flux_weights = 1.0 + 0.5 * torch.clamp(normalized_log, 0, 1)

        return flux_weights

    def _get_error_magnitude_weights(self, sxr_un, preds_squeezed_un):
        """Weight based on prediction error magnitude - larger errors get more attention"""

        # Relative error is more meaningful for flux prediction
        relative_error = torch.abs(sxr_un - preds_squeezed_un) / (sxr_un + 1e-10)

        # Focus more on samples with large relative errors
        error_percentile_75 = torch.quantile(relative_error, 0.75)
        error_percentile_90 = torch.quantile(relative_error, 0.90)

        error_weights = torch.ones_like(relative_error)
        error_weights = torch.where(relative_error > error_percentile_75, 1.5, error_weights)
        error_weights = torch.where(relative_error > error_percentile_90, 2.0, error_weights)

        return error_weights

    def _get_adaptive_range_weights(self, sxr_un):
        """Adaptive weights based on recent performance in different flux ranges"""

        range_weights = torch.ones_like(sxr_un)

        # Get performance multipliers for different ranges
        quiet_mult = self._get_range_performance_multiplier(self.quiet_errors)
        moderate_mult = self._get_range_performance_multiplier(self.moderate_errors)
        high_mult = self._get_range_performance_multiplier(self.high_errors)
        extreme_mult = self._get_range_performance_multiplier(self.extreme_errors)

        # Apply multipliers based on flux range
        range_weights = torch.where(sxr_un < self.c_threshold, quiet_mult, range_weights)
        range_weights = torch.where((sxr_un >= self.c_threshold) & (sxr_un < self.m_threshold),
                                    moderate_mult, range_weights)
        range_weights = torch.where((sxr_un >= self.m_threshold) & (sxr_un < self.x_threshold),
                                    high_mult, range_weights)
        range_weights = torch.where(sxr_un >= self.x_threshold, extreme_mult, range_weights)

        return range_weights

    def _get_range_performance_multiplier(self, error_history):
        """Get multiplier based on recent performance in a flux range"""
        if len(error_history) < 5:
            return 1.0

        recent_avg_error = np.mean(list(error_history))

        # If recent errors are high, increase weight for this range
        if recent_avg_error > 2.0:  # High error
            return 2.0
        elif recent_avg_error > 1.0:  # Moderate error
            return 1.5
        else:
            return 1.0  # Good performance

    def _get_bias_correction_weights(self, sxr_un, preds_squeezed_un):
        """Correct for systematic bias (consistent under/over prediction)"""

        bias_weights = torch.ones_like(sxr_un)

        if len(self.bias_tracking) < 10:
            return bias_weights

        recent_bias = np.mean(list(self.bias_tracking))

        # If we're systematically under-predicting, boost weight when actual > pred
        if recent_bias < -0.2:  # Consistent under-prediction
            under_pred_mask = preds_squeezed_un < sxr_un
            bias_weights = torch.where(under_pred_mask, 1.5, bias_weights)

        # If we're systematically over-predicting, boost weight when actual < pred
        elif recent_bias > 0.2:  # Consistent over-prediction
            over_pred_mask = preds_squeezed_un > sxr_un
            bias_weights = torch.where(over_pred_mask, 1.5, bias_weights)

        return bias_weights

    def _get_curriculum_weights(self, base_loss):
        """Curriculum learning for regression - gradually focus on harder examples"""

        if self.epoch < 10:
            # Early training: focus on easier examples (smaller losses)
            curriculum_factor = 0.5
        elif self.epoch < 30:
            # Mid training: gradual transition
            curriculum_factor = 0.5 + 0.5 * (self.epoch - 10) / 20
        else:
            # Late training: focus on all examples equally
            curriculum_factor = 1.0

        # Harder examples = higher loss values
        loss_difficulty = (base_loss - base_loss.min()) / (base_loss.max() - base_loss.min() + 1e-8)
        curriculum_weights = 1.0 + curriculum_factor * loss_difficulty

        return curriculum_weights

    def _update_regression_tracking(self, sxr_un, preds_squeezed_un, base_loss):
        """Update performance tracking for different flux ranges"""

        # Track errors by flux range
        quiet_mask = sxr_un < self.c_threshold
        moderate_mask = (sxr_un >= self.c_threshold) & (sxr_un < self.m_threshold)
        high_mask = (sxr_un >= self.m_threshold) & (sxr_un < self.x_threshold)
        extreme_mask = sxr_un >= self.x_threshold

        # Calculate relative errors for each range
        relative_errors = torch.abs(sxr_un - preds_squeezed_un) / (sxr_un + 1e-10)

        if quiet_mask.any():
            self.quiet_errors.append(relative_errors[quiet_mask].mean().item())
        if moderate_mask.any():
            self.moderate_errors.append(relative_errors[moderate_mask].mean().item())
        if high_mask.any():
            self.high_errors.append(relative_errors[high_mask].mean().item())
        if extreme_mask.any():
            self.extreme_errors.append(relative_errors[extreme_mask].mean().item())

        # Track overall relative error
        self.relative_errors.append(relative_errors.mean().item())

        # Track prediction bias (positive = over-prediction, negative = under-prediction)
        bias = torch.log10(preds_squeezed_un + 1e-10) - torch.log10(sxr_un + 1e-10)
        self.bias_tracking.append(bias.mean().item())

    def get_performance_info(self):
        """Get current performance info for logging"""
        info = {
            'epoch': self.epoch,
            'range_multipliers': {
                'quiet_mult': self._get_range_performance_multiplier(self.quiet_errors),
                'moderate_mult': self._get_range_performance_multiplier(self.moderate_errors),
                'high_mult': self._get_range_performance_multiplier(self.high_errors),
                'extreme_mult': self._get_range_performance_multiplier(self.extreme_errors)
            }
        }

        if len(self.relative_errors) > 0:
            info['recent_relative_error'] = np.mean(list(self.relative_errors))
        if len(self.bias_tracking) > 0:
            info['recent_bias'] = np.mean(list(self.bias_tracking))

        return info