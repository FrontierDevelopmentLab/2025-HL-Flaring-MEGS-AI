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

def normalize_sxr(unnormalized_values, sxr_norm, channel_idx=None):
    """Convert from unnormalized to normalized space
    
    Args:
        unnormalized_values: Tensor of unnormalized values
        sxr_norm: Normalization parameters - can be:
            - Single array [mean, std] for backward compatibility
            - Dict with 'a' and 'b' keys for separate normalization
        channel_idx: Index of channel (0 for 'a', 1 for 'b') if using separate normalization
    """
    log_values = torch.log10(unnormalized_values + 1e-8)
    
    if isinstance(sxr_norm, dict):
        # Separate normalization for each channel
        if channel_idx is None:
            raise ValueError("channel_idx must be provided when using separate normalization")
        channel_key = 'a' if channel_idx == 0 else 'b'
        mean = float(sxr_norm[channel_key][0].item())
        std = float(sxr_norm[channel_key][1].item())
    else:
        # Single normalization (backward compatibility)
        mean = float(sxr_norm[0].item())
        std = float(sxr_norm[1].item())
    
    normalized = (log_values - mean) / std
    return normalized

def unnormalize_sxr(normalized_values, sxr_norm, channel_idx=None):
    """Convert from normalized to unnormalized space
    
    Args:
        normalized_values: Tensor of normalized values
        sxr_norm: Normalization parameters - can be:
            - Single array [mean, std] for backward compatibility
            - Dict with 'a' and 'b' keys for separate normalization
        channel_idx: Index of channel (0 for 'a', 1 for 'b') if using separate normalization
    """
    if isinstance(sxr_norm, dict):
        # Separate normalization for each channel
        if channel_idx is None:
            raise ValueError("channel_idx must be provided when using separate normalization")
        channel_key = 'a' if channel_idx == 0 else 'b'
        mean = float(sxr_norm[channel_key][0].item())
        std = float(sxr_norm[channel_key][1].item())
    else:
        # Single normalization (backward compatibility)
        mean = float(sxr_norm[0].item())
        std = float(sxr_norm[1].item())
    
    return 10 ** (normalized_values * std + mean) - 1e-8

class ViTLocal(pl.LightningModule):
    def __init__(self, model_kwargs, sxr_norm, base_weights=None, sxr_channels=['a', 'b']):
        super().__init__()
        self.model_kwargs = model_kwargs
        self.lr = model_kwargs['lr']
        self.sxr_channels = sxr_channels  # ['a', 'b'] or ['b'] or ['a']
        self.save_hyperparameters()
        filtered_kwargs = dict(model_kwargs)
        filtered_kwargs.pop('lr', None)
        filtered_kwargs.pop('num_classes', None)
        # Add num_output_channels to the model kwargs
        filtered_kwargs['num_output_channels'] = len(sxr_channels)
        self.model = VisionTransformerLocal(**filtered_kwargs)
        #Set the base weights based on the number of samples in each class within training data
        self.base_weights = base_weights
        self.adaptive_loss = SXRRegressionDynamicLoss(window_size=15000, base_weights=self.base_weights)
        self.sxr_norm = sxr_norm

    
    def forward(self, x, return_attention=True):
        return self.model(x, self.sxr_norm, return_attention=return_attention)

    def configure_optimizers(self):
        # Use AdamW with weight decay for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.00001,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Restart every 20 epochs
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
        imgs, sxr = batch  # sxr is [B, num_channels] based on sxr_channels config
        raw_preds, raw_patch_contributions = self.model(imgs, self.sxr_norm)  # raw_preds is [B, num_channels]
                
        
        # Unnormalize using appropriate normalization for each channel
        sxr_un = torch.zeros_like(sxr)
        for i in range(len(self.sxr_channels)):
            if isinstance(self.sxr_norm, dict):
                # Use separate normalization for each channel
                sxr_un[:, i] = unnormalize_sxr(sxr[:, i], self.sxr_norm, channel_idx=i)
            else:
                # Use single normalization (backward compatibility)
                sxr_un[:, i] = unnormalize_sxr(sxr[:, i], self.sxr_norm)

        # The model outputs raw SXR values, so we need to normalize them for loss calculation
        # Handle shape: if only 1 channel and raw_preds is [B, 1], keep it that way
        if len(self.sxr_channels) == 1 and len(raw_preds.shape) == 2 and raw_preds.shape[1] == 1:
            # Single channel case: normalize directly
            if isinstance(self.sxr_norm, dict):
                norm_preds = normalize_sxr(raw_preds.squeeze(-1), self.sxr_norm, channel_idx=0).unsqueeze(-1)
            else:
                norm_preds = normalize_sxr(raw_preds.squeeze(-1), self.sxr_norm).unsqueeze(-1)
        else:
            # Multi-channel or different shape: normalize each channel
            norm_preds = torch.zeros_like(raw_preds)
            for i in range(len(self.sxr_channels)):
                if isinstance(self.sxr_norm, dict):
                    # Use separate normalization for each channel
                    norm_preds[:, i] = normalize_sxr(raw_preds[:, i], self.sxr_norm, channel_idx=i)
                else:
                    # Use single normalization (backward compatibility)
                    norm_preds[:, i] = normalize_sxr(raw_preds[:, i], self.sxr_norm)
        
        # Find SXR-B index for adaptive loss calculation
        sxr_b_idx = None
        for i, channel in enumerate(self.sxr_channels):
            if channel == 'b':
                sxr_b_idx = i
                break
        
        if sxr_b_idx is None:
            raise ValueError("SXR-B channel not found in sxr_channels. Must include 'b' channel.")
        
        #print(f"raw_preds: {raw_preds}")
        # Calculate adaptive loss only for SXR-B
        # Note: dataloader guarantees SXR-A at index 0, SXR-B at index 1 (if both present)
        adaptive_loss, adaptive_weights = self.adaptive_loss.calculate_loss(
            norm_preds[:, sxr_b_idx], sxr[:, sxr_b_idx], sxr_un[:, sxr_b_idx]
        )

        if len(self.sxr_channels) == 2:
            # Calculate SXR-A loss (unweighted)
            huber_loss_a = F.huber_loss(norm_preds[:, 0], sxr[:, 0], delta=0.3, reduction='mean')

            # Detach to get current magnitudes (don't affect gradients)
            loss_a_magnitude = huber_loss_a.detach()
            loss_b_magnitude = adaptive_loss.detach()

            # Normalize and apply 30/70 split
            # This ensures the gradient contributions are actually 30/70
            total_magnitude = loss_a_magnitude + loss_b_magnitude + 1e-10  # avoid division by zero

            weighted_a_loss = huber_loss_a * (0.3 / (loss_a_magnitude / total_magnitude + 1e-10))
            adaptive_loss_scaled = adaptive_loss * (0.7 / (loss_b_magnitude / total_magnitude + 1e-10))

            loss = adaptive_loss_scaled + weighted_a_loss
            
            # Debug: log raw values for both channels
            if mode == "train":
                self.log("debug/raw_pred_a_mean", raw_preds[:, 0].mean(), on_step=True, on_epoch=False)
                self.log("debug/raw_pred_b_mean", raw_preds[:, 1].mean(), on_step=True, on_epoch=False)
                self.log("debug/norm_pred_a_mean", norm_preds[:, 0].mean(), on_step=True, on_epoch=False)
                self.log("debug/norm_pred_b_mean", norm_preds[:, 1].mean(), on_step=True, on_epoch=False)
                self.log("debug/target_a_mean", sxr[:, 0].mean(), on_step=True, on_epoch=False)
                self.log("debug/target_b_mean", sxr[:, 1].mean(), on_step=True, on_epoch=False)
                self.log("debug/huber_loss_a_mean", huber_loss_a.mean(), on_step=True, on_epoch=False)
        else:
            weighted_a_loss = 0.0
            loss = adaptive_loss


        loss = adaptive_loss + weighted_a_loss
#log the losses during training
        if mode == "train":
            self.log("weighted_b_loss", adaptive_loss, on_step=True, on_epoch=True)
            self.log("weighted_a_loss", weighted_a_loss, on_step=True, on_epoch=True)
            self.log("total_loss", loss, on_step=True, on_epoch=True)
            #log the weights during training for each flare class
            # Log the current adaptive multipliers/weights for each flare class instead of element-wise weights
            current_multipliers = self.adaptive_loss.current_multipliers
            self.log("adaptive_weights_quiet", current_multipliers['quiet_weight'], on_step=True, on_epoch=True)
            self.log("adaptive_weights_c_class", current_multipliers['c_weight'], on_step=True, on_epoch=True)
            self.log("adaptive_weights_m_class", current_multipliers['m_weight'], on_step=True, on_epoch=True)
            self.log("adaptive_weights_x_class", current_multipliers['x_weight'], on_step=True, on_epoch=True)
        elif mode == "val":
            self.log("val_weighted_b_loss", adaptive_loss, on_step=False, on_epoch=True)
            self.log("val_weighted_a_loss", weighted_a_loss, on_step=False, on_epoch=True)
            self.log("val_total_loss", loss, on_step=False, on_epoch=True)

    

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def apply_wavelength_dropout(self, x, dropout_prob=0.3):
        """Randomly zero out some wavelengths during training"""
        if self.training and torch.rand(1).item() < dropout_prob:
            # x shape: [B, H, W, num_channels]
            num_keep = torch.randint(1, self.model_kwargs['num_channels'], (1,)).item()
            keep_indices = torch.randperm(self.model_kwargs['num_channels'])[:num_keep]

            mask = torch.zeros(self.model_kwargs['num_channels'], device=x.device)
            mask[keep_indices] = 1.0

            x = x * mask.view(1, 1, 1, -1)
        return x


class VisionTransformerLocal(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_channels,
            num_heads,
            num_layers,
            patch_size,
            num_patches,
            dropout,
            num_output_channels=2,

    ):
        """Vision Transformer that outputs flux contributions per patch.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
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
            LocalAttentionBlock(embed_dim, hidden_dim, num_heads, num_patches, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_output_channels))  # Output configurable number of SXR channels
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.grid_h = int(math.sqrt(num_patches))
        self.grid_w = int(math.sqrt(num_patches))
        self.pos_embedding_2d = nn.Parameter(torch.randn(1, self.grid_h, self.grid_w, embed_dim))


    def forward(self, x, sxr_norm, return_attention=False):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add positional encoding (no CLS token for local attention)
        x = self._add_2d_positional_encoding(x)

        # Apply Transformer blocks
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [T, B, embed_dim]

        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                x, attn_weights = block(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = block(x)

        patch_embeddings = x.transpose(0, 1)  # [B, num_patches, embed_dim]
        patch_logits = self.mlp_head(patch_embeddings)  # [B, num_patches, num_output_channels] - normalized log predictions

    
        # Convert to raw flux with MUCH looser clamps
        if isinstance(sxr_norm, dict):
            patch_flux_raw = torch.zeros_like(patch_logits)
            for i in range(patch_logits.shape[-1]):
                channel_key = 'a' if i == 0 else 'b'
                mean = float(sxr_norm[channel_key][0].item())
                std = float(sxr_norm[channel_key][1].item())
                # Key fix: Don't clamp individual patches so aggressively
                # Let the model learn the right scale
                patch_flux_raw[:, :, i] = torch.exp(
                    (patch_logits[:, :, i] * std + mean) * np.log(10)
                )  # Using exp instead of 10** can be more stable
        else:
            mean, std = sxr_norm
            patch_flux_raw = torch.exp((patch_logits * std + mean) * np.log(10))
        
        # Only clamp patches very loosely (or not at all)
        patch_flux_raw = torch.clamp(patch_flux_raw, min=1e-12, max=10.0)  # Much wider range
        
        # Sum patches
        global_flux_raw = patch_flux_raw.sum(dim=1)
        
        # Clamp global sum (this is your actual prediction)
        global_flux_raw = torch.clamp(global_flux_raw, min=1e-10, max=10.0)  # Wider range
        
        if return_attention:
            return global_flux_raw, attention_weights, patch_flux_raw
        else:
            return global_flux_raw, patch_flux_raw
        
    def _add_2d_positional_encoding(self, x):
        """Add learned 2D positional encoding to patch embeddings"""
        B, T, embed_dim = x.shape
        num_patches = T  # All tokens are patches (no CLS token)
        
        # Reshape patches to 2D grid: [B, grid_h, grid_w, embed_dim]
        patch_embeddings = x.reshape(B, self.grid_h, self.grid_w, embed_dim)
        
        # Add learned 2D positional encoding
        # Broadcasting: [B, grid_h, grid_w, embed_dim] + [1, grid_h, grid_w, embed_dim]
        patch_embeddings = patch_embeddings + self.pos_embedding_2d
        
        # Reshape back to sequence format: [B, num_patches, embed_dim]
        x = patch_embeddings.reshape(B, num_patches, embed_dim)
                
        return x
    
    def forward_for_callback(self, x, return_attention=True):
        """Forward method compatible with AttentionMapCallback"""
        global_flux_raw, attention_weights, patch_flux_raw = self.forward(x, return_attention=return_attention)
        # Callback expects (outputs, attention_weights, _)
        return global_flux_raw, attention_weights    


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


class LocalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_patches, dropout=0.0, local_window=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.local_window = local_window
        self.num_patches = num_patches
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
        
        # Pre-compute attention mask for local interactions
        self.register_buffer('attention_mask', self._create_local_attention_mask())

    def _create_local_attention_mask(self):
        """Create attention mask for local interactions only"""
        # This creates a mask where only nearby patches can attend to each other
        # For a 32x32 grid of patches with local_window=3, each patch can only
        # attend to patches within a 3x3 window around it
        
        # This is a simplified version - you'd need to implement based on your grid size
        num_patches = self.num_patches  # 32x32 patches
        grid_size = int(math.sqrt(num_patches))
        
        # Create mask for patches only: [num_patches, num_patches]
        mask = torch.zeros(num_patches, num_patches)
        
        # Patches can only attend to nearby patches
        for i in range(num_patches):
            row_i, col_i = i // grid_size, i % grid_size
            for j in range(num_patches):
                row_j, col_j = j // grid_size, j % grid_size
                # Only allow attention if patches are within local_window distance
                if abs(row_i - row_j) <= self.local_window // 2 and abs(col_i - col_j) <= self.local_window // 2:
                    mask[i, j] = 1
        
        return mask.bool()

    def forward(self, x, return_attention=False):
        inp_x = self.layer_norm_1(x)
        
        if return_attention:
            # Apply local attention mask
            attn_output, attn_weights = self.attn(
                inp_x, inp_x, inp_x, 
                attn_mask=self.attention_mask,
                average_attn_weights=False
            )
            x = x + attn_output
            x = x + self.linear(self.layer_norm_2(x))
            return x, attn_weights
        else:
            attn_output = self.attn(
                inp_x, inp_x, inp_x,
                attn_mask=self.attention_mask
            )[0]
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
    def __init__(self, window_size=15000, base_weights=None):    
        self.c_threshold = 1e-6
        self.m_threshold = 1e-5
        self.x_threshold = 1e-4

        self.window_size = window_size
        self.quiet_errors = deque(maxlen=window_size)
        self.c_errors = deque(maxlen=window_size)
        self.m_errors = deque(maxlen=window_size)
        self.x_errors = deque(maxlen=window_size)

        #Calculate the base weights based on the number of samples in each class within training data
        if base_weights is None:
            self.base_weights = self._get_base_weights()
        else:
            self.base_weights = base_weights

    def _get_base_weights(self):
        #Calculate the base weights based on the number of samples in each class within training data
        return {
            'quiet': 1.5,    # Increase from current value
            'c_class': 1.0,  # Keep as baseline
            'm_class': 8.0,  # Maintain M-class focus
            'x_class': 20.0  # Maintain X-class focus
        }

    def calculate_loss(self, preds_norm, sxr_norm, sxr_un):
        base_loss = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
        #base_loss = F.mse_loss(preds_norm, sxr_norm, reduction='none')
        weights = self._get_adaptive_weights(sxr_un)
        self._update_tracking(sxr_un, sxr_norm, preds_norm)
        weighted_loss = base_loss * weights
        loss = weighted_loss.mean()
        return loss, weights

    def _get_adaptive_weights(self, sxr_un):
        device = sxr_un.device

        # Get continuous multipliers per class with custom params
        quiet_mult = self._get_performance_multiplier(
            self.quiet_errors, max_multiplier=1.5, min_multiplier=0.6, sensitivity=0.05, sxrclass='quiet'  # Was 0.2
        )
        c_mult = self._get_performance_multiplier(
            self.c_errors, max_multiplier=2, min_multiplier=0.7, sensitivity=0.08, sxrclass='c_class'    # Was 0.3
        )
        m_mult = self._get_performance_multiplier(
            self.m_errors, max_multiplier=5.0, min_multiplier=0.8, sensitivity=0.1, sxrclass='m_class'   # Was 0.4
        )
        x_mult = self._get_performance_multiplier(
            self.x_errors, max_multiplier=8.0, min_multiplier=0.8, sensitivity=0.12, sxrclass='x_class'  # Was 0.5
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
        weights = weights / (mean_weight)

        # Clamp extreme weights
        #weights = torch.clamp(weights, min=0.01, max=40.0)

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
            'quiet': {'min_samples': 2500, 'recent_window': 800},
            'c_class': {'min_samples': 2500, 'recent_window': 800},
            'm_class': {'min_samples': 1500, 'recent_window': 500},
            'x_class': {'min_samples': 1000, 'recent_window': 300}
        }

        # target_errors = {
        #     'quiet': 0.15,
        #     'c_class': 0.08,
        #     'm_class': 0.05,
        #     'x_class': 0.05
        # }
        
        #target = target_errors[sxrclass]

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


        
        # if len(error_history) < class_params[sxrclass]['min_samples']:
        #     return 1.0
        
        # recent = np.mean(list(error_history)[-class_params[sxrclass]['recent_window']:])
        
        # if recent > target:  # Not meeting target - increase weight
        #     excess_error = (recent - target) / target
        #     multiplier = 1.0 + sensitivity * excess_error
        # else:  # Meeting/exceeding target
        #     if sxrclass == 'quiet':
        #         # Can reduce quiet weight significantly
        #         multiplier = max(0.5, 1.0 - 0.5 * (target - recent) / target)
        #     else:
        #         # Keep important classes weighted well even when performing good
        #         multiplier = max(0.8, 1.0 - 0.2 * (target - recent) / target)
        
        # return np.clip(multiplier, min_multiplier, max_multiplier)

    def _update_tracking(self, sxr_un, sxr_norm, preds_norm):
        sxr_un_np = sxr_un.detach().cpu().numpy()

        #Huber loss
        error = F.huber_loss(preds_norm, sxr_norm, delta=.3, reduction='none')
        #error = F.mse_loss(preds_norm, sxr_norm, reduction='none')
        error = error.detach().cpu().numpy()

    
        quiet_mask = sxr_un_np < self.c_threshold
        if quiet_mask.sum() > 0:
            self.quiet_errors.append(float(np.mean(error[quiet_mask])))

        c_mask = (sxr_un_np >= self.c_threshold) & (sxr_un_np < self.m_threshold)
        if c_mask.sum() > 0:
            self.c_errors.append(float(np.mean(error[c_mask])))

        m_mask = (sxr_un_np >= self.m_threshold) & (sxr_un_np < self.x_threshold)
        if m_mask.sum() > 0:
            self.m_errors.append(float(np.mean(error[m_mask])))

        x_mask = sxr_un_np >= self.x_threshold
        if x_mask.sum() > 0:
            self.x_errors.append(float(np.mean(error[x_mask])))


    def get_current_multipliers(self):
        """Get current performance multipliers for logging"""
        return {
            'quiet_mult': self._get_performance_multiplier(
                self.quiet_errors, max_multiplier=1.5, min_multiplier=0.6, sensitivity=0.2, sxrclass='quiet'
            ),
            'c_mult': self._get_performance_multiplier(
                self.c_errors, max_multiplier=2, min_multiplier=0.7, sensitivity=0.3, sxrclass='c_class'
            ),
            'm_mult': self._get_performance_multiplier(
                self.m_errors, max_multiplier=5.0, min_multiplier=0.8, sensitivity=0.8, sxrclass='m_class'
            ),
            'x_mult': self._get_performance_multiplier(
                self.x_errors, max_multiplier=8.0, min_multiplier=0.8, sensitivity=1.0, sxrclass='x_class'
            ),
            'quiet_count': len(self.quiet_errors),
            'c_count': len(self.c_errors),
            'm_count': len(self.m_errors),
            'x_count': len(self.x_errors),
            'quiet_error': np.mean(self.quiet_errors) if self.quiet_errors else 0.0,
            'c_error': np.mean(self.c_errors) if self.c_errors else 0.0,
            'm_error': np.mean(self.m_errors) if self.m_errors else 0.0,
            'x_error': np.mean(self.x_errors) if self.x_errors else 0.0,
            'quiet_weight': getattr(self, 'current_multipliers', {}).get('quiet_weight', 0.0),
            'c_weight': getattr(self, 'current_multipliers', {}).get('c_weight', 0.0),
            'm_weight': getattr(self, 'current_multipliers', {}).get('m_weight', 0.0),
            'x_weight': getattr(self, 'current_multipliers', {}).get('x_weight', 0.0)
        }