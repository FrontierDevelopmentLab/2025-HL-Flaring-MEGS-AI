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
        imgs, sxr = batch  # sxr is normalized
        preds = self.model(imgs)  # preds are also normalized (model trained on normalized data)
        preds_squeezed = torch.squeeze(preds)

        # Calculate base loss (on normalized values)
        base_loss = F.huber_loss(preds_squeezed, sxr, delta=1.0, reduction='none')

        # ===== UNNORMALIZE FOR THRESHOLD COMPARISONS =====
        sxr_un = unnormalize_sxr(sxr,norm)  # Convert to actual flux values

        preds_squeezed_un = unnormalize_sxr(preds_squeezed,norm)  # Convert predictions too

        # ===== M/X CLASS FLARE SPECIFIC THRESHOLDS =====
        # GOES SXR flux thresholds for flare classes (in original units):
        c_class_threshold = 1e-6  # C-class flares
        m_class_threshold = 1e-5  # M-class flares
        x_class_threshold = 1e-4  # X-class flares

        # ===== STRATIFIED FLUX WEIGHTING =====
        # Much more aggressive weighting for M/X class events
        flux_weights = torch.ones_like(sxr_un)
        flux_weights = torch.where(sxr_un >= c_class_threshold, 2.0, flux_weights)  # C-class: 2x
        flux_weights = torch.where(sxr_un >= m_class_threshold, 10.0, flux_weights)  # M-class: 10x
        flux_weights = torch.where(sxr_un >= x_class_threshold, 10.0, flux_weights)  # X-class: 25x

        # ===== PREDICTION-BASED WEIGHTING FOR M/X =====
        # Heavy penalty when model fails to predict M/X class flares
        pred_weights = torch.ones_like(preds_squeezed_un)

        # If actual is M/X but prediction is too low → massive penalty
        missed_m_flares = (sxr_un >= m_class_threshold) & (preds_squeezed_un < m_class_threshold * 0.5)
        missed_x_flares = (sxr_un >= x_class_threshold) & (preds_squeezed_un < x_class_threshold * 0.5)

        pred_weights = torch.where(missed_m_flares, 10.0, pred_weights)  # 15x penalty for missed M
        pred_weights = torch.where(missed_x_flares, 10.0, pred_weights)  # 50x penalty for missed X

        # If prediction is M/X class → moderate penalty to reduce false positives
        pred_weights = torch.where(preds_squeezed_un >= m_class_threshold, 3.0, pred_weights)

        # ===== FALSE ALARM REDUCTION =====
        # Penalize false alarms (predicting M/X when actual is lower)
        false_m_alarms = (preds_squeezed_un >= m_class_threshold) & (sxr_un < c_class_threshold)
        false_x_alarms = (preds_squeezed_un >= x_class_threshold) & (sxr_un < m_class_threshold)

        false_alarm_weights = torch.ones_like(sxr_un)
        false_alarm_weights = torch.where(false_m_alarms, 5.0, false_alarm_weights)  # 5x penalty for false M
        false_alarm_weights = torch.where(false_x_alarms, 5.0, false_alarm_weights)  # 8x penalty for false X


        # ===== COMBINE ALL WEIGHTS =====
        total_weights = flux_weights * pred_weights * false_alarm_weights

        # Cap maximum weight to prevent training instability
        total_weights = torch.clamp(total_weights, min=1.0, max=100.0)

        # Apply weights to loss
        weighted_loss = base_loss * total_weights
        loss = weighted_loss.mean()

        # ===== M/X CLASS SPECIFIC METRICS =====
        # Track performance specifically on M/X class events (using unnormalized values)
        m_class_mask = sxr_un >= m_class_threshold
        x_class_mask = sxr_un >= x_class_threshold

        if m_class_mask.any():
            m_class_mae = F.l1_loss(preds_squeezed[m_class_mask], sxr[m_class_mask])
            self.log(f"{mode}_m_class_mae", m_class_mae, sync_dist=True)

            # M-class detection rate (did we predict >= M when actual >= M?)
            # FIXED: Use unnormalized predictions for threshold comparison
            m_detection_rate = (preds_squeezed_un[m_class_mask] >= m_class_threshold).float().mean()
            self.log(f"{mode}_m_detection_rate", m_detection_rate, sync_dist=True)

        if x_class_mask.any():
            x_class_mae = F.l1_loss(preds_squeezed[x_class_mask], sxr[x_class_mask])
            self.log(f"{mode}_x_class_mae", x_class_mae, sync_dist=True)

            # X-class detection rate
            # FIXED: Use unnormalized predictions for threshold comparison
            x_detection_rate = (preds_squeezed_un[x_class_mask] >= x_class_threshold).float().mean()
            self.log(f"{mode}_x_detection_rate", x_detection_rate, sync_dist=True)

        # False alarm rates
        # FIXED: Use unnormalized values for threshold comparisons
        quiet_mask = sxr_un < c_class_threshold
        if quiet_mask.any():
            false_m_rate = (preds_squeezed_un[quiet_mask] >= m_class_threshold).float().mean()
            false_x_rate = (preds_squeezed_un[quiet_mask] >= x_class_threshold).float().mean()
            self.log(f"{mode}_false_m_rate", false_m_rate, sync_dist=True)
            self.log(f"{mode}_false_x_rate", false_x_rate, sync_dist=True)

        # Calculate standard metrics (on normalized values)
        mae = F.l1_loss(preds_squeezed, sxr)
        mse = F.mse_loss(preds_squeezed, sxr)

        # Log all metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{mode}_mae", mae, sync_dist=True)
        self.log(f"{mode}_mse", mse, sync_dist=True)
        self.log(f"{mode}_avg_weight", total_weights.mean(), sync_dist=True)
        self.log(f"{mode}_max_weight", total_weights.max(), sync_dist=True)
        self.log(f"{mode}_mx_flare_ratio", (sxr_un >= m_class_threshold).float().mean(), sync_dist=True)  # FIXED: Use unnormalized

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