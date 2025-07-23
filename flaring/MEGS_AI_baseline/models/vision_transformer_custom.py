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
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # return [optimizer], [lr_scheduler]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3),
            'monitor': 'val_loss',  # name of the metric to monitor
            'interval': 'epoch',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        imgs, sxr = batch
        preds = self.model(imgs)

        # Change loss function for regression
        loss = F.huber_loss(torch.squeeze(preds), sxr)  # or F.l1_loss() or F.huber_loss()

        # Change accuracy to a regression metric
        # OR use RMSE:
        # rmse = torch.sqrt(F.mse_loss(preds, labels))

        self.log(f"{mode}_loss", loss)
        self.log('learning_rate', self.lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        #self.log(f"{mode}_mae", mae)  # or f"{mode}_rmse" if using RMSE
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