import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

class SpectralAttention(nn.Module):
    """Spectral Attention Module"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        return self.mlp(x)

class EfficientAttention(nn.Module):
    """Efficient Attention with Linear Complexity"""
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.softmax(dim=-2)
        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out

class FastViTBlock(nn.Module):
    """FastViT Block integrating efficient attention and an FFN"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FastViTFlaringModel(pl.LightningModule):
    """FastViT Model for Solar Flare Prediction"""
    def __init__(self, d_input=6, d_output=1, eve_norm=(0, 1), lr=1e-4,
                 image_size=224, patch_size=16, embed_dim=768, depth=6, num_heads=8):
        super().__init__()
        self.save_hyperparameters()
        self.eve_norm = eve_norm
        self.lr = lr

        # Patch embedding
        self.patch_embed = nn.Conv2d(d_input, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FastViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # Spectral attention and regression head
        self.spectral_attention = SpectralAttention(embed_dim)
        self.regression_head = nn.Linear(embed_dim, d_output)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        # Input shape: (B, H, W, C) -> convert to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling + spectral attention
        x = self.spectral_attention(x.mean(dim=1))
        output = self.regression_head(x)
        return output.squeeze(-1)

    def training_step(self, batch, batch_idx):
        (aia_img, _), sxr_target = batch
        sxr_pred = self(aia_img)
        loss = self.loss_func(sxr_pred, sxr_target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (aia_img, _), sxr_target = batch
        sxr_pred = self(aia_img)
        loss = self.loss_func(sxr_pred, sxr_target)
        self.log('valid_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        (aia_img, _), sxr_target = batch
        sxr_pred = self(aia_img)
        loss = self.loss_func(sxr_pred, sxr_target)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (aia_img, _), _ = batch
        sxr_pred = self(aia_img)
        sxr_pred = sxr_pred * self.eve_norm[1] + self.eve_norm[0]
        sxr_pred = 10 ** sxr_pred - 1e-8
        return sxr_pred