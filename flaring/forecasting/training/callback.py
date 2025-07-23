import wandb
from pytorch_lightning import Callback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import sunpy.visualization.colormaps as cm
import astropy.units as u
from PIL import Image
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

# Set up colormap
sdoaia94 = matplotlib.colormaps['sdoaia94']

def unnormalize_sxr(normalized_values, sxr_norm):
    """Convert normalized SXR values back to physical units"""
    if isinstance(normalized_values, torch.Tensor):
        normalized_values = normalized_values.cpu().numpy()
    normalized_values = np.array(normalized_values, dtype=np.float32)
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8

class ImagePredictionLogger_SXR(Callback):
    """Callback to log SXR predictions vs ground truth"""
    def __init__(self, data_samples, sxr_norm):
        super().__init__()
        self.data_samples = data_samples
        self.sxr_norm = sxr_norm

    def on_validation_epoch_end(self, trainer, pl_module):
        aia_images = []
        true_sxr = []
        pred_sxr = []

        for aia, target in self.data_samples:
            aia = aia.to(pl_module.device).unsqueeze(0)

            # Get prediction (handle both single output and tuple cases)
            pred = pl_module(aia)
            if isinstance(pred, tuple):
                pred = pred[0]  # Take predictions only if model returns tuple

            pred_sxr.append(pred.item())
            aia_images.append(aia.squeeze(0).cpu().numpy())
            true_sxr.append(target.item())

        # Unnormalize values
        true_unorm = unnormalize_sxr(true_sxr, self.sxr_norm)
        pred_unnorm = unnormalize_sxr(pred_sxr, self.sxr_norm)

        # Create and log plots
        fig1 = self.plot_aia_sxr(aia_images, true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux plots": wandb.Image(fig1)})
        plt.close(fig1)

        fig2 = self.plot_aia_sxr_difference(aia_images, true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux difference plots": wandb.Image(fig2)})
        plt.close(fig2)

    def plot_aia_sxr(self, val_aia, val_sxr, pred_sxr):
        """Plot comparison of true vs predicted SXR values"""
        num_samples = len(val_aia)
        fig, ax = plt.subplots(figsize=(10, 5))

        indices = np.arange(num_samples)
        ax.scatter(indices, val_sxr, label='Ground truth', color='blue')
        ax.scatter(indices, pred_sxr, label='Prediction', color='orange')

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Soft X-ray flux [W/m²]")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="--")

        fig.tight_layout()
        return fig

    def plot_aia_sxr_difference(self, val_aia, val_sxr, pred_sxr):
        """Plot difference between true and predicted SXR values"""
        num_samples = len(val_aia)
        fig, ax = plt.subplots(figsize=(10, 5))

        differences = np.array(val_sxr) - np.array(pred_sxr)
        ax.bar(range(num_samples), differences, color='blue')

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Difference (True - Predicted) [W/m²]")
        ax.grid(True, which="both", ls="--")

        fig.tight_layout()
        return fig

class AttentionMapCallback(Callback):
    """Callback to visualize attention maps from transformer models"""
    def __init__(self, log_every_n_epochs=1, num_samples=4):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_attention(trainer, pl_module)

    def _visualize_attention(self, trainer, pl_module):
        """Main visualization method"""
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        pl_module.eval()
        with torch.no_grad():
            try:
                batch = next(iter(val_dataloader))
                imgs, labels = batch
                imgs = imgs[:self.num_samples].to(pl_module.device)

                # Get predictions with attention weights
                outputs, attention_weights = pl_module(imgs, return_attention=True)

                # Visualize each sample
                for sample_idx in range(min(self.num_samples, imgs.size(0))):
                    fig = self._create_attention_figure(
                        imgs[sample_idx],
                        attention_weights,
                        sample_idx,
                        trainer.current_epoch,
                        getattr(pl_module.model, 'patch_size', 16)  # Default to 16 if not found
                    )
                    trainer.logger.experiment.log({
                        f"Attention Sample {sample_idx}": wandb.Image(fig)
                    })
                    plt.close(fig)
            except Exception as e:
                print(f"Error visualizing attention: {e}")

    def _create_attention_figure(self, image, attention_weights, sample_idx, epoch, patch_size):
        """Create a composite figure showing image and attention"""
        # Convert and prepare image
        img_np = image.cpu().numpy()
        if img_np.shape[0] in [1, 3, 6]:  # Channels first format
            img_np = np.transpose(img_np, (1, 2, 0))

        # Create RGB composite (using channels 0, 2, 4 for R,G,B)
        rgb_img = np.stack([
            (img_np[..., 0] + 1) / 2,  # Channel 0 -> Red
            (img_np[..., 2] + 1) / 2,  # Channel 2 -> Green
            (img_np[..., 4] + 1) / 2   # Channel 4 -> Blue
        ], axis=-1)
        rgb_img = np.clip(rgb_img, 0, 1)

        # Process attention weights
        last_layer_attention = attention_weights[-1][sample_idx]  # [num_heads, seq_len, seq_len]
        cls_attention = last_layer_attention.mean(0)[0, 1:].cpu()  # Average heads, get CLS attention

        # Reshape attention to spatial grid
        H, W = rgb_img.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size
        attention_map = cls_attention.reshape(grid_h, grid_w).numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(rgb_img)
        axes[0].set_title(f'Original Image (Epoch {epoch})')
        axes[0].axis('off')

        # Attention heatmap
        attention_resized = zoom(attention_map, (H/grid_h, W/grid_w), order=1)
        im = axes[1].imshow(np.log1p(attention_resized), cmap='hot')
        plt.colorbar(im, ax=axes[1])
        axes[1].set_title('Log-Scaled Attention Map')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(rgb_img)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.4)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

class MultiHeadAttentionCallback(AttentionMapCallback):
    """Extended callback to visualize individual attention heads"""
    def _create_attention_figure(self, image, attention_weights, sample_idx, epoch, patch_size):
        # First show the standard attention visualization
        fig = super()._create_attention_figure(image, attention_weights, sample_idx, epoch, patch_size)

        # Then add individual head visualizations
        last_layer_attention = attention_weights[-1][sample_idx]  # [num_heads, seq_len, seq_len]
        num_heads = last_layer_attention.size(0)

        # Create figure for heads
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        head_fig, head_axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

        if num_heads == 1:
            head_axes = [head_axes]
        elif rows == 1:
            head_axes = head_axes.reshape(1, -1)

        # Process image for overlay
        img_np = image.cpu().numpy()
        if img_np.shape[0] in [1, 3, 6]:
            img_np = np.transpose(img_np, (1, 2, 0))
        H, W = img_np.shape[:2]

        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            ax = head_axes[row, col] if rows > 1 else head_axes[col]

            # Get attention for this head
            head_attn = last_layer_attention[head_idx, 0, 1:].cpu()  # CLS to patches
            attn_map = head_attn.reshape(H//patch_size, W//patch_size).numpy()
            attn_resized = zoom(attn_map, (H/(H//patch_size), W/(W//patch_size)), order=1)

            # Plot
            ax.imshow(np.log1p(attn_resized), cmap='hot')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')

        # Hide unused subplots
        for idx in range(num_heads, rows*cols):
            row = idx // cols
            col = idx % cols
            ax = head_axes[row, col] if rows > 1 else head_axes[col]
            ax.axis('off')

        plt.tight_layout()
        return fig