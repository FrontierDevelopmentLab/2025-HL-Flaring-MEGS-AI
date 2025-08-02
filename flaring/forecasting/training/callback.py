import wandb
from pytorch_lightning import Callback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import sunpy.visualization.colormaps as cm
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
from PIL import Image
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

# Custom Callback
sdoaia94 = matplotlib.colormaps['sdoaia94']


def unnormalize_sxr(normalized_values, sxr_norm):
    if isinstance(normalized_values, torch.Tensor):
        normalized_values = normalized_values.cpu().numpy()
    normalized_values = np.array(normalized_values, dtype=np.float32)
    return 10 ** (normalized_values * float(sxr_norm[1].item()) + float(sxr_norm[0].item())) - 1e-8


class ImagePredictionLogger_SXR(Callback):

    def __init__(self, data_samples, sxr_norm):
        super().__init__()
        self.data_samples = data_samples
        self.val_aia = data_samples[0]
        self.val_sxr = data_samples[1]
        self.sxr_norm = sxr_norm

    def on_validation_epoch_end(self, trainer, pl_module):

        aia_images = []
        true_sxr = []
        pred_sxr = []
        # print(self.val_samples)
        for aia, target in self.data_samples:
            #device = torch.device("cuda:0")
            aia = aia.to(pl_module.device).unsqueeze(0)
            # Get prediction

            pred = pl_module(aia)
            #pred = self.unnormalize_sxr(pred)
            pred_sxr.append(pred.item())
            aia_images.append(aia.squeeze(0).cpu().numpy())
            true_sxr.append(target.item())

        true_unorm = unnormalize_sxr(true_sxr,self.sxr_norm)
        pred_unnorm = unnormalize_sxr(pred_sxr,self.sxr_norm)
        fig1 = self.plot_aia_sxr(aia_images,true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux plots": wandb.Image(fig1)})
        plt.close(fig1)
        fig2 = self.plot_aia_sxr_difference(aia_images, true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"Soft X-ray flux difference plots": wandb.Image(fig2)})
        plt.close(fig2)

    def plot_aia_sxr(self, val_aia, val_sxr, pred_sxr):
        num_samples = len(val_aia)
        fig, axes = plt.subplots(1, 1, figsize=(5, 2))

        for i in range(num_samples):
            axes.scatter(i, val_sxr[i], label='Ground truth' if i == 0 else "", color='blue')
            axes.scatter(i, pred_sxr[i], label='Prediction' if i == 0 else "", color='orange')
        axes.set_xlabel("Index")
        axes.set_ylabel("Soft x-ray flux [W/m2]")
        axes.set_yscale('log')
        axes.legend()

        fig.tight_layout()
        return fig

    def plot_aia_sxr_difference(self, val_aia, val_sxr, pred_sxr):
        num_samples = len(val_aia)
        fig, axes = plt.subplots(1, 1, figsize=(5, 2))
        for i in range(num_samples):
            # print("Aia images:", val_aia[i])
            axes.scatter(i, val_sxr[i]-pred_sxr[i], label='Soft X-ray Flux Difference', color='blue')
            axes.set_xlabel("Index")
            axes.set_ylabel("Soft X-ray Flux Difference (True - Pred.) [W/m2]")

        fig.tight_layout()
        return fig


class AttentionMapCallback(Callback):
    def __init__(self, log_every_n_epochs=1, num_samples=4, save_dir="attention_maps"):
        """
        Callback to visualize attention maps during training.

        Args:
            log_every_n_epochs: How often to log attention maps
            num_samples: Number of samples to visualize
            save_dir: Directory to save attention maps
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_attention(trainer, pl_module)

    def _visualize_attention(self, trainer, pl_module):
        # Get a batch from validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return

        pl_module.eval()
        with torch.no_grad():
            # Get a batch of data
            batch = next(iter(val_dataloader))
            imgs, labels = batch

            # Move to device
            imgs = imgs[:self.num_samples].to(pl_module.device)

            # Get predictions with attention weights
            outputs, attention_weights = pl_module(imgs, return_attention=True)

            # Visualize attention for each sample
            for sample_idx in range(min(self.num_samples, imgs.size(0))):

                map = self._plot_attention_map(
                    imgs[sample_idx],
                    attention_weights,
                    sample_idx,
                    trainer.current_epoch,
                    pl_module.model.patch_size
                )
                trainer.logger.experiment.log({"Attention plots": wandb.Image(map)})
                plt.close(map)

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size):
        """
        Plot attention map for a single image.

        Args:
            image: Input image tensor [C, H, W]
            attention_weights: List of attention weights from each layer
            sample_idx: Index of the sample in the batch
            epoch: Current epoch number
            patch_size: Size of patches
        """
        # Convert image to numpy and transpose
        img_np = image.cpu().numpy()
        if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3]:  # Check if channels first
            img_np = np.transpose(img_np, (1, 2, 0))


        # Get attention from the last layer
        last_layer_attention = attention_weights[-1]  # [B, num_heads, seq_len, seq_len]

        # Extract attention for this sample
        sample_attention = last_layer_attention[sample_idx]  # [num_heads, seq_len, seq_len]

        # Average across heads
        avg_attention = sample_attention.mean(dim=0)  # [seq_len, seq_len]

        # Get attention from CLS token to patches (exclude CLS->CLS)
        cls_attention = avg_attention[0, 1:].cpu()  # [num_patches]

        # Calculate grid size - NOW USING CORRECT DIMENSIONS
        H, W = img_np.shape[:2]  # Now this is correct after transpose
        grid_h, grid_w = H // patch_size, W // patch_size

        # Reshape attention to spatial grid
        attention_map = cls_attention.reshape(grid_h, grid_w)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Original image
        # if img_np.shape[2] == 1:  # Grayscale
        #     img_display = (img_np[:, :, 0] + 1) / 2
        #     axes[0].imshow(img_display, cmap='gray')
        # elif img_np.shape[2] == 3:  # RGB
        #     # Normalize RGB image properly
        #     img_display = (img_np + 1) / 2  # Assuming images are in [-1, 1] range
        #     img_display = np.clip(img_display, 0, 1)  # Ensure valid range
        #     axes[0].imshow(img_display)
        # else:  # Multi-channel (6 channels in your case)
        #     # Option 1: Display first channel as grayscale
        #     img_display = (img_np[:, :, 0] + 1) / 2
        #     axes[0].imshow(img_display, cmap='gray')

            # Option 2: Create RGB composite from 3 channels (uncomment if preferred)
        if len(img_np[0,0,:]) >= 6:  # Ensure we have enough channels
            rgb_channels = [0, 2, 4]  # Select which channels to use for R, G, B
            img_display = np.stack([(img_np[:, :, i] + 1) / 2 for i in rgb_channels], axis=2)
            img_display = np.clip(img_display, 0, 1)
        else:
            # If not enough channels, use grayscale
            img_display = (img_np[:, :, 0] + 1) / 2
            img_display = np.stack([img_display] * 3, axis=2)
        axes[0].imshow(img_display)
        axes[0].set_title(f'Original Image (Epoch {epoch})')
        axes[0].axis('off')

        # Plot 2: Attention heatmap
        attention_np = np.log1p(attention_map.numpy())
        # Resize attention map to match image size
        attention_resized = zoom(attention_np, (H / grid_h, W / grid_w), order=1)

        # Create colormap for attention - FIX: Use the scalar values, not RGB
        im = axes[1].imshow(attention_resized, cmap='hot')
        axes[1].set_title(f'Attention Map (Sample {sample_idx})')
        axes[1].axis('off')
        # FIXED: Create colorbar from the scalar image, not RGB
        plt.colorbar(im, ax=axes[1])

        # Plot 3: Overlay attention on image
        #img_display_overlay = (img_np[:, :, 0] + 1) / 2
        axes[2].imshow(img_display)

        # Overlay attention with proper alpha blending
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[2].set_title(f'Log-Scaled Attention Overlay (Sample {sample_idx})')
        axes[2].axis('off')

        plt.tight_layout()

        plt.tight_layout()
        return fig


class MultiHeadAttentionCallback(AttentionMapCallback):
    """Extended callback to visualize individual attention heads."""

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size):
        # Call parent method for average attention
        super()._plot_attention_map(image, attention_weights, sample_idx, epoch, patch_size)

        # Also plot individual heads
        self._plot_individual_heads(image, attention_weights, sample_idx, epoch, patch_size)

    def _plot_individual_heads(self, image, attention_weights, sample_idx, epoch, patch_size):
        """Plot attention maps for individual heads."""
        img_np = image.cpu().numpy()
        #img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        last_layer_attention = attention_weights[-1][sample_idx]  # [num_heads, seq_len, seq_len]
        num_heads = last_layer_attention.size(0)

        # Calculate grid size
        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size

        # Create subplot grid
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if num_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols

            # Get attention for this head
            head_attention = last_layer_attention[head_idx, 0, 1:].cpu()  # CLS to patches
            attention_map = head_attention.reshape(grid_h, grid_w)

            ax = axes[row, col] if rows > 1 else axes[col]
            im = ax.imshow(attention_map.numpy(), cmap='hot', interpolation='nearest')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        # Hide unused subplots
        for idx in range(num_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/heads_epoch_{epoch}_sample_{sample_idx}.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
