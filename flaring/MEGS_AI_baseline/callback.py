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
        self.val_aia = data_samples[0][0]
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
    def __init__(self, log_every_n_epochs=10, num_samples=4, save_dir="attention_maps"):
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
                self._plot_attention_map(
                    imgs[sample_idx],
                    attention_weights,
                    sample_idx,
                    trainer.current_epoch,
                    pl_module.model.patch_size
                )

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
        # Convert image to numpy for plotting
        print(image.shape)
        img_np = image.cpu().numpy()
        print(img_np.shape)
        # Normalize image for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # Get attention from the last layer (or you can average across layers)
        last_layer_attention = attention_weights[-1]  # [B, num_heads, seq_len, seq_len]

        # Extract attention for this sample
        sample_attention = last_layer_attention[sample_idx]  # [num_heads, seq_len, seq_len]

        # Average across heads (or you can visualize individual heads)
        avg_attention = sample_attention.mean(dim=0)  # [seq_len, seq_len]

        # Get attention from CLS token to patches (exclude CLS->CLS)
        cls_attention = avg_attention[0, 1:].cpu()  # [num_patches]

        # Calculate grid size
        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size
        print(grid_h, grid_w)
        # Reshape attention to spatial grid
        attention_map = cls_attention.reshape(grid_h, grid_w)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Original image
        axes[0].imshow(img_np[::0])
        axes[0].set_title(f'Original Image (Epoch {epoch})')
        axes[0].axis('off')

        # Plot 2: Attention heatmap
        im = axes[1].imshow(attention_map.numpy(), cmap='hot', interpolation='nearest')
        axes[1].set_title(f'Attention Map (Sample {sample_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # Plot 3: Overlay attention on image
        axes[2].imshow(img_np[::0])

        # Overlay attention as colored patches
        for i in range(grid_h):
            for j in range(grid_w):
                attention_val = attention_map[i, j].item()
                # Create a colored rectangle with alpha based on attention
                rect = patches.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=0,
                    facecolor='red',
                    alpha=attention_val * 0.7  # Scale alpha by attention
                )
                axes[2].add_patch(rect)

        axes[2].set_title(f'Attention Overlay (Sample {sample_idx})')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

        # Save the plot
        # import os
        # os.makedirs(self.save_dir, exist_ok=True)
        # plt.savefig(f'{self.save_dir}/attention_epoch_{epoch}_sample_{sample_idx}.png',
        #             dpi=150, bbox_inches='tight')
        # plt.close()
