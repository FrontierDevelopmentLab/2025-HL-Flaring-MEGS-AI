import wandb
import os
import torch
from pytorch_lightning import Callback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.visualization.colormaps as cm
import astropy.units as u
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import zoom

# Register SDO AIA 94 colormap
sdoaia94 = matplotlib.colormaps['sdoaia94']

class ImagePredictionLogger_SXR(Callback):
    def __init__(self, data_samples, sxr_norm):
        super().__init__()
        self.data_samples = data_samples
        self.sxr_norm = sxr_norm

    def on_validation_epoch_end(self, trainer, pl_module):
        aia_images = []
        true_sxr = []
        pred_sxr = []

        for aia, target in self.data_samples:
            aia = aia.to(pl_module.device).unsqueeze(0)  # [1, T, C, H, W]
            with torch.no_grad():
                pred = pl_module(aia)  # [1, T_out]
                if isinstance(pred, tuple):
                    pred = pred[0]  # Handle attention output
            aia_images.append(aia.squeeze(0).cpu().numpy())  # [T, C, H, W]
            true_sxr.append(target.cpu().numpy())  # [T_out]
            pred_sxr.append(pred.cpu().numpy().squeeze())  # [T_out]

        true_sxr = [pl_module.denormalize_sxr(torch.tensor(t)).cpu().numpy() for t in true_sxr]
        pred_sxr = [pl_module.denormalize_sxr(torch.tensor(p)).cpu().numpy() for p in pred_sxr]

        fig1 = self.plot_aia_sxr(aia_images, true_sxr, pred_sxr, pl_module.input_sequence_length)
        trainer.logger.experiment.log({"Soft X-ray Flux Sequence Plots": wandb.Image(fig1)})
        plt.close(fig1)

        fig2 = self.plot_aia_sxr_difference(aia_images, true_sxr, pred_sxr, pl_module.input_sequence_length)
        trainer.logger.experiment.log({"Soft X-ray Flux Difference Sequence Plots": wandb.Image(fig2)})
        plt.close(fig2)

    def plot_aia_sxr(self, aia_images, true_sxr, pred_sxr, input_seq_length):
        num_samples = len(aia_images)
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))

        for i in range(num_samples):
            aia_seq = aia_images[i].transpose(0, 2, 3, 1)[:, :, :, 0]  # [T, H, W]
            num_cols = min(6, input_seq_length)
            num_rows = int(np.ceil(input_seq_length / num_cols))
            mosaic = np.zeros((num_rows * 512, num_cols * 512))
            for t in range(input_seq_length):
                row = t // num_cols
                col = t % num_cols
                mosaic[row*512:(row+1)*512, col*512:(col+1)*512] = aia_seq[t]

            ax = axes[i, 0] if num_samples > 1 else axes[0]
            ax.imshow(mosaic, cmap='sdoaia94')
            ax.set_title(f'Sample {i}: AIA Sequence (Channel 0)')
            ax.axis('off')

            time_steps = np.arange(input_seq_length, input_seq_length + len(true_sxr[i]))
            ax = axes[i, 1] if num_samples > 1 else axes[1]
            ax.plot(time_steps, true_sxr[i], 'b-', label='True SXR', marker='o')
            ax.plot(time_steps, pred_sxr[i], 'r--', label='Predicted SXR', marker='x')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('SXR Flux (W/m²)')
            ax.set_title(f'Sample {i}: SXR Prediction')
            ax.set_yscale('log')
            ax.legend()
            ax.grid(True)

        fig.tight_layout()
        return fig

    def plot_aia_sxr_difference(self, aia_images, true_sxr, pred_sxr, input_seq_length):
        num_samples = len(aia_images)
        fig, axes = plt.subplots(num_samples, 1, figsize=(6, 3 * num_samples))

        for i in range(num_samples):
            ax = axes[i] if num_samples > 1 else axes
            time_steps = np.arange(input_seq_length, input_seq_length + len(true_sxr[i]))
            ax.plot(time_steps, true_sxr[i] - pred_sxr[i], 'b-', label='True - Pred', marker='o')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('SXR Flux Difference (W/m²)')
            ax.set_title(f'Sample {i}: SXR Difference (True - Predicted)')
            ax.legend()
            ax.grid(True)

        fig.tight_layout()
        return fig

class AttentionMapCallback(Callback):
    def __init__(self, log_every_n_epochs=1, num_samples=4, save_dir="attention_maps"):
        """
        Callback to visualize attention maps for sequence inputs during training and validation.

        Args:
            log_every_n_epochs: How often to log attention maps during validation
            num_samples: Number of samples to visualize
            save_dir: Directory to save attention maps
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch == 0 and batch_idx < 2:  # Log for first 2 batches of epoch 0
            #print(f"Training epoch 0, batch {batch_idx}: Running AttentionMapCallback")
            self._visualize_attention(trainer, pl_module, batch=batch, phase="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        #print(f"Validation epoch {trainer.current_epoch}: Running AttentionMapCallback")
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._visualize_attention(trainer, pl_module, phase="val")

    def _visualize_attention(self, trainer, pl_module, batch=None, phase="val"):
        if batch is None:
            val_dataloader = trainer.val_dataloaders
            if val_dataloader is None:
               # print(f"No {phase} dataloader available")
                return
            batch = next(iter(val_dataloader))
        imgs, labels = batch  # imgs: [B, T, C, H, W], labels: [B, T_out]
       # print(f"{phase.capitalize()} batch shapes: imgs={imgs.shape}, labels={labels.shape}")
        imgs = imgs[:self.num_samples].to(pl_module.device)

        with torch.no_grad():
            outputs, attention_weights = pl_module(imgs, return_attention=True)
       # print(f"Attention weights shape: {attention_weights.shape}")

        patch_size = pl_module.hparams.model_kwargs.get('patch_size')
        #print(f"Using patch_size: {patch_size}")

        for sample_idx in range(min(self.num_samples, imgs.size(0))):
            frame = imgs[sample_idx, 0]  # [C, H, W]
            map_fig = self._plot_attention_map(
                frame, attention_weights, sample_idx, trainer.current_epoch, patch_size
            )
            trainer.logger.experiment.log({
                f"{phase.capitalize()} Attention Map Sample {sample_idx} Epoch {trainer.current_epoch}": wandb.Image(map_fig)
            })
            plt.savefig(
                f'{self.save_dir}/{phase}_attention_epoch_{trainer.current_epoch}_sample_{sample_idx}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close(map_fig)

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size):
        """
        Plot attention map for a single frame.

        Args:
            image: Input frame tensor [C, H, W]
            attention_weights: Attention weights from ViT [n_layers, B, num_heads, seq_len, seq_len]
            sample_idx: Index of the sample in the batch
            epoch: Current epoch number
            patch_size: Size of patches
        """
        img_np = image.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size

        last_layer_attention = attention_weights[-1, sample_idx]  # [num_heads, seq_len, seq_len]
        avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]
        cls_attention = avg_attention[0, 1:].cpu()  # [num_patches]

        attention_map = cls_attention.reshape(grid_h, grid_w)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if img_np.shape[2] >= 3:
            img_display = np.clip(img_np[:, :, [0, 2, 4]], 0, 1)  # Channels 0, 2, 4 for RGB
        else:
            img_display = np.stack([np.clip(img_np[:, :, 0], 0, 1)] * 3, axis=2)
        axes[0].imshow(img_display, cmap='sdoaia94')
        axes[0].set_title(f'Original Frame (Epoch {epoch})')
        axes[0].axis('off')

        attention_np = np.log1p(attention_map.numpy())
        attention_resized = zoom(attention_np, (H / grid_h, W / grid_w), order=1)
        im = axes[1].imshow(attention_resized, cmap='hot')
        axes[1].set_title(f'Attention Map (Sample {sample_idx})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        axes[2].imshow(img_display, cmap='sdoaia94')
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[2].set_title(f'Log-Scaled Attention Overlay (Sample {sample_idx})')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

class MultiHeadAttentionCallback(AttentionMapCallback):
    """Extended callback to visualize individual attention heads."""
    def __init__(self, log_every_n_epochs=1, num_samples=4, save_dir="attention_maps"):
        super().__init__(log_every_n_epochs, num_samples, save_dir)

    def _plot_attention_map(self, image, attention_weights, sample_idx, epoch, patch_size):
        avg_fig = super()._plot_attention_map(image, attention_weights, sample_idx, epoch, patch_size)
        trainer = self.trainer
        trainer.logger.experiment.log({
            f"Average Attention Sample {sample_idx} Epoch {epoch}": wandb.Image(avg_fig)
        })
        plt.close(avg_fig)

        img_np = image.cpu().numpy().transpose(1, 2, 0)
        H, W = img_np.shape[:2]
        grid_h, grid_w = H // patch_size, W // patch_size
        last_layer_attention = attention_weights[-1, sample_idx]  # [num_heads, seq_len, seq_len]
        num_heads = last_layer_attention.size(0)

        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(rows, cols) if rows > 1 else np.array([axes])

        for head_idx in range(num_heads):
            row = head_idx // cols
            col = head_idx % cols
            head_attention = last_layer_attention[head_idx, 0, 1:].cpu()  # [num_patches]
            attention_map = head_attention.reshape(grid_h, grid_w)
            attention_np = np.log1p(attention_map.numpy())
            attention_resized = zoom(attention_np, (H / grid_h, W / grid_w), order=1)

            ax = axes[row, col]
            ax.imshow(attention_resized, cmap='hot')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')
            plt.colorbar(ax.imshow(attention_resized, cmap='hot'), ax=ax)

        for idx in range(num_heads, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/heads_epoch_{epoch}_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        trainer.logger.experiment.log({
            f"Attention Heads Sample {sample_idx} Epoch {epoch}": wandb.Image(fig)
        })
        return fig