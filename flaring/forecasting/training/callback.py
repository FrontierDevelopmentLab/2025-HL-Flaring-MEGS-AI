
import wandb
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

# Set non-interactive backend for matplotlib to avoid display issues on servers
import matplotlib
matplotlib.use('Agg')

def unnormalize_sxr(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y= torch.tensor(y)
    y = y * norm_stdev[None] + norm_mean[None]
    return y

class AttentionMapCallback(Callback):
    def __init__(self, log_every_n_epochs=1, num_samples=4):
        super().__init__()
        self.log_every_n_epochs = max(1, log_every_n_epochs)
        self.num_samples = max(1, min(num_samples, 4))
        print(f"Rank {0}: Initialized AttentionMapCallback with log_every_n_epochs={self.log_every_n_epochs}, num_samples={self.num_samples}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during validation sanity check
        if trainer.sanity_checking:
            print(f"Rank {trainer.global_rank}: Skipping validation sanity check")
            return

        # Only log on main process in DDP
        if trainer.global_rank != 0:
            print(f"Rank {trainer.global_rank}: Skipping logging (non-main process)")
            return

        # Skip if not at logging interval
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            print(f"Rank {trainer.global_rank}: Epoch {trainer.current_epoch} not logged (log_every_n_epochs={self.log_every_n_epochs})")
            return

        print(f"Rank {trainer.global_rank}: Running AttentionMapCallback for epoch {trainer.current_epoch}")

        try:
            # Test Wandb logging
            print(f"Rank {trainer.global_rank}: Testing Wandb logging")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.text(0.5, 0.5, f"Test Image Epoch {trainer.current_epoch}", ha='center', va='center')
            trainer.logger.experiment.log({"Test_Attention": wandb.Image(fig)}, commit=True)
            plt.close(fig)

            # Get validation data
            val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            batch = next(iter(val_loader))
            imgs, _ = batch
            print(f"Rank {trainer.global_rank}: Validation batch shape: {imgs.shape}")
            imgs = imgs[:self.num_samples].to(pl_module.device)

            print(f"Rank {trainer.global_rank}: Processing {self.num_samples} samples for attention visualization at epoch {trainer.current_epoch}")
            print(f"Rank {trainer.global_rank}: Input images shape: {imgs.shape}")

            # Get attention weights
            with torch.no_grad():
                outputs = pl_module(imgs, return_attention=True)

                # Handle both single output and (output, attention) cases
                if isinstance(outputs, tuple):
                    preds, attn_weights = outputs
                else:
                    raise ValueError("Model didn't return attention weights despite return_attention=True")

                # Explicit check for attention weights
                if attn_weights is None or len(attn_weights) == 0:
                    raise ValueError("No attention weights returned")

                print(f"Rank {trainer.global_rank}: Attention weights shape: {attn_weights.shape}")

            # Process samples
            for idx in range(min(self.num_samples, len(imgs))):
                print(f"Rank {trainer.global_rank}: Creating attention figure for sample {idx}")
                fig = self._create_attention_figure(
                    trainer=trainer,
                    image=imgs[idx],
                    attn_weights=attn_weights[:, idx] if attn_weights.dim() == 5 else attn_weights,  # Handle [layers, batch, num_heads, seq_len, seq_len]
                    sample_idx=idx,
                    epoch=trainer.current_epoch,
                    patch_size=getattr(pl_module.model, 'patch_size', 8)  # Default to 8
                )
                trainer.logger.experiment.log({
                    f"Attention/E{trainer.current_epoch}_S{idx}": wandb.Image(fig)
                }, commit=False)
                plt.close(fig)

            # Commit all logged images at once
            trainer.logger.experiment.log({}, commit=True)
            print(f"Rank {trainer.global_rank}: Successfully logged {self.num_samples} attention maps for epoch {trainer.current_epoch}")

        except Exception as e:
            print(f"Rank {trainer.global_rank}: ⚠️ AttentionMapCallback error: {str(e)}")
            fig = self._create_error_figure()
            trainer.logger.experiment.log({
                f"Attention/Error_E{trainer.current_epoch}": wandb.Image(fig)
            }, commit=True)
            plt.close(fig)

class AttentionMapCallback(Callback):
    def __init__(self, log_every_n_epochs=1, num_samples=4):
        super().__init__()
        self.log_every_n_epochs = max(1, log_every_n_epochs)
        self.num_samples = max(1, min(num_samples, 4))
        print(f"Initialized AttentionMapCallback with log_every_n_epochs={self.log_every_n_epochs}, num_samples={self.num_samples}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during validation sanity check or if not logging epoch
        if trainer.sanity_checking or trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        # Only log on main process in DDP
        if trainer.global_rank != 0:
            return

        try:
            # Get validation data
            val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
            batch = next(iter(val_loader))
            imgs, _ = batch
            imgs = imgs[:self.num_samples].to(pl_module.device)

            # Handle sequence inputs - use first frame for visualization
            if imgs.ndim == 5:  # [B, T, H, W, C]
                print(f"Processing sequence input, using first frame for attention visualization")
                imgs = imgs[:, 0]  # Take first frame [B, H, W, C]

            # Get attention weights
            with torch.no_grad():
                outputs = pl_module(imgs, return_attention=True)
                if isinstance(outputs, tuple):
                    preds, attn_weights = outputs
                else:
                    raise ValueError("Model didn't return attention weights despite return_attention=True")

            # Process samples
            for idx in range(min(self.num_samples, len(imgs))):
                fig = self._create_attention_figure(
                    image=imgs[idx],
                    attn_weights=attn_weights[:, idx] if attn_weights.dim() == 5 else attn_weights,
                    sample_idx=idx,
                    epoch=trainer.current_epoch,
                    patch_size=getattr(pl_module.model, 'patch_size', 8)
                )
                trainer.logger.experiment.log({
                    f"Attention/E{trainer.current_epoch}_S{idx}": wandb.Image(fig)
                }, commit=False)
                plt.close(fig)

            # Commit all logged images
            trainer.logger.experiment.log({}, commit=True)

        except Exception as e:
            print(f"⚠️ AttentionMapCallback error: {str(e)}")
            fig = self._create_error_figure()
            trainer.logger.experiment.log({
                f"Attention/Error_E{trainer.current_epoch}": wandb.Image(fig)
            }, commit=True)
            plt.close(fig)

    def _create_attention_figure(self, image, attn_weights, sample_idx, epoch, patch_size):
        """Create attention visualization figure"""
        try:
            # Convert image to numpy and handle different formats
            img_np = image.detach().cpu().numpy()
            if img_np.shape[0] in {1, 3, 6}:  # CHW format
                img_np = np.transpose(img_np, (1, 2, 0))  # HWC

            # Create RGB composite
            rgb_img = np.zeros((img_np.shape[0], img_np.shape[1], 3))
            for i, ch in enumerate([0, 2, 4]):  # R,G,B channels
                if ch < img_np.shape[-1]:
                    channel = img_np[..., ch]
                    rgb_img[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

            # Process attention weights
            last_layer_attn = attn_weights[-1].mean(0) if attn_weights.dim() == 4 else attn_weights[-1]
            cls_attn = last_layer_attn[0, 1:] if last_layer_attn.shape[0] > 1 else last_layer_attn[1:]

            # Reshape and resize attention
            grid_size = int(np.sqrt(cls_attn.shape[0]))
            attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
            zoom_factor = img_np.shape[0] / grid_size
            attn_resized = zoom(attn_map, (zoom_factor, zoom_factor), order=1)
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plt.suptitle(f"Epoch {epoch} - Sample {sample_idx}", y=1.05)

            # Panel 1: Original Image
            axes[0].imshow(rgb_img)
            axes[0].set_title('AIA Composite')
            axes[0].axis('off')

            # Panel 2: Attention Heatmap
            axes[1].imshow(attn_resized, cmap='hot')
            axes[1].set_title('Attention Heatmap')
            axes[1].axis('off')

            # Panel 3: Overlay
            axes[2].imshow(rgb_img)
            axes[2].imshow(attn_resized, cmap='hot', alpha=0.4)
            axes[2].set_title('Attention Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"⚠️ Error in _create_attention_figure: {str(e)}")
            return self._create_error_figure()

    def _create_error_figure(self):
        """Fallback figure when errors occur"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Visualization Error\nCheck Logs",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        return fig

class ImagePredictionLogger_SXR(Callback):
    """Modified to handle both single predictions and sequences"""

    def __init__(self, data_samples, sxr_norm, log_every_n_epochs=1):
        super().__init__()
        self.data_samples = data_samples
        self.sxr_norm = sxr_norm
        self.log_every_n_epochs = log_every_n_epochs
        self.epoch_counter = 0

    def denormalize_sxr(self, normalized):
        """Convert normalized SXR back to original scale"""
        return 10 ** (normalized * self.sxr_norm[1] + self.sxr_norm[0])

    def _create_rgb_composite(self, img_np):
        """Convert 6-channel AIA image to 3-channel RGB composite"""
        # Select channels for RGB (adjust these indices based on your channel order)
        r_channel = img_np[..., 0]  # 94Å
        g_channel = img_np[..., 2]  # 193Å
        b_channel = img_np[..., 4]  # 335Å

        # Normalize each channel
        r_norm = (r_channel - r_channel.min()) / (r_channel.max() - r_channel.min() + 1e-8)
        g_norm = (g_channel - g_channel.min()) / (g_channel.max() - g_channel.min() + 1e-8)
        b_norm = (b_channel - b_channel.min()) / (b_channel.max() - b_channel.min() + 1e-8)

        # Stack into RGB image
        rgb_img = np.stack([r_norm, g_norm, b_norm], axis=-1)
        return rgb_img

    def on_validation_epoch_end(self, trainer, pl_module):
        self.epoch_counter += 1
        if self.epoch_counter % self.log_every_n_epochs != 0:
            return

        # Get a batch of sample data
        sample_imgs, sample_targets = zip(*self.data_samples)
        sample_imgs = torch.stack(sample_imgs).to(pl_module.device)
        sample_targets = torch.stack(sample_targets).to(pl_module.device)

        # Get predictions
        with torch.no_grad():
            predictions = pl_module(sample_imgs)
            if isinstance(predictions, tuple):  # Handle attention case
                predictions = predictions[0]

        # Handle sequence vs single prediction
        if sample_imgs.ndim == 5:  # Sequence input [B, T, H, W, C]
            # For sequences, we'll log the first and last frames
            sample_imgs = sample_imgs[:, 0]  # First frame
            if sample_targets.ndim == 2:  # Sequence target
                sample_targets = sample_targets[:, -1]  # Last target
                predictions = predictions[:, -1]  # Last prediction

        # Denormalize values
        denorm_targets = self.denormalize_sxr(sample_targets.cpu().numpy())
        denorm_preds = self.denormalize_sxr(predictions.cpu().numpy())

        # Log to wandb
        logged_images = []
        for img, tgt, pred in zip(sample_imgs, denorm_targets, denorm_preds):
            # Convert to numpy and handle different formats
            img_np = img.cpu().numpy()
            if img_np.shape[0] in {1, 3, 6}:  # CHW format
                img_np = np.transpose(img_np, (1, 2, 0))  # HWC

            # Create RGB composite
            rgb_img = self._create_rgb_composite(img_np)

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_img)
            ax.set_title(f"Target: {tgt:.2f}, Pred: {pred:.2f}")
            ax.axis('off')
            plt.tight_layout()

            # Log the figure
            logged_images.append(wandb.Image(fig, caption=f"Target: {tgt:.2f}, Pred: {pred:.2f}"))
            plt.close(fig)

        trainer.logger.experiment.log({"val/examples": logged_images})
    def _create_comparison_plot(self, true, pred):
        """Enhanced comparison plot with error bars"""
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(true))

        # Plot with error ranges
        ax.errorbar(x, true, yerr=0.1*true, fmt='o', label='True', capsize=5)
        ax.errorbar(x, pred, yerr=0.1*pred, fmt='s', label='Predicted', capsize=5)

        ax.set_yscale('log')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('SXR Flux (W/m²)')
        ax.set_title('True vs Predicted SXR Flux')
        ax.legend()
        ax.grid(True, which='both', alpha=0.5)
        plt.tight_layout()
        return fig

    def _create_error_plot(self, true, pred):
        """Enhanced error plot with relative errors"""
        fig, ax = plt.subplots(figsize=(12, 6))
        errors = 100 * (pred - true) / (true + 1e-8)  # Percentage error

        ax.bar(range(len(errors)), errors, color=['r' if e > 0 else 'g' for e in errors])
        ax.axhline(0, color='k', linestyle='--')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Prediction Error (%)')
        ax.set_title('SXR Prediction Errors')
        ax.grid(True, axis='y', alpha=0.5)
        plt.tight_layout()
        return fig