
import wandb
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import zoom

# Set non-interactive backend for matplotlib to avoid display issues on servers
import matplotlib
matplotlib.use('Agg')

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

    def _create_attention_figure(self, trainer, image, attn_weights, sample_idx, epoch, patch_size):
        """Create 3-panel attention visualization"""
        try:
            # Convert and validate image
            img_np = image.detach().cpu().numpy()
            if img_np.ndim == 4:  # [T,C,H,W] sequence
                print(f"Rank {trainer.global_rank}: Sample {sample_idx}: Using first frame from sequence with shape {img_np.shape}")
                img_np = img_np[0]  # Take first frame
            if img_np.shape[0] in {1, 3, 6}:  # CHW format
                img_np = np.transpose(img_np, (1, 2, 0))  # HWC
            print(f"Rank {trainer.global_rank}: Sample {sample_idx}: Image shape after processing: {img_np.shape}")

            # Validate image size
            expected_image_size = 512  # Based on config
            if img_np.shape[0] != expected_image_size or img_np.shape[1] != expected_image_size:
                raise ValueError(f"Image size {img_np.shape[0]}x{img_np.shape[1]} does not match expected {expected_image_size}x{expected_image_size}")

            # Create RGB composite
            rgb_img = np.zeros((img_np.shape[0], img_np.shape[1], 3))
            for i, ch in enumerate([0, 2, 4]):  # R,G,B channels
                if ch < img_np.shape[-1]:
                    channel = img_np[..., ch]
                    rgb_img[..., i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

            # Get attention from last layer and average heads
            last_layer_attn = attn_weights[-1].mean(0) if attn_weights.dim() == 4 else attn_weights[-1]  # [num_heads, seq_len, seq_len] -> [seq_len, seq_len]
            cls_attn = last_layer_attn[0, 1:] if last_layer_attn.shape[0] > 1 else last_layer_attn[1:]  # CLS token attention to patches
            print(f"Rank {trainer.global_rank}: Sample {sample_idx}: CLS attention shape: {cls_attn.shape}")

            # Calculate grid size
            seq_len = cls_attn.shape[0]
            grid_size = int(np.sqrt(seq_len))
            if grid_size * grid_size != seq_len:
                raise ValueError(f"Attention length {seq_len} isn't square (grid_size={grid_size})")

            # Validate patch size
            expected_grid_size = 512 // patch_size
            if grid_size != expected_grid_size:
                raise ValueError(f"Grid size {grid_size} does not match expected {expected_grid_size} for patch_size={patch_size}")

            # Reshape and resize attention
            attn_map = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
            zoom_factor = img_np.shape[0] / grid_size
            attn_resized = zoom(attn_map, (zoom_factor, zoom_factor), order=1)

            # Normalize attention map
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            print(f"Rank {trainer.global_rank}: Sample {sample_idx}: Attention map shape after resize: {attn_resized.shape}")

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
            print(f"Rank {trainer.global_rank}: Sample {sample_idx}: Attention figure created successfully")
            return fig

        except Exception as e:
            print(f"Rank {trainer.global_rank}: ⚠️ Error in _create_attention_figure for sample {sample_idx}: {str(e)}")
            return self._create_error_figure()

    def _create_error_figure(self):
        """Fallback figure when errors occur"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Visualization Error\nCheck Logs",
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        print(f"Rank {trainer.global_rank}: Created error figure due to visualization failure")
        return fig

class ImagePredictionLogger_SXR(Callback):
    """Enhanced SXR logger with debug features"""
    def __init__(self, data_samples, sxr_norm, log_every_n_epochs=1):
        super().__init__()
        self.data_samples = data_samples[:4]  # Limit samples
        self.sxr_norm = sxr_norm
        self.log_every_n_epochs = max(1, log_every_n_epochs)
        self._debug_data = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        try:
            pred_sxr, true_sxr = [], []
            debug_info = {'epoch': trainer.current_epoch}

            for i, (aia, target) in enumerate(self.data_samples):
                with torch.no_grad():
                    pred = pl_module(aia.unsqueeze(0).to(pl_module.device))
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    pred_sxr.append(pred.item())
                    true_sxr.append(target.item())

            # Unnormalize and validate
            true_unorm = 10**(np.array(true_sxr)*self.sxr_norm[1] + self.sxr_norm[0])
            pred_unorm = 10**(np.array(pred_sxr)*self.sxr_norm[1] + self.sxr_norm[0])

            debug_info.update({
                'true_values': true_unorm.tolist(),
                'pred_values': pred_unorm.tolist()
            })

            # Create and log plots
            fig1 = self._create_comparison_plot(true_unorm, pred_unorm)
            fig2 = self._create_error_plot(true_unorm, pred_unorm)

            trainer.logger.experiment.log({
                "SXR_Predictions": wandb.Image(fig1),
                "SXR_Errors": wandb.Image(fig2),
                "sxr_debug": wandb.Table(
                    columns=list(debug_info.keys()),
                    data=[list(debug_info.values())]
                )
            })

            plt.close(fig1)
            plt.close(fig2)
            self._debug_data.append(debug_info)

        except Exception as e:
            print(f"⚠️ SXR Logger error: {str(e)}")

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