import wandb
from pytorch_lightning import Callback
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import sunpy.visualization.colormaps as cm
import astropy.units as u

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
        for (aia, _), target in self.data_samples:
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
