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

def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
    return y


class ImagePredictionLogger_SXR(Callback):

    def __init__(self, data_samples, sxr_norm):
        super().__init__()
        self.data_samples = data_samples
        self.val_aia = data_samples[0][0]
        self.val_sxr = data_samples[0][1]
        self.sxr_norm = sxr_norm

    def unnormalize_sxr(self, normalized_values):
        if isinstance(normalized_values, torch.Tensor):
            normalized_values = normalized_values.cpu().numpy()
        normalized_values = np.array(normalized_values, dtype=np.float32)
        return 10 ** (normalized_values * float(self.sxr_norm[1].item()) + float(self.sxr_norm[0].item())) - 1e-8

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

        true_unorm = self.unnormalize_sxr(true_sxr)
        pred_unnorm = self.unnormalize_sxr(pred_sxr)
        print("Aia images:", aia_images)
        print("Sxr images:", true_unorm)
        print("Sxr images:", pred_unnorm)
        fig = self.plot_aia_sxr(aia_images,true_unorm, pred_unnorm)
        trainer.logger.experiment.log({"AIA 94Å Images and Soft X-ray flux plots": wandb.Image(fig)})
        plt.close(fig)

    def plot_aia_sxr(self, val_aia, val_sxr, pred_sxr):
        num_samples = len(val_aia)
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 10))



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


class ImagePredictionLogger(Callback):
    def __init__(self, val_imgs, val_eve, names, aia_wavelengths):
        super().__init__()
        self.val_imgs, self.val_eve = val_imgs, val_eve
        self.names = names
        self.aia_wavelengths = aia_wavelengths

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        # Get model prediction
        # pred_eve = pl_module.forward(val_imgs).cpu().numpy()
        pred_eve = pl_module.forward_unnormalize(val_imgs).cpu().numpy()
        val_eve = unnormalize(self.val_eve, pl_module.eve_norm).numpy()
        val_imgs = val_imgs.cpu().numpy()

        # create matplotlib figure
        fig = self.plot_aia_eve(val_imgs, val_eve, pred_eve)
        # Log the images to wandb
        trainer.logger.experiment.log({"AIA Images and EVE bar plots": wandb.Image(fig)})
        plt.close(fig)

    def plot_aia_eve(self, val_imgs, val_eve, pred_eve):
        """
        Function to plot a 4 channel AIA stack and the EVE barplots

        Arguments:
        ----------
            val_imgs: numpy array
                Stack with 4 image channels
            val_eve: numpy array
                Stack of ground-truth eve channels
            pred_eve: numpy array
                Stack of predicted eve channels
        Returns:
        --------
            fig: matplotlib figure
                figure with plots
        """
        samples = pred_eve.shape[0]
        n_aia_wavelengths = len(self.aia_wavelengths)
        wspace = 0.2
        hspace = 0.125
        dpi = 100

        if n_aia_wavelengths < 3:
            nrows = 1
            ncols = n_aia_wavelengths
            fig = plt.figure(figsize=(9 + 9 / 4 * n_aia_wavelengths, 3 * samples), dpi=dpi)
            gs = fig.add_gridspec(samples, n_aia_wavelengths + 3, wspace=wspace, hspace=hspace)
        elif n_aia_wavelengths < 5:
            nrows = 2
            ncols = 2
            fig = plt.figure(figsize=(9 + 9 / 4 * 2, 6 * samples), dpi=dpi)
            gs = fig.add_gridspec(2 * samples, 5, wspace=wspace, hspace=hspace)
        elif n_aia_wavelengths < 7:
            nrows = 2
            ncols = 3
            fig = plt.figure(figsize=(9 + 9 / 4 * 3, 6 * samples), dpi=dpi)
            gs = fig.add_gridspec(2 * samples, 6, wspace=wspace, hspace=hspace)
        else:
            nrows = 2
            ncols = 4
            fig = plt.figure(figsize=(15, 5 * samples), dpi=dpi)
            gs = fig.add_gridspec(2 * samples, 7, wspace=wspace, hspace=hspace)

        cmaps_all = ['sdoaia94', 'sdoaia131', 'sdoaia171', 'sdoaia193', 'sdoaia211',
                     'sdoaia304', 'sdoaia335', 'sdoaia1600', 'sdoaia1700']
        cmaps = [cmaps_all[i] for i in self.aia_wavelengths]
        n_plots = 0

        for s in range(samples):
            for i in range(nrows):
                for j in range(ncols):
                    if n_plots < n_aia_wavelengths:
                        ax = fig.add_subplot(gs[s * nrows + i, j])
                        ax.imshow(val_imgs[s, i * ncols + j], cmap=plt.get_cmap(cmaps[i * ncols + j]), origin='lower')
                        ax.text(0.01, 0.99, cmaps[i * ncols + j], horizontalalignment='left', verticalalignment='top',
                                color='w', transform=ax.transAxes)
                        ax.set_axis_off()
                        n_plots += 1
            n_plots = 0
            # eve data
            ax5 = fig.add_subplot(gs[s * nrows, ncols:])
            if self.names is not None:
                ax5.bar(np.arange(0, len(val_eve[s, :])), val_eve[s, :], label='ground truth')
                ax5.bar(np.arange(0, len(pred_eve[s, :])), pred_eve[s, :], width=0.5, label='prediction', alpha=0.5)
                ax5.set_xticks(np.arange(0, len(val_eve[s, :])))
                ax5.set_xticklabels(self.names, rotation=45)
            else:
                ax5.plot(np.arange(0, len(val_eve[s, :])), val_eve[s, :], label='ground truth', alpha=0.5,
                         drawstyle='steps-mid')
                ax5.plot(np.arange(0, len(pred_eve[s, :])), pred_eve[s, :], label='prediction', alpha=0.5,
                         drawstyle='steps-mid')
            ax5.set_yscale('log')
            ax5.legend()

            ax6 = fig.add_subplot(gs[s * nrows + 1, ncols:])
            if self.names is not None:
                ax6.bar(np.arange(0, len(val_eve[s, :])), np.abs(pred_eve[s, :] - val_eve[s, :]) / val_eve[s, :] * 100,
                        label='relative error (%)')
                ax6.set_xticks(np.arange(0, len(val_eve[s, :])))
                ax6.set_xticklabels(self.names, rotation=45)
            else:
                ax6.plot(np.arange(0, len(val_eve[s, :])), np.abs(pred_eve[s, :] - val_eve[s, :]) / val_eve[s, :] * 100,
                         label='relative error (%)', alpha=0.5, drawstyle='steps-mid')
            ax6.set_yscale('log')
            ax6.legend()

        fig.tight_layout()
        return fig


class SpectrumPredictionLogger(ImagePredictionLogger):
    def __init__(self, val_imgs, val_eve, names, aia_wavelengths):
        super().__init__(val_imgs, val_eve, names, aia_wavelengths)

    def plot_aia_eve(self, val_imgs, val_eve, pred_eve):
        """
        Function to plot a 4 channel AIA stack and the EVE barplots

        Arguments:
        ----------
            val_imgs: numpy array
                Stack with 4 image channels
            val_eve: numpy array
                Stack of ground-truth eve channels
            pred_eve: numpy array
                Stack of predicted eve channels
        Returns:
        --------
            fig: matplotlib figure
                figure with plots
        """
        samples = pred_eve.shape[0]
        n_aia_wavelengths = len(self.aia_wavelengths)
        wspace = 0.2
        hspace = 0.125
        dpi = 200

        fig = plt.figure(figsize=(5, 5), dpi=dpi)
        gs = fig.add_gridspec(2, 1, wspace=wspace, hspace=hspace)

        # eve data
        s = 0
        ax5 = fig.add_subplot(gs[0, 0])
        if self.names is not None:
            ax5.bar(np.arange(0, len(val_eve[s, :])), val_eve[s, :], label='ground truth')
            ax5.bar(np.arange(0, len(pred_eve[s, :])), pred_eve[s, :], width=0.5, label='prediction', alpha=0.5)
            ax5.set_xticks(np.arange(0, len(val_eve[s, :])))
            ax5.set_xticklabels(self.names, rotation=45)
        else:
            ax5.plot(np.arange(0, len(val_eve[s, :])), val_eve[s, :], label='ground truth', alpha=0.5,
                     drawstyle='steps-mid')
            ax5.plot(np.arange(0, len(pred_eve[s, :])), pred_eve[s, :], label='prediction', alpha=0.5,
                     drawstyle='steps-mid')
        ax5.set_yscale('log')
        ax5.legend()

        ax6 = fig.add_subplot(gs[1, 0])
        if self.names is not None:
            ax6.bar(np.arange(0, len(val_eve[s, :])), np.abs(pred_eve[s, :] - val_eve[s, :]) / val_eve[s, :] * 100,
                    label='relative error (%)')
            ax6.set_xticks(np.arange(0, len(val_eve[s, :])))
            ax6.set_xticklabels(self.names, rotation=45)
        else:
            ax6.plot(np.arange(0, len(val_eve[s, :])), np.abs(pred_eve[s, :] - val_eve[s, :]) / val_eve[s, :] * 100,
                     label='relative error (%)', alpha=0.5, drawstyle='steps-mid')
        ax6.set_yscale('log')
        ax6.legend()

        fig.tight_layout()
        return fig
