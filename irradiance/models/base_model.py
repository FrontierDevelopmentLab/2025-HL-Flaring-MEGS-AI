import sys
import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import HuberLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

class BaseModel(LightningModule):

    def __init__(self, eve_norm, model, loss_func=HuberLoss(), lr=1e-4):
        super().__init__()
        self.eve_norm = eve_norm
        self.loss_func = loss_func
        self.model = model
        self.lr = lr

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented, please implement it in the child class.")

    def forward_unnormalize(self, x):
        x = self.forward(x)
        return self.unnormalize(x, self.eve_norm)
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y, self.eve_norm)
        y_pred = self.unnormalize(y_pred, self.eve_norm)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y, self.eve_norm) 
        y_pred = self.unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y, self.eve_norm) 
        y_pred = self.unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def unnormalize(self, y, eve_norm):
        eve_norm = torch.tensor(eve_norm).float()
        norm_mean = eve_norm[0]
        norm_stdev = eve_norm[1]
        y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
        return y
    


class BaseDEMModel(LightningModule):

    def __init__(self, eve_norm, uv_norm, model, t_query_points, loss_func=HuberLoss(), lr=1e-4):
        super().__init__()
        self.eve_norm = eve_norm
        self.uv_norm = uv_norm
        self.loss_func = loss_func
        self.model = model
        self.lr = lr
        self.t_query_points = t_query_points

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented, please implement it in the child class.")

    def forward_unnormalize(self, x):
        x = self.forward(x)
        return self.unnormalize(x, self.eve_norm)
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).reshape(x.shape[0], x.shape[1], -1, 3, 3) # channel, pixels, 3x3
        y = x[:, :, :, 1, 1] # central pixel
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE", rae.mean(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).reshape(x.shape[0], x.shape[1], -1, 3, 3) # channel, pixels, 3x3
        y = x[:, :, :, 1, 1] # central pixel
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).reshape(x.shape[0], x.shape[1], -1, 3, 3) # channel, pixels, 3x3
        y = x[:, :, :, 1, 1] # central pixel
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        #computing relative absolute error
        epsilon = sys.float_info.min
        rae = torch.abs((y - y_pred) / (torch.abs(y) + epsilon)) * 100
        av_rae = rae.mean()
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_MAE", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_RAE", av_rae, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss