
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    def __init__(self, model, eve_norm, loss_func, lr):
        super().__init__()
        self.model = model
        self.eve_norm = eve_norm  # Used for SXR normalization (mean, std)
        self.loss_func = loss_func
        self.lr = lr

    def forward(self, x, sxr=None):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        (x, sxr), target = batch
        pred = self(x, sxr)
        # pred = pred * self.eve_norm[1] + self.eve_norm[0]  # Denormalize for loss
        # target = target * self.eve_norm[1] + self.eve_norm[0]  # Denormalize target
        loss = self.loss_func(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x, sxr), target = batch
        pred = self(x, sxr)
        # pred = pred * self.eve_norm[1] + self.eve_norm[0]
        # target = target * self.eve_norm[1] + self.eve_norm[0]
        loss = self.loss_func(pred, target)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        (x, sxr), target = batch
        pred = self(x, sxr)
        pred = pred * self.eve_norm[1] + self.eve_norm[0]
        target = target * self.eve_norm[1] + self.eve_norm[0]
        loss = self.loss_func(pred, target)
        self.log('test_loss', loss)
        return loss
