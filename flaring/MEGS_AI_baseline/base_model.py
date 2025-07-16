import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    def __init__(self, model, loss_func, lr):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                    verbose=True),
            'monitor': 'val_loss',  # name of the metric to monitor
            'interval': 'epoch',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_func(torch.squeeze(pred), target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss_func(torch.squeeze(pred), target)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        (x, sxr), target = batch
        pred = self(x)
        loss = self.loss_func(torch.squeeze(pred), target)
        self.log('test_loss', loss)
        return loss