import torch
from torch import nn
from torch.nn import HuberLoss
from irradiance.models.base_model import BaseModel
from irradiance.models.efficientnet import EfficientnetIrradiance
from irradiance.models.chopped_alexnet import ChoppedAlexnet
class LinearIrradianceModel(BaseModel):

    def __init__(self, d_input, d_output, eve_norm, loss_func= HuberLoss(), lr=1e-2, use_std=True):

        self.n_channels = d_input
        self.outSize = d_output
        self.use_std = use_std
        if use_std:
            model = nn.Linear(2*self.n_channels, self.outSize)
        else:
            model = nn.Linear(self.n_channels, self.outSize)
        super().__init__(model=model, eve_norm=eve_norm, loss_func=loss_func, lr=lr)

    def forward(self, x):
        mean_irradiance = torch.torch.mean(x, dim=(2,3))
        std_irradiance = torch.torch.std(x, dim=(2,3))
        if self.use_std:
            x = self.model(torch.cat((mean_irradiance, std_irradiance), dim=1))
        else:
            x = self.model(mean_irradiance)
        return x
    

class HybridIrradianceModel(BaseModel):

    def __init__(self, 
                 d_input, 
                 d_output, 
                 eve_norm, 
                 cnn_model='resnet', 
                 ln_model=True, 
                 ln_params=None, 
                 lr=1e-4, 
                 cnn_dp=0.75, 
                 loss_func= HuberLoss()):
        super().__init__(model=None, eve_norm=eve_norm, loss_func=loss_func, lr=lr)
        self.n_channels = d_input
        self.outSize = d_output
        self.ln_params = ln_params  
        self.lr = lr      

        # Linear model
        self.ln_model = None      
        if ln_model:
            self.ln_model = EfficientnetIrradiance(d_input, d_output, eve_norm)
        if self.ln_params is not None:
            self.ln_model.weight = torch.nn.Parameter(self.ln_params['weight'])
            self.ln_model.bias = torch.nn.Parameter(self.ln_params['bias'])
        
        # CNN model
        efficientnets = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                          'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']
        self.cnn_model = None
        self.cnn_lambda = 1.
        if cnn_model == 'resnet':
            self.cnn_model = ChoppedAlexnet(d_input, d_output, eve_norm)
        elif cnn_model in efficientnets:
            self.cnn_model = EfficientnetIrradiance(d_input, d_output, eve_norm, model=cnn_model, dp=cnn_dp)

        # Error
        if self.ln_model is None and self.cnn_model is None:
            raise ValueError('Please pass at least one model.')



    def forward(self, x):
        # Hybrid model
        if self.ln_model is not None and self.cnn_model is not None:
            return self.ln_model.forward(x) + self.cnn_lambda * self.cnn_model.forward(x)
        # Linear model only
        elif self.ln_model is not None:
            return self.ln_model.forward(x)
        # CNN model only
        elif self.cnn_model is not None:
            return self.cnn_model.forward(x)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_train_mode(self, mode):
        if mode == 'linear':
            self.cnn_lambda = 0
            self.cnn_model.freeze()
            self.ln_model.unfreeze()
        elif mode == 'cnn':
            self.cnn_lambda = 0.01
            self.cnn_model.unfreeze()
            self.ln_model.freeze()
        elif mode == 'both':
            self.cnn_lambda = 0.01
            self.cnn_model.unfreeze()
            self.ln_model.unfreeze()
        else:
            raise NotImplemented(f'Mode not supported: {mode}')
    