import torch
import torch.nn as nn
from torch.nn import HuberLoss
from models.base_model import BaseModel

class LinearIrradianceModel(BaseModel):
    def __init__(self, d_input, d_output, eve_norm, loss_func=HuberLoss(), lr=1e-2):
        self.n_channels = d_input
        self.outSize = d_output
        model = nn.Linear(2 * self.n_channels, self.outSize)
        super().__init__(model=model, eve_norm=eve_norm, loss_func=loss_func, lr=lr)

    def forward(self, x, sxr=None, **kwargs):
        # If x is a tuple (aia_img, sxr_val), extract the AIA image tensor
        if isinstance(x, (list, tuple)):
            x = x[0]

        # Debug: Print input shape
        print(f"Input shape to LinearIrradianceModel.forward: {x.shape}")

        # Expect x shape: (batch_size, H, W, C)
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch_size, H, W, C), got shape {x.shape}")
        if x.shape[-1] != self.n_channels:
            raise ValueError(f"AIA image has {x.shape[-1]} channels, expected {self.n_channels}")

        # Calculate mean and std across spatial dimensions (H,W)
        # First permute to (batch_size, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Now calculate mean/std across dimensions 2 and 3 (H,W)
        mean_irradiance = torch.mean(x, dim=(2, 3))  # Shape: (batch_size, n_channels)
        std_irradiance = torch.std(x, dim=(2, 3))    # Shape: (batch_size, n_channels)

        # Debug: Print shapes after mean and std
        print(f"mean_irradiance shape: {mean_irradiance.shape}, std_irradiance shape: {std_irradiance.shape}")

        input_features = torch.cat((mean_irradiance, std_irradiance), dim=1)  # Shape: (batch_size, 2 * n_channels)
        print(f"Input features shape to linear layer: {input_features.shape}")

        if input_features.shape[1] != 2 * self.n_channels:
            raise ValueError(f"Expected {2 * self.n_channels} features, got {input_features.shape[1]}")

        return self.model(input_features)

class HybridIrradianceModel(BaseModel):
    def __init__(self, d_input, d_output, eve_norm, cnn_model='resnet', ln_model=True, ln_params=None, lr=1e-4, cnn_dp=0.75, loss_func=HuberLoss()):
        super().__init__(model=None, eve_norm=eve_norm, loss_func=loss_func, lr=lr)
        self.n_channels = d_input
        self.outSize = d_output
        self.ln_params = ln_params
        self.ln_model = None
        if ln_model:
            self.ln_model = LinearIrradianceModel(d_input, d_output, eve_norm, loss_func=loss_func, lr=lr)
        if self.ln_params is not None and self.ln_model is not None:
            self.ln_model.model.weight = nn.Parameter(self.ln_params['weight'])
            self.ln_model.model.bias = nn.Parameter(self.ln_params['bias'])
        self.cnn_model = None
        self.cnn_lambda = 1.
        if cnn_model == 'resnet':
            self.cnn_model = nn.Sequential(
                nn.Conv2d(d_input, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, d_output),
                nn.Dropout(cnn_dp)
            )
        elif cnn_model.startswith('efficientnet'):
            raise NotImplementedError("EfficientNet requires timm; replace with custom CNN or install timm")
        if self.ln_model is None and self.cnn_model is None:
            raise ValueError('Please pass at least one model.')

    def forward(self, x, sxr=None, **kwargs):
        # If x is a tuple (aia_img, sxr_val), extract the AIA image tensor
        if isinstance(x, (list, tuple)):
            x = x[0]

        # Debug: Print input shape
        print(f"Input shape to HybridIrradianceModel.forward: {x.shape}")

        # Expect x shape: (batch_size, H, W, C)
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch_size, H, W, C), got shape {x.shape}")
        if x.shape[-1] != self.n_channels:
            raise ValueError(f"AIA image has {x.shape[-1]} channels, expected {self.n_channels}")

        # Convert to (batch_size, C, H, W) for CNN
        x_cnn = x.permute(0, 3, 1, 2)

        if self.ln_model is not None and self.cnn_model is not None:
            # For linear model, keep original (B,H,W,C) format
            return self.ln_model(x) + self.cnn_lambda * self.cnn_model(x_cnn)
        elif self.ln_model is not None:
            return self.ln_model(x)
        elif self.cnn_model is not None:
            return self.cnn_model(x_cnn)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_train_mode(self, mode):
        if mode == 'linear':
            self.cnn_lambda = 0
            if self.cnn_model: self.cnn_model.eval()
            if self.ln_model: self.ln_model.train()
        elif mode == 'cnn':
            self.cnn_lambda = 0.01
            if self.cnn_model: self.cnn_model.train()
            if self.ln_model: self.ln_model.eval()
        elif mode == 'both':
            self.cnn_lambda = 0.01
            if self.cnn_model: self.cnn_model.train()
            if self.ln_model: self.ln_model.train()
        else:
            raise NotImplementedError(f'Mode not supported: {mode}')