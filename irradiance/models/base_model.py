import sys
import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import HuberLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2
from irradiance.utilities.temperature_response import TemperatureResponse

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

        rae = torch.abs((y - y_pred) / (torch.abs(y))) * 100
        av_rae = rae.mean(rae[torch.isfinite(rae)])
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_sp", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE_sp", av_rae, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y, self.eve_norm) 
        y_pred = self.unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        rae = torch.abs((y - y_pred) / (torch.abs(y))) * 100
        av_rae = rae.mean(rae[torch.isfinite(rae)])
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        self.log("valid_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss_sp", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE_sp", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE_sp", av_rae, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred, y)

        y = self.unnormalize(y, self.eve_norm) 
        y_pred = self.unnormalize(y_pred, self.eve_norm)

        #computing relative absolute error
        rae = torch.abs((y - y_pred) / (torch.abs(y))) * 100
        av_rae = rae.mean(rae[torch.isfinite(rae)])
        av_rae_wl = rae.mean(0)
        # compute average cross-correlation
        cc = torch.tensor([torch.corrcoef(torch.stack([y[i], y_pred[i]]))[0, 1] for i in range(y.shape[0])]).mean()
        # mean absolute error
        mae = torch.abs(y - y_pred).mean()

        # self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss_sp", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_MAE_sp", mae, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_RAE_sp", av_rae, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_correlation_coefficient", cc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def unnormalize(self, y, eve_norm):
        eve_norm = torch.tensor(eve_norm).float()
        norm_mean = eve_norm[0]
        norm_stdev = eve_norm[1]
        y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
        return y
    


class BaseDEMModel(BaseModel):

    def __init__(self, 
                 eve_norm, 
                 uv_norm, 
                 model, 
                 t_query_points, 
                 wavelengths: list, 
                 loss_func=HuberLoss(), 
                 lr=1e-4, 
                 kanfov=3,
                 stride=None,
                 base_temp_exponent=0,
                 intensity_factor=1e25):
        super().__init__(eve_norm=eve_norm, model=model, loss_func=loss_func, lr=lr)
        if stride is None:
            self.stride = kanfov
        else:
            self.stride = stride
        self.uv_norm = torch.mean(torch.Tensor(uv_norm['mean'][0:len(wavelengths)])) # TODO: Make sure the means match the wavelengths
        self.uv_norm_wl = uv_norm # TODO: Make sure the means match the wavelengths
        self.t_query_points = t_query_points
        self.kanfov = kanfov
        self.temp_resp = TemperatureResponse()
        self.wavelengths = wavelengths
        self.base_temp_exponent = base_temp_exponent
        self.intensity_factor = intensity_factor
        self.calibration = nn.Parameter(torch.Tensor([1.0]))

    
    def intensity_calculation(self, x):
        # Create tiles
        x = x.unfold(2, self.kanfov, self.stride).unfold(3, self.kanfov, self.stride) # batch, channel, pixel_x, pixel_y, kanfov, kanfov
        
        # Store center pixels
        y = x[:, :, :, :, self.kanfov // 2, self.kanfov // 2] # central pixel --> batch, channel, pixel_x, pixel_y
        y = y.reshape(y.shape[0], y.shape[1], -1) # batch, channel, center_pixel_x*center_pixel_y
        y = y.transpose(1, 2) # batch, center_pixel_x*center_pixel_y, channel

        # Reshape input
        x = x.reshape(x.shape[0], x.shape[1], -1, self.kanfov, self.kanfov) # batch, channel, pixel_x*pixel_y, kanfov, kanfov
        x = x.transpose(1, 2) # batch, pixel_x*pixel_y, channel, kanfov, kanfov
        x = x.reshape(x.shape[0], x.shape[1], -1) # batch, pixelx_*pixel_y, channel*kanfov*kanfov
        x = x[:, :, None, :].expand((x.shape[0], x.shape[1], self.t_query_points.shape[0], x.shape[2])) # batch, pixel_x*pixel_y, t_query_points, channel*kanfov*kanfov
        
        # Normalize input
        x = x/self.uv_norm

        # Concatenate log10(T) subtracting base exponent
        x = torch.cat((x, (self.t_query_points-self.base_temp_exponent)[None, None, :, None].expand(x.shape[0], x.shape[1], x.shape[2], 1)), dim=3)

        # Get DEM
        dem = self(x).squeeze() # batch, pixel_x*pixel_y, t_query_points
        dem = dem[:,:,:,None].expand((dem.shape[0], dem.shape[1], dem.shape[2], y.shape[2])) # batch, channel, t_query_points
        
        # Calculate temperature response function
        t_resp = torch.zeros(self.t_query_points.shape[0], len(self.wavelengths), device=dem.device) # channel, t_query_points
        for i, wl in enumerate(self.wavelengths):
            t_resp[:, i] = self.temp_resp.response[wl]['interpolator'](self.t_query_points)

        # Expand temperature response function
        t_resp = t_resp[None, None, :, :].expand((dem.shape[0], dem.shape[1], dem.shape[2], dem.shape[3])) # batch, channel, t_query_points
        
        # Integrate temperature response function and DEM to get itensity
        intensity = self.calibration*self.intensity_factor*torch.trapezoid(dem * t_resp, x=torch.pow(10, self.t_query_points), dim=2) # batch, channel
        
        return intensity, dem, y

    
    def training_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x)
        
        # Compare with target
        loss = self.loss_func(intensity, intensity_target)

        rae_dem = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE_dem", torch.mean(rae_dem[torch.isfinite(rae_dem)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_dem", loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x)
        
        # Compare with target
        loss = self.loss_func(intensity, intensity_target)

        #computing relative absolute error
        rae = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        av_rae = torch.mean(rae[torch.isfinite(rae)])
        # compute average cross-correlation
        cc = torch.corrcoef(torch.stack([intensity_target.reshape(-1), intensity.reshape(-1)]))[0, 1]
        # mean absolute error
        mae = torch.abs(intensity_target - intensity).mean()

        self.log("valid_loss", on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE_dem", mae, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE_dem", torch.mean(rae[torch.isfinite(rae)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss_dem", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x)
        
        # Compare with target
        loss = self.loss_func(intensity, intensity_target)

        #computing relative absolute error
        rae = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        av_rae = torch.mean(rae[torch.isfinite(rae)])
        # compute average cross-correlation
        cc = torch.corrcoef(torch.stack([intensity_target.reshape(-1), intensity.reshape(-1)]))[0, 1]
        # mean absolute error
        mae = torch.abs(intensity_target - intensity).mean()

        # self.log("test_loss", loss + loss_dem_negative, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_MAE_dem", mae, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_RAE_dem", torch.mean(rae[torch.isfinite(rae)]), on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss_dem", loss, on_epoch=True, prog_bar=True, logger=True)
    
        return loss
    
