# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import *
from torch.nn import HuberLoss
from irradiance.models.base_model import BaseModel, BaseDEMModel
from pytorch_optimizer import create_optimizer


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKANIrradiance(BaseModel):
    def __init__(
        self,
        eve_norm,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        loss_func = HuberLoss(),
        lr=1e-4,
        use_std=False   
    ) -> None:
        super().__init__(model=None, eve_norm=eve_norm, loss_func=loss_func, lr=lr)
        self.use_std = use_std
        if use_std:
            layers_hidden[0] = layers_hidden[0]*2
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        # Calculating mean and std of images to take them as input to 1D KAN
        mean_irradiance = torch.torch.mean(x, dim=(2,3))
        std_irradiance = torch.torch.std(x, dim=(2,3))
        if self.use_std:
            x = torch.cat((mean_irradiance, std_irradiance), dim=1)
        else:
            x = mean_irradiance
        for layer in self.layers:
            x = layer(x)
        return x
    

class KANDEM(BaseDEMModel):
    def __init__(
        self,
        eve_norm,
        uv_norm,
        wavelengths,
        t_query_points,
        kanfov,
        layers_hidden: List[int],
        grid_min: List[float],
        grid_max: List[float],
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        loss_func = HuberLoss(),
        lr=1e-4,
        base_temp_exponent=0,
        intensity_factor=1e25,
        stride = None
    ) -> None:
        super().__init__(eve_norm=eve_norm, 
                         uv_norm=uv_norm, 
                         wavelengths=wavelengths, 
                         kanfov=kanfov, 
                         model=None, 
                         t_query_points=t_query_points, 
                         loss_func=loss_func, 
                         lr=lr,
                         base_temp_exponent=base_temp_exponent,
                         intensity_factor = intensity_factor,
                         stride=stride)
        
        self.save_hyperparameters()

        # specify the KAN model
        self.layers = nn.ModuleList([
                FastKANLayer(
                    in_dim, out_dim,
                    grid_min=grid_min_l,
                    grid_max=grid_max_l,
                    num_grids=num_grids,
                    use_base_update=use_base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_weight_init_scale,
                ) for in_dim, out_dim, grid_min_l, grid_max_l in zip(layers_hidden[:-1], layers_hidden[1:], grid_min, grid_max)
            ])
        
 
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class KANDEMSpectrum(BaseDEMModel):
    def __init__(
        self,
        eve_norm,
        uv_norm,
        wavelengths,
        t_query_points,
        kanfov,
        layers_hidden_dem: List[int],
        layers_hidden_sp: List[int],
        grid_min_dem: List[float],
        grid_min_sp: List[float],
        grid_max_dem: List[float],
        grid_max_sp: List[float],
        num_grids_dem: int = 8,
        num_grids_sp: int = 8,
        use_base_update_dem: bool = True,
        use_base_update_sp: bool = True,
        spline_weight_init_scale_dem: float = 0.1,
        spline_weight_init_scale_sp: float = 0.1,
        use_layernorm = True,
        base_activation = F.silu,
        base_temp_exponent=0,
        intensity_factor=1e25,
        lr=1e-4,
        loss_func = HuberLoss(),
        stride=None,
        log_sploss_factor = 1.0,
        lin_sploss_factor = 1.0,
        hybrid = False
    ) -> None:
        super().__init__(eve_norm=eve_norm, 
                         uv_norm=uv_norm, 
                         wavelengths=wavelengths, 
                         kanfov=kanfov, 
                         model=None, 
                         t_query_points=t_query_points, 
                         loss_func=loss_func, 
                         lr=lr,
                         base_temp_exponent=base_temp_exponent,
                         intensity_factor = intensity_factor,
                         stride=stride)
        self.lr = lr
        self.log_sploss_factor = log_sploss_factor
        self.lin_sploss_factor  = lin_sploss_factor
        self.hybrid = hybrid
        self.save_hyperparameters()

        # specify the DEM KAN model
        self.dem_layers = nn.ModuleList([
                FastKANLayer(
                    in_dim, out_dim,
                    grid_min=grid_min_l,
                    grid_max=grid_max_l,
                    num_grids=num_grids_dem,
                    use_base_update=use_base_update_dem,
                    base_activation=base_activation,
                    use_layernorm=use_layernorm,
                    spline_weight_init_scale=spline_weight_init_scale_dem,
                ) for in_dim, out_dim, grid_min_l, grid_max_l in zip(layers_hidden_dem[:-1], layers_hidden_dem[1:], grid_min_dem, grid_max_dem)
            ])
        
        # specif the spectrum KAN model
        self.spectrum_layers = nn.ModuleList([
                FastKANLayer(
                    in_dim, out_dim,
                    grid_min=grid_min_l,
                    grid_max=grid_max_l,
                    num_grids=num_grids_sp,
                    use_base_update=use_base_update_sp,
                    base_activation=base_activation,
                    use_layernorm=use_layernorm,
                    spline_weight_init_scale=spline_weight_init_scale_sp,
                ) for in_dim, out_dim, grid_min_l, grid_max_l in zip(layers_hidden_sp[:-1], layers_hidden_sp[1:], grid_min_sp, grid_max_sp)
            ])
        
        if hybrid:
            linear_modl = nn.Linear(2*len(wavelengths), layers_hidden_sp[-1])
        
        self.eve_calibration = nn.Parameter(torch.Tensor([1.0]))
    
    
    def forward_unnormalize(self, x):
        intensity, dem, intensity_target = self.intensity_calculation(x) # dem: batch, channel, t_query_points
        dem = dem[:, :, :, 0] # batch, pixels, t_query_points
        spectrum = self.forward_spectrum(dem)
        spectrum = self.eve_calibration*torch.mean(spectrum, dim=1)
        return self.unnormalize(spectrum, self.eve_norm)
    
    def forward(self, x):
        for layer in self.dem_layers:
            x = layer(x)
        return F.relu(x)

    def forward_spectrum(self, x):
        for layer in self.spectrum_layers:
            x = layer(x)
        return x
    
    def forward_linear(self, x):
        mean_irradiance = torch.torch.mean(x, dim=(2,3))
        x = self.model(mean_irradiance)
        return x
    
    def training_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x) # dem: batch, channel, t_query_points
        dem = dem[:, :, :, 0] # batch, pixels, t_query_points
        spectrum = self.forward_spectrum(dem)
        spectrum = self.eve_calibration*torch.mean(spectrum, dim=1)
        if self.hybrid:
            spectrum = self.forward_linear(x)
        # Compare with target
        loss_dem = self.loss_func(intensity, intensity_target)
        loss_sp = self.loss_func(spectrum, y)

        # Add log loss
        eps=1e-10
        loss_log_sp = self.loss_func(torch.log(F.relu(self.unnormalize(spectrum, self.eve_norm))+eps), torch.log(F.relu(self.unnormalize(y, self.eve_norm))+eps))

        rae_dem = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        rae_sp = torch.abs((y - spectrum) / (torch.abs(y))) * 100
        self.log("train_loss_dem", loss_dem, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE_dem", torch.mean(rae_dem[torch.isfinite(rae_dem)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss_sp", loss_sp, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_RAE_sp", torch.mean(rae_sp[torch.isfinite(rae_sp)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("train_log_loss_sp", loss_log_sp, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp, on_epoch=True, prog_bar=True, logger=True)

        return loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp

    
    def validation_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x)
        dem = dem[:, :, :, 0] # batch, pixels, t_query_points
        spectrum = self.forward_spectrum(dem)
        spectrum = self.eve_calibration*torch.mean(spectrum, dim=1)
        if self.hybrid:
            spectrum = self.forward_linear(x)
        # Compare with target
        loss_dem = self.loss_func(intensity, intensity_target)
        loss_sp = self.loss_func(spectrum, y)

        # Add log loss
        eps=1e-10
        loss_log_sp = self.loss_func(torch.log(F.relu(self.unnormalize(spectrum, self.eve_norm))+eps), torch.log(F.relu(self.unnormalize(y, self.eve_norm))+eps))

        rae_dem = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        rae_sp = torch.abs((y - spectrum) / (torch.abs(y))) * 100
        mae_dem = torch.abs(intensity_target - intensity).mean()
        mae_sp = torch.abs(y - spectrum).mean()
        self.log("valid_loss_dem", loss_dem, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE_dem", torch.mean(rae_dem[torch.isfinite(rae_dem)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss_sp", loss_sp, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_RAE_sp", torch.mean(rae_sp[torch.isfinite(rae_sp)]), on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE_dem", mae_dem, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_MAE_sp", mae_sp, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_log_loss_sp", loss_log_sp, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp, on_epoch=True, prog_bar=True, logger=True)

        return loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp
    
    
    def test_step(self, batch, batch_nb):
        x, y = batch

        intensity, dem, intensity_target = self.intensity_calculation(x)
        dem = dem[:, :, :, 0] # batch, pixels, t_query_points
        spectrum = self.forward_spectrum(dem)
        spectrum = self.eve_calibration*torch.mean(spectrum, dim=1)
        if self.hybrid:
            spectrum = self.forward_linear(x)
        # Compare with target
        loss_dem = self.loss_func(intensity, intensity_target)
        loss_sp = self.loss_func(spectrum, y)
        # Add log loss
        eps=1e-10
        loss_log_sp = self.loss_func(torch.log(F.relu(self.unnormalize(spectrum, self.eve_norm))+eps), torch.log(F.relu(self.unnormalize(y, self.eve_norm))+eps))

        rae_dem = torch.abs((intensity_target - intensity) / (torch.abs(intensity_target))) * 100
        rae_sp = torch.abs((y - spectrum) / (torch.abs(y))) * 100
        mae_dem = torch.abs(intensity_target - intensity).mean()
        mae_sp = torch.abs(y - spectrum).mean()
        # self.log("test_loss_dem", loss_dem, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_RAE_dem", torch.mean(rae_dem[torch.isfinite(rae_dem)]), on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss_sp", loss_sp, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_RAE_sp", torch.mean(rae_sp[torch.isfinite(rae_sp)]), on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_MAE_dem", mae_dem, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_MAE_sp", mae_sp, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_loss", loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp, on_epoch=True, prog_bar=True, logger=True)

        return loss_dem + self.lin_sploss_factor*loss_sp + self.log_sploss_factor*loss_log_sp
    

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self,
            'adamp',
            lr=self.lr,
            use_gc=True,
            use_lookahead=True,
        )
        return optimizer