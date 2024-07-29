import torch
from torch import nn
from astropy import units as u
from sunpy.io.special import read_genx
from xitorch.interpolate import Interp1D

class TemperatureResponse:

    def __init__(self, temp_resp_path = "irradiance/data/aia_temp_resp.genx", device=None, aia_exp_time=2.9):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device
    
        aia_resp = read_genx(temp_resp_path)
        self.response = {}
        for key in aia_resp.keys():
            if key != 'HEADER':
                wavelength = int(key[1:])
                log_temperature = aia_resp[f'A{wavelength}']['LOGTE']  #log_temperature
                response = aia_resp[f'A{wavelength}']['TRESP']*aia_exp_time  # multiply response by typical AIA exposure time
                
                self.response[wavelength] = {}
                self.response[wavelength]['interpolator'] = Interp1D(torch.from_numpy(log_temperature).float().to(self.device), torch.from_numpy(response).float().to(self.device), method='linear', extrap=0)
                self.response[wavelength]['log_T'] = torch.from_numpy(log_temperature).float().to(self.device)
                self.response[wavelength]['response'] = torch.from_numpy(response).float().to(self.device)
                