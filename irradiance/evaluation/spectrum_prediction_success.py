import torch
from irradiance.utilities.data_loader import IrradianceDataModule
from irradiance.inference import ipredict
import os
import numpy as np
from irradiance.inference import ipredict
from tqdm import tqdm
import matplotlib.pyplot as plt

def unnormalize(y, eve_norm):
    eve_norm = torch.tensor(eve_norm).float()
    norm_mean = eve_norm[0]
    norm_stdev = eve_norm[1]
    y = y * norm_stdev[None].to(y) + norm_mean[None].to(y)
    return y

ckpt_path_kan = '/home/christophschirninger/megsai_checkpoints/kan_baseline_mean.ckpt'
ckpt_path_linear = '/home/christophschirninger/megsai_checkpoints/linear_baseline_7wl_nostd.ckpt'
load_ckpt_kan = torch.load(ckpt_path_kan)
load_ckpt_linear = torch.load(ckpt_path_linear)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_kan = load_ckpt_kan['model'].to(device)
model_linear = load_ckpt_linear['model'].to(device)
matches = '/mnt/disks/data-extended/matches/AIA_256_stacks_EVS_MEGS-A_matches.csv'
eve_data = '/mnt/disks/data-extended/preprocessed/EVE/EVS_MEGS-A_standardized.npy'
input_wl_kan = load_ckpt_kan[load_ckpt_kan['instrument']]
input_wl_linear = load_ckpt_linear[load_ckpt_linear['instrument']]
uv_norm = np.load('/mnt/disks/data-extended/preprocessed/AIA_256_EVS_MEGS-A_stats.npz', allow_pickle=True)['AIA'].item()
eve_norm_kan = model_kan.eve_norm
eve_norm_linear = model_linear.eve_norm

#input_wl = state[state['instrument']]
data_module_kan = IrradianceDataModule(stacks_csv_path=matches, eve_npy_path=eve_data, uv_norm=uv_norm, eve_norm=model_kan.eve_norm, wavelengths=input_wl_kan,
                                num_workers=os.cpu_count() // 2)
data_module_kan.setup()

data_module_linear = IrradianceDataModule(stacks_csv_path=matches, eve_npy_path=eve_data, uv_norm=uv_norm, eve_norm=model_linear.eve_norm, wavelengths=input_wl_linear,
                                num_workers=os.cpu_count() // 2)
data_module_linear.setup()

#test_eve = torch.stack([unnormalize(eve, eve_norm) for image, eve in data_module.test_ds])
test_eve_kan = torch.stack([eve for image, eve in data_module_kan.test_ds])
test_eve_linear = torch.stack([eve for image, eve in data_module_linear.test_ds])
test_aia_kan = torch.stack([torch.tensor(image) for image, eve in data_module_kan.test_ds])
test_aia_linear = torch.stack([torch.tensor(image) for image, eve in data_module_linear.test_ds])
test_irradiance_kan = [irr for irr in tqdm(ipredict(model_kan, test_aia_kan, return_images=False), total=len(test_aia_kan))]
test_irradiance_linear = [irr for irr in tqdm(ipredict(model_linear, test_aia_linear, return_images=False), total=len(test_aia_linear))]

test_irradiance_kan = torch.stack(test_irradiance_kan).numpy()
test_irradiance_linear = torch.stack(test_irradiance_linear).numpy()
test_eve_kan = test_eve_kan.numpy()
test_eve_linear = test_eve_linear.numpy()

test_error_kan = np.abs(test_eve_kan - test_irradiance_kan)
test_error_averaged_kan = np.sum(test_error_kan, axis=0)/test_error_kan.shape[0]

test_error_linear = np.abs(test_eve_linear - test_irradiance_linear)
test_error_averaged_linear = np.sum(test_error_linear, axis=0)/test_error_linear.shape[0]

test_error_all_av_kan = np.sum(test_error_averaged_kan)/len(test_error_averaged_kan)
print(test_error_all_av_kan)
test_error_all_av_linear = np.sum(test_error_averaged_linear)/len(test_error_averaged_linear)
print(test_error_all_av_linear)

# plot the irradiance
plt.figure(figsize=(10, 5))
plt.plot(test_error_kan[0], label='KAN error')
plt.plot(test_error_linear[0], label='Linear error', alpha=0.6) 
plt.legend()
#plt.yscale('log')
plt.xlabel('Wavelength')
plt.ylabel('Irradiance error')
plt.savefig('KAN_linear_error_abs.jpg')


# plot the averaged error
plt.figure(figsize=(10, 5))
plt.plot(test_error_averaged_kan, label='KAN error')
plt.plot(test_error_averaged_linear, label='Linear error', alpha=0.6)
plt.legend()
plt.title('Averaged irradiance error over time')
plt.xlabel('Wavelength')
plt.ylabel('Averaged irradiance error')
plt.savefig('KAN_linear_error_averaged_abs.jpg')

