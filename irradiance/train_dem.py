import argparse
import os
import sys
import yaml
import itertools
import wandb

import albumentations as A
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback, StochasticWeightAveraging
from irradiance.models.kan_success import KANDEM, KANDEMSpectrum
from irradiance.models.mlp import MLPDEMSpectrum
from irradiance.utilities.data_loader import IrradianceDataModule
from irradiance.utilities.callback import ImagePredictionLogger, SpectrumPredictionLogger

torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint', type=str, required=True, help='Path to checkpoint.')
parser.add_argument('-model', dest='model', default='megs_ai_config.yaml', required=False)
parser.add_argument('-matches_table', dest='matches_table', type=str,
                    default="/mnt/disks/preprocessed_data/AIA/matches_eve_aia.csv",
                    help='matches_table')
parser.add_argument('-eve_data', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_converted.npy",
                    help='Path to converted SDO/EVE data.')
parser.add_argument('-eve_norm', type=str, default="/mnt/disks/preprocessed_data/EVE/megsa_normalization.npy",
                    help='Path to converted SDO/EVE normalization.')
parser.add_argument('-uv_norm', type=str, default="/mnt/disks/data-extended/preprocessed/AIA_256_EVS_MEGS-AB_stats.npz",
                    help='Path to calculated AIA normalization.')
parser.add_argument('-eve_wl', type=str, default=None,
                    help='Path to SDO/EVE wavelength names.')
parser.add_argument('-instrument', type=str, required=True, 
                    help='Instrument wavelengths to use as input.')
args = parser.parse_args()
with open(args.model, 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.SafeLoader)

dic_values = [i for i in config_data['model'].values()]
combined_parameters = list(itertools.product(*dic_values))

# Paths
matches_table = args.matches_table
checkpoint = args.checkpoint
eve_data = args.eve_data
eve_norm = args.eve_norm
uv_norm = args.uv_norm
eve_wl = args.eve_wl
instrument = args.instrument

# EVE and UV: Normalization data
eve_norm = np.load(eve_norm)
uv_norm = np.load(uv_norm, allow_pickle=True)['AIA'].item()

# Perform all combination of free parameters (if there are any)
n = 0
for parameter_set in combined_parameters:

    # Read configuration file
    run_config= {}
    for key, item in zip(list(config_data['model'].keys()), parameter_set):
        run_config[key] = item
    # Seed: For reproducibility
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data augmentation
    # TODO: Incorporate augmentation or not, depending on branch that is being trained
    # if run_config['linear_architecture == 'linear' or run_config['model_architecture == 'complex':
    #     train_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })
    # else:
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=180, p=0.9, value=0, border_mode=1),
        ToTensorV2()], additional_targets={'y': 'image', })
    val_transforms = A.Compose([ToTensorV2()], additional_targets={'y': 'image', })

    if (run_config['val_months'][0] in run_config['test_months']) is False and (run_config['val_months'][1] in run_config['test_months']) is False: 

        # Initialize data loader
        data_loader = IrradianceDataModule(matches_table, eve_data, eve_norm, uv_norm, run_config[instrument], 
                                           num_workers=os.cpu_count() // 2,
                                           train_transforms=train_transforms, 
                                           val_transforms=val_transforms,
                                           batch_size=2,
                                           val_months=run_config['val_months'], 
                                           test_months=run_config['test_months'],
                                           holdout_months=run_config['holdout_months'],
                                           norm_eve=True,
                                           norm_uv=False,
                                           )
        data_loader.setup()
        
              
        # Initialize logger
        if len(combined_parameters) > 1:
            wb_name = f"{instrument}_{n}"
        else:
            wb_name = os.path.basename(checkpoint)
              
        wandb_logger = WandbLogger(entity=config_data['wandb']['entity'],
                                project=config_data['wandb']['project'],                            
                                #group=config_data['wandb']['group'],
                                job_type=config_data['wandb']['job_type'],
                                tags=config_data['wandb']['tags'],
                                name=config_data['wandb']['wb_name'],
                                notes=config_data['wandb']['notes'],
                                config=run_config)                           

        # Plot callback
        total_n_valid = len(data_loader.valid_ds)
        plot_data = [data_loader.valid_ds[i] for i in range(0, total_n_valid, total_n_valid // 4)]
        plot_images = torch.stack([image for image, eve in plot_data])
        plot_eve = torch.stack([eve for image, eve in plot_data])
        if eve_wl is not None:
            eve_wl = np.load(eve_wl, allow_pickle=True)
        
        if config_data['logging'] == 'images':
            image_callback = ImagePredictionLogger(plot_images, plot_eve, eve_wl, run_config[instrument])
        elif config_data['logging'] == 'spectrum':
            image_callback = SpectrumPredictionLogger(plot_images, plot_eve, eve_wl, run_config[instrument])
        else:
            raise NotImplementedError(f"{config_data['logging']} is not implemented.")
            
        # Checkpoint callback
        checkpoint_path = os.path.split(checkpoint)[0]
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                              monitor='valid_loss', mode='min', save_top_k=1,
                                              filename=checkpoint)
        
        kanfov = run_config['kanfov']
        stride = run_config['kanstride']
        wavelengths = [94, 131, 171, 193, 211, 304, 335]
        t_query_points_n= run_config['t_query_points_n']
        
        if run_config['architecture'] == 'DEM':
            model = KANDEM(eve_norm=eve_norm,
                        uv_norm=uv_norm,
                        kanfov = kanfov,
                        wavelengths=wavelengths,
                        layers_hidden=[len(wavelengths) * kanfov * kanfov + 1, 128, 1],
                        grid_min = [0, -3],
                        grid_max = [7, 3],
                        num_grids = 8,
                        use_base_update = True,
                        base_activation = F.silu,
                        spline_weight_init_scale = 0.1,
                        lr = run_config['lr'],
                        t_query_points=torch.linspace(4, 9, t_query_points_n).to(device), # 20 query points between 10^4 and 10^9 K
                        base_temp_exponent = 3,
                        intensity_factor = 1e20,
                        )
            
            # Initialize trainer
            trainer = Trainer(
                default_root_dir=checkpoint_path,
                accelerator="gpu",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=run_config['epochs'],
                callbacks=[checkpoint_callback],
                logger=wandb_logger,
                log_every_n_steps=10
                )
                
        elif run_config['architecture'] == 'KANDEMSpectrum':
            model = KANDEMSpectrum(eve_norm = eve_norm,
                                uv_norm = uv_norm,
                                kanfov = kanfov,
                                wavelengths = wavelengths,
                                t_query_points = torch.log10(torch.linspace(10**5, 10**7, t_query_points_n)).to(device),
                                layers_hidden_dem = [len(wavelengths) * kanfov * kanfov + 1, 128, 64, 1],
                                layers_hidden_sp = [t_query_points_n, 128, 64, eve_norm.shape[1]],
                                grid_min_dem = [0, 0, 0],
                                grid_min_sp = [0, -3, -3],
                                grid_max_dem = [5, 8, 8],
                                grid_max_sp = [7, 3, 3],
                                num_grids_dem= 16,
                                num_grids_sp = 8,
                                use_base_update_dem = True,
                                use_base_update_sp = True,
                                spline_weight_init_scale_dem = 0.1,
                                spline_weight_init_scale_sp = 0.1,
                                base_activation = F.silu,
                                use_layernorm = True,
                                base_temp_exponent=4,
                                intensity_factor=1e20,
                                lr=run_config['lr'],
                                stride=stride,
                                log_sploss_factor = run_config['log_sploss_factor'],
                                lin_sploss_factor = run_config['lin_sploss_factor'],
                                hybrid=False
                                )
            
            # Initialize trainer
            trainer = Trainer(
                default_root_dir=checkpoint_path,
                accelerator="gpu",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=run_config['epochs'],
                callbacks=[image_callback, checkpoint_callback, StochasticWeightAveraging(swa_lrs=1.e-3, swa_epoch_start=0.7, annealing_epochs=10)],
                logger=wandb_logger,
                log_every_n_steps=10,

                )

        elif run_config['architecture'] == 'MLPDEMSpectrum':
            model = MLPDEMSpectrum(eve_norm = eve_norm,
                                uv_norm = uv_norm,
                                kanfov = kanfov,
                                wavelengths = wavelengths,
                                t_query_points = torch.log10(torch.linspace(10**5, 10**7, t_query_points_n)).to(device),
                                layers_hidden_dem = [len(wavelengths) * kanfov * kanfov + 1, 128, 64, 1],
                                layers_hidden_sp = [t_query_points_n, 128, 64, eve_norm.shape[1]],
                                base_activation = F.silu,
                                use_layernorm = False,
                                base_temp_exponent=4,
                                intensity_factor=1e20,
                                lr=run_config['lr'],
                                stride=stride,
                                log_sploss_factor = run_config['log_sploss_factor'],
                                lin_sploss_factor = run_config['lin_sploss_factor'],
                                )
            
            # Initialize trainer
            trainer = Trainer(
                default_root_dir=checkpoint_path,
                accelerator="gpu",
                devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
                max_epochs=run_config['epochs'],
                callbacks=[image_callback, checkpoint_callback],
                logger=wandb_logger,
                log_every_n_steps=10
                )


        # Train the model ⚡
        trainer.fit(model, data_loader)

        save_dictionary = run_config
        save_dictionary['model'] = model
        save_dictionary['instrument'] = instrument
        if len(combined_parameters) > 1:
            # TODO: Modify
            full_checkpoint_path = f"{checkpoint}_{n}_sunerf.ckpt"
            n = n + 1
        else:
            full_checkpoint_path = f"{checkpoint}_sunerf.ckpt"
        torch.save(save_dictionary, full_checkpoint_path)

        # Evaluate on test set
        # Load model from checkpoint
        # TODO: Correct: KeyError: 'pytorch-lightning_version'
        state = torch.load(full_checkpoint_path)
        model = state['model']
        trainer.test(model, data_loader)

        # Finalize logging
        wandb.finish()




