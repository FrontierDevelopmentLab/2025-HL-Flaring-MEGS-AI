
import argparse
import os
import yaml
import itertools
import wandb
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.nn import HuberLoss
from SDOAIA_dataloader import AIA_GOESDataModule
from linear_and_hybrid import LinearIrradianceModel, HybridIrradianceModel
from callback import ImagePredictionLogger_SXR

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint_dir', type=str, required=True, help='Directory to save checkpoints.')
parser.add_argument('-model', type=str, default='config.yaml', help='Path to model config YAML.')
parser.add_argument('-aia_dir', type=str, required=True, help='Path to AIA .npy files.')
parser.add_argument('-sxr_dir', type=str, required=True, help='Path to SXR .npy files.')
parser.add_argument('-sxr_norm', type=str, help='Path to SXR normalization (mean, std).')
parser.add_argument('-instrument', type=str, default='AIA_6', help='Instrument (e.g., AIA_6 for 6 wavelengths).')
args = parser.parse_args()

# Load config
with open(args.model, 'r') as stream:
    config_data = yaml.load(stream, Loader=yaml.SafeLoader)

dic_values = [i for i in config_data['model'].values()]
combined_parameters = list(itertools.product(*dic_values))

# Paths and normalization
checkpoint_dir = args.checkpoint_dir
aia_dir = args.aia_dir
sxr_dir = args.sxr_dir

sxr_norm = np.load(args.sxr_norm)
instrument = args.instrument

n = 0
for parameter_set in combined_parameters:
    run_config = {key: item for key, item in zip(config_data['model'].keys(), parameter_set)}
    torch.manual_seed(run_config['seed'])
    np.random.seed(run_config['seed'])

    # DataModule
    data_loader = AIA_GOESDataModule(
        aia_dir=aia_dir,
        sxr_dir=sxr_dir,
        batch_size=32,
        num_workers=os.cpu_count(),
        sxr_norm=sxr_norm,
        # train_transforms=train_transforms,
        # val_transforms=val_transforms,
        val_split=0.2,
        test_split=0.1
    )
    data_loader.setup()

    # Logger
    wb_name = f"{instrument}_{n}" if len(combined_parameters) > 1 else "aia_sxr_model"
    wandb_logger = WandbLogger(
        entity=config_data['wandb']['entity'],
        project=config_data['wandb']['project'],
        job_type=config_data['wandb']['job_type'],
        tags=config_data['wandb']['tags'],
        name=wb_name,
        notes=config_data['wandb']['notes'],
        config=run_config
    )

    # Logging callback
    total_n_valid = len(data_loader.valid_ds)
    plot_data = [data_loader.valid_ds[i] for i in range(0, total_n_valid, max(1, total_n_valid // 4))]
    print(plot_data[0])  # Print first sample for debugging
    plot_samples = plot_data  # Keep as list of ((aia, sxr), target)
    #sxr_callback = SXRPredictionLogger(plot_samples)

    sxr_plot_callback = ImagePredictionLogger_SXR(plot_samples, sxr_norm)


    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='valid_loss',
        mode='min',
        save_top_k=1,
        filename=f"{wb_name}-{{epoch:02d}}-{{valid_loss:.4f}}"
    )

    # Model
    if run_config['architecture'] == 'linear':
        model = LinearIrradianceModel(
            d_input=6,
            d_output=1,
            lr=run_config.get('lr', 1e-4),
            loss_func=MSELoss()
        )
    elif run_config['architecture'] == 'hybrid':
        model = HybridIrradianceModel(
            d_input=6,
            d_output=1,
            cnn_model=run_config['cnn_model'],
            ln_model=True,
            cnn_dp=run_config.get('cnn_dp', 0.75),
            lr=run_config.get('lr', 1e-4)
        )
    else:
        raise NotImplementedError(f"Architecture {run_config['architecture']} not supported.")

    # Trainer
    trainer = Trainer(
        default_root_dir=checkpoint_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=run_config.get('epochs', 10),
        callbacks=[sxr_callback, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, data_loader)

    # Save checkpoint
    save_dictionary = run_config
    save_dictionary['model'] = model
    save_dictionary['instrument'] = instrument
    full_checkpoint_path = os.path.join(checkpoint_dir, f"{wb_name}_{n}.ckpt")
    torch.save(save_dictionary, full_checkpoint_path)

    # Test
    trainer.test(model, dataloaders=data_loader.test_dataloader())

    # Finalize
    wandb.finish()
    n += 1