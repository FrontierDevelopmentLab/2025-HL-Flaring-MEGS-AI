
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

# SXR Prediction Logger
# class SXRPredictionLogger(Callback):
#     def __init__(self, val_samples):
#         super().__init__()
#         self.val_samples = val_samples
#
#     def on_validation_epoch_end(self, trainer, pl_module):
#         # val_samples is a list of ((aia, sxr), target)
#         for (aia, sxr), target in self.val_samples:
#             aia, sxr, target = aia.to(pl_module.device), sxr.to(pl_module.device), target.to(pl_module.device)
#             pred = pl_module(aia.unsqueeze(0))  # Add batch dimension
#             trainer.logger.experiment.log({
#                 "val_pred_sxr": pred.cpu().numpy(),
#                 "val_target_sxr": target.cpu().numpy()
#             })

# Compute SXR normalization
def compute_sxr_norm(sxr_dir):
    sxr_values = []
    for f in Path(sxr_dir).glob("*.npy"):
        sxr = np.load(f)
        sxr = np.atleast_1d(sxr).flatten()[0]
        sxr_values.append(np.log10(sxr + 1e-8))
    sxr_values = np.array(sxr_values)
    if len(sxr_values) == 0:
        raise ValueError(f"No SXR files found in {sxr_dir}")
    return np.mean(sxr_values), np.std(sxr_values)

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
if args.sxr_norm:
    sxr_norm = np.load(args.sxr_norm)
else:
    sxr_norm = compute_sxr_norm(sxr_dir)
instrument = args.instrument

# Transforms
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Remove clone/detach
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
])
val_transforms = transforms.Compose([
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Remove clone/detach
])

# Training loop
n = 0
for parameter_set in combined_parameters:
    run_config = {key: item for key, item in zip(config_data['model'].keys(), parameter_set)}
    torch.manual_seed(run_config['seed'])
    np.random.seed(run_config['seed'])

    # DataModule
    data_loader = AIA_GOESDataModule(
        aia_dir=aia_dir,
        sxr_dir=sxr_dir,
        sxr_norm=sxr_norm,
        batch_size=16,
        num_workers=os.cpu_count() // 2,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
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
    plot_samples = plot_data  # Keep as list of ((aia, sxr), target)
    #sxr_callback = SXRPredictionLogger(plot_samples)

    sxr_plot_callback = ImagePredictionLogger_SXR(plot_data[0][0], plot_data[0][1], sxr_norm, plot_samples)


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
            eve_norm=sxr_norm,
            lr=run_config.get('lr', 1e-2),
            loss_func=HuberLoss()
        )
    elif run_config['architecture'] == 'hybrid':
        model = HybridIrradianceModel(
            d_input=6,
            d_output=1,
            eve_norm=sxr_norm,
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