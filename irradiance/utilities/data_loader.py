import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

from irradiance.data.utils import loadMapStack


class IrradianceDataModule(pl.LightningDataModule):

    def __init__(self, stacks_csv_path, eve_npy_path, wavelengths, batch_size: int = 32, num_workers=None,
                 train_transforms=None, val_transforms=None, val_months=[10, 1], test_months=[11,12], holdout_months=None):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        stacks_csv_path: path to the matches
        eve_npy_path: path to the EVE data file
        """
        super().__init__()
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        self.stacks_csv_path = stacks_csv_path
        self.eve_npy_path = eve_npy_path
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.wavelengths = wavelengths
        self.val_months = val_months
        self.test_months = test_months
        self.holdout_months = holdout_months

    # TODO: Extract SDO dats for train/valid/test sets
    def setup(self, stage=None):
        # load data stacks (paired samples)
        df = pd.read_csv(self.stacks_csv_path, parse_dates=['eve_dates'])
        eve_data = np.load(self.eve_npy_path)

        assert len(eve_data) == len(df), 'Inconsistent data state. EVE and AIA stacks do not match.'


        test_cond_2014 = df.eve_dates.dt.year.isin([2014])
        val_condition = df.eve_dates.dt.month.isin(self.val_months)
        test_condition = df.eve_dates.dt.month.isin(self.test_months) | test_cond_2014 

        if self.holdout_months is not None:
            holdout_condition = df.eve_dates.dt.month.isin(self.holdout_months)
            train_condition = ~(val_condition | test_condition | holdout_condition)
        else:
            train_condition = ~(val_condition | test_condition)

        valid_df = df[val_condition]
        test_df = df[test_condition]
        train_df = df[train_condition]

        self.train_ds = IrradianceDataset(np.array(train_df.aia_stack), eve_data[train_df.index], 
                                          self.wavelengths, self.train_transforms)
        self.valid_ds = IrradianceDataset(np.array(valid_df.aia_stack), eve_data[valid_df.index], 
                                          self.wavelengths, self.val_transforms)
        self.test_ds = IrradianceDataset(np.array(test_df.aia_stack), eve_data[test_df.index], 
                                         self.wavelengths, self.val_transforms)

        self.train_dates = train_df['eve_dates']
        self.valid_dates = valid_df['eve_dates']
        self.test_dates = test_df['eve_dates']


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class IrradianceDataset(Dataset):

    def __init__(self, euv_paths, eve_irradiance, wavelengths, transformations=None):
        """ Loads paired data samples of AIA EUV images and EVE irradiance measures.

        Input data needs to be paired.
        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        eve_irradiance: list of the EVE data values (n_samples, eve_channels)
        """
        assert len(euv_paths) == len(eve_irradiance), 'Input data needs to be paired!'
        self.euv_paths = euv_paths
        self.eve_irradiance = eve_irradiance
        self.transformations = transformations
        self.wavelengths = wavelengths

    def __len__(self):
        return len(self.euv_paths)

    def __getitem__(self, idx):
        euv_images = np.load(self.euv_paths[idx])[self.wavelengths, ...]
        eve_data = self.eve_irradiance[idx]
        # TODO: Understand what's happening below???
        if self.transformations is not None:
            # transform as RGB + y to use transformations
            euv_images = euv_images.transpose()
            # transformed = self.transformations(image=euv_images[..., :3], y=euv_images[..., 3:])
            # euv_images = torch.cat([transformed['image'], transformed['y']], dim=0)
            transformed = self.transformations(image=euv_images)
            euv_images = transformed['image']

        return euv_images, torch.tensor(eve_data, dtype=torch.float32)


class FITSDataset(Dataset):
    def __init__(self, paths, resolution=512, map_reproject=False, aia_preprocessing=True):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        paths: list of numpy paths as string (n_samples, )
        """
        self.paths = paths
        self.resolution = resolution
        self.map_reproject = map_reproject
        self.aia_preprocessing = aia_preprocessing

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        euv_images = loadMapStack(self.paths[idx], resolution=self.resolution, remove_nans=True,
                                  map_reproject=self.map_reproject, aia_preprocessing=self.aia_preprocessing)
        return torch.tensor(euv_images, dtype=torch.float32)


class NumpyDataset(Dataset):

    def __init__(self, euv_paths, euv_wavelengths):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        euv_paths: list of numpy paths as string (n_samples, )
        """
        self.euv_paths = euv_paths
        self.euv_wavelengths = euv_wavelengths

    def __len__(self):
        return len(self.euv_paths)

    def __getitem__(self, idx):
        euv_images = np.load(self.euv_paths[idx])[self.euv_wavelengths, ...]
        return torch.tensor(euv_images, dtype=torch.float32)


class ArrayDataset(Dataset):

    def __init__(self, data):
        """ Loads data samples of AIA EUV images.

        Parameters
        ----------
        data: list of numpy paths as string (n_samples, )
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)