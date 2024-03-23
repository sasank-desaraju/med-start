"""
Sasank Desaraju
3/23/24
"""

# create datamodule for our Chestmnist.npz dataset

#TODO: Do this correctly for the ChestDataset class lol. Really simple just instantiate the datasets in setup and then return them in the dataloaders.
# Maybe submit some config.dataset kwargs?

import os
import torch
import monai
import numpy as np
import lightning as L

class ChestMNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = '../data', batch_size: int = 32, num_workers: int = 0, foo: int = 4):
        super().__init__()
        self.foo = foo
        print(self.foo)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = monai.transforms.Compose(
            [
                monai.transforms.LoadNiftid(keys=["image", "label"]),
                monai.transforms.AddChanneld(keys=["image", "label"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    
    def prepare_data(self):
        # download, split, etc...
        # monai.data.Dataset.from_json(
        #     os.path.join(self.data_dir, "dataset.json"), 
        #     root_dir=self.data_dir, 
        #     transform=self.transform,
        # )
        return
    
    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # stage defines if we are at fit or test step
        if stage == 'fit' or stage is None:
            data = np.load(os.path.join(self.data_dir, 'chestmnist.npz'))
            self.train_images = torch.from_numpy(data['train_images']).float()
            self.train_labels = torch.from_numpy(data['train_labels']).long()
            self.val_images = torch.from_numpy(data['val_images']).float()
            self.val_labels = torch.from_numpy(data['val_labels']).long()
        if stage == 'test' or stage is None:
            data = np.load(os.path.join(self.data_dir, 'chestmnist.npz'))
            self.test_images = torch.from_numpy(data['test_images']).float()
            self.test_labels = torch.from_numpy(data['test_labels']).long()

        # Assign train/val datasets for use in dataloaders
        chestmnist = monai.data.Dataset.from_json(
            os.path.join(self.data_dir, "dataset.json"), 
            root_dir=self.data_dir, 
            transform=self.transform,
        )
        self.chestmnist_train, self.chestmnist_val, self.chestmnist_test = chestmnist.split(
            [0.8, 0.1, 0.1], 
            shuffle=True, 
            seed=42,
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.chestmnist_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.chestmnist_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.chestmnist_test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
        )
