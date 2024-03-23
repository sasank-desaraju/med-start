"""
Sasank Desaraju
4/4/23
"""

import torch
import pytorch_lightning as pl
import numpy as np
import os
from skimage import io
import cv2

from dataset import FistulaDataset


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        #self.img_dir = self.config.datamodule['IMAGE_DIRECTORY']
        #self.train_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'train_' + self.config.init['MODEL_NAME'] + '.csv'
        #self.val_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'val_' + self.config.init['MODEL_NAME'] + '.csv'
        #self.test_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'test_' + self.config.init['MODEL_NAME'] + '.csv'
        #self.naive_data = os.getcwd() + '/data/' + self.config.init['MODEL_NAME'] + '/' + 'naive_' + self.config.init['MODEL_NAME'] + '.csv'
        # "Naive" refers to a special test dataset that contains images from a dog that was not used in training or validation.
        # This differs from the normal test set in that the normal test set contains images from dogs that were used in training and validation.
        # For Sasank's MS thesis, for example, of the 10 dogs, dogs 1-9 were used in training, validation, and the normal test set, and dog 10 was used in the naive test set.

        # Data loader parameters
        # TODO: clean this up since we pulled config into this class
        self.batch_size = self.config.datamodule['BATCH_SIZE']
        self.num_workers = self.config.datamodule['NUM_WORKERS']
        self.pin_memory = self.config.datamodule['PIN_MEMORY']
        self.shuffle = self.config.datamodule['SHUFFLE']

        #self.log(batch_size=self.batch_size)
        # other constants

        # TODO: check train dataset length and integrity
        # TODO: check val dataset length and integrity

    def setup(self, stage):

        self.training_set = FistulaDataset(config=self.config,
                                            evaluation_type='train')
        self.validation_set = FistulaDataset(config=self.config,
                                            evaluation_type='val')
        self.test_set = FistulaDataset(config=self.config,
                                            evaluation_type='test')

        return

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.training_set,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.validation_set,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_set,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            shuffle=False)
