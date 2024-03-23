"""
Sasank Desaraju
3/23/24
"""

import torch
import numpy as np
import pandas as pd
import os

class ChestDataset(torch.utils.data.Dataset):
    def __init__(self, config, stage, transform=None):
        self.config = config
        self.stage = stage
        self.transform = transform
        # sself.data = []
        # self.labels = []

        self.data = pd.read_csv(os.path.join(self.config.dataset['SPLITS_DIR'], self.config.dataset['SPLIT_NAME'], self.stage + '_' + self.config.dataset['SPLIT_NAME'] + '.csv'))
        self.images = np.load(self.config.dataset['DATA_SRC'])[self.stage + '_images']
        self.labels = np.load(self.config.dataset['DATA_SRC'])[self.stage + '_labels']

        assert self.images[0].shape == (64, 64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[self.data.iloc[idx]['image']]
        assert image.shape == (64, 64)
        label = self.labels[self.data.iloc[idx]['label']]
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': label}

        return sample