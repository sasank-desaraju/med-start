import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import random
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from math import floor, ceil
import pandas as pd

class FistulaDataset(torch.utils.data.Dataset):
    def __init__(self, config, evaluation_type, transform=None):
        self.config = config
        self.transform = self.config.transform      # TODO: Use MONAI transform
        
        if evaluation_type in ['train', 'val', 'test']:
            self.evaluation_type = evaluation_type
        else:
            raise ValueError('evaluation_type must be one of \'train\', \'val\', or \'test\'.')

        self.data = pd.read_csv(os.path.join(self.config.etl['DATA_DIR'], self.config.dataset['DATA_NAME'], self.evaluation_type + '_' + self.config.dataset['DATA_NAME'] + '.csv'))

        # * Load in the .nii.gz image and label files
        #! Not true: Each .nii file is about 50 MB when extracted. Thus, each pair is about 100 MB. train split has 32 images to about 3.2 GB of data loaded into memory. Not great, not terrible.
        # The above is cap because loading all images into GPU exceeds the A100's 80 GB memory. Now, we are using a MONAI CacheDataset to load the data.
        self.images = []
        self.labels = []
        self.patient_ids = []
        for i in range(len(self.data)):
            image = sitk.ReadImage(os.path.join(self.config.dataset['IMAGE_ROOT'], self.data['image'][i]), imageIO='NiftiImageIO')
            label = sitk.ReadImage(os.path.join(self.config.dataset['IMAGE_ROOT'], self.data['label'][i]), imageIO='NiftiImageIO')
            patient_id = self.data['patient_id'][i]
            self.images.append(image)
            self.labels.append(label)
            self.patient_ids.append(patient_id)
        

        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetSize(self.config.dataset['IMAGE_SIZE'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        label_np = sitk.GetArrayFromImage(label)
        #print('np.unique(label_np): ', np.unique(label_np))         # Returns [0 1] as expected
        #print('label_np pixel type: ', label_np.dtype)              # Returns
        # print the fraction of pixels that are 1 vs 0
        #print('fraction of pixels that are 1: ', np.sum(label_np == 1) / (np.sum(label_np == 0) + np.sum(label_np == 1)))

        #print('image size pre-resampling: ', image.GetSize())
        #print('label size pre-resampling: ', label.GetSize())

        # ? Should this go before or after resampling?
        #if self.transform != None & self.config.dataset['USE_TRANSFORMS']:
        if self.config.dataset['USE_TRANSFORMS']:
            image = self.transform(image=image)['image']
            label = self.transform(image=label)['image']
        
        # * Assert that the image and label are the same size and have the same spacing, origin, and direction
        np.testing.assert_almost_equal(image.GetSpacing(), label.GetSpacing(), decimal=5, err_msg='image and label spacing are not the same')
        np.testing.assert_almost_equal(image.GetSpacing(), label.GetSpacing(), decimal=5, err_msg='image and label spacing are not the same')
        np.testing.assert_almost_equal(image.GetOrigin(), label.GetOrigin(), decimal=5, err_msg='image and label origin are not the same')
        np.testing.assert_almost_equal(image.GetDirection(), label.GetDirection(), decimal=5, err_msg='image and label direction are not the same')

        # * Get the input parameters for resampling
        input_spacing = image.GetSpacing()
        input_origin = image.GetOrigin()
        input_direction = image.GetDirection()

        # * Configure the resampler
        self.resampler.SetOutputSpacing(input_spacing)
        self.resampler.SetOutputOrigin(input_origin)
        self.resampler.SetOutputDirection(input_direction)

        # * Resample the image and label
        self.resampler.SetInterpolator(sitk.sitkBSpline)
        self.resampler.SetDefaultPixelValue(0)
        image_resampled = self.resampler.Execute(image)
        #self.resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        self.resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        self.resampler.SetDefaultPixelValue(0)
        label_resampled = self.resampler.Execute(label)

        # * Make sure the label is a binary image since this is a binary segmentation task

        label_resampled_np = sitk.GetArrayFromImage(label_resampled)
        #print('np.unique(label_resampled_np): ', np.unique(label_resampled_np))     # Returns [0] only. Why?
        #print('fraction of pixels that are 1: ', np.sum(label_resampled_np == 1) / (np.sum(label_resampled_np == 0) + np.sum(label_resampled_np == 1)))
        #print('label_resampled_np pixel type: ', label_resampled_np.dtype)          # Returns
        if np.unique(label_resampled_np).tolist() != [0, 1]:
            print('Label is not binary. Binarizing now!')
            label_resampled_np[label_resampled_np != 0] = 1

        #print('np.unique(label_resampled_np): ', np.unique(label_resampled_np))
        assert np.unique(label_resampled_np).tolist() == [0, 1], 'Label is still not binary'
        
        label_resampled = sitk.GetImageFromArray(label_resampled_np)

        #print('image size post-resampling: ', image_resampled.GetSize())
        #print('label size post-resampling: ', label_resampled.GetSize())

        # * Convert the image and label to tensors. Add a channel dimension to the image tensor
        image_resampled = sitk.GetArrayFromImage(image_resampled)
        image_resampled = torch.from_numpy(image_resampled).float()
        image_resampled = image_resampled.unsqueeze(0)
        label_resampled = sitk.GetArrayFromImage(label_resampled)
        label_resampled = torch.from_numpy(label_resampled).float()
        label_resampled = label_resampled.unsqueeze(0)
        image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        label = sitk.GetArrayFromImage(label)
        label = torch.from_numpy(label).float()
        label = label.unsqueeze(0)

        #print('image tensor shape: ', image_resampled.shape)
        #print('label tensor shape: ', label_resampled.shape)


        sample = {
                    'image': image_resampled,
                    'label': label_resampled,
                    #'image_original': image,
                    #'label_original': label,
                    'patient_id': patient_id
                }

        return sample