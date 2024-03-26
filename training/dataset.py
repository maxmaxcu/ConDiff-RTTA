# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from glob import glob

try:
    import pyspng
except ImportError:
    pyspng = None

from torch.utils import data
from torchvision import transforms

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class TCIRDataset(data.Dataset):
    def __init__(self,path='',phase='train', **kwargs):
        dict = {'train':'2003_2013','valid':'2014_2014','test':'2015_2016'}
        self.dataset_x = np.load(os.path.join(path,f"x_{dict[phase]}.npy"))[:,:,:-1,:-1]
        self.dataset_y = np.load(os.path.join(path,f"y_{dict[phase]}.npy"))
        
        if phase=='train':
            dataset_y_t4_t5 = self.dataset_y[np.where((self.dataset_y>65)&(self.dataset_y<90))]
            dataset_y_t5_t8 = self.dataset_y[np.where((self.dataset_y>90))].repeat(2,axis=0)
            dataset_x_t4_t5 = self.dataset_x[np.where((self.dataset_y>65)&(self.dataset_y<90))]
            dataset_x_t5_t8 = self.dataset_x[np.where((self.dataset_y>90))].repeat(2,axis=0)
            self.dataset_y = np.concatenate([self.dataset_y, dataset_y_t4_t5, dataset_y_t5_t8],axis=0)
            self.dataset_x = np.concatenate([self.dataset_x, dataset_x_t4_t5, dataset_x_t5_t8],axis=0)
        self.dataset_y = torch.tensor(self.dataset_y,dtype=torch.float32)
        self.dataset_x = torch.tensor(self.dataset_x, dtype=torch.float32)
        self.transform = transforms.Compose([
        MinMaxNorm(min=76,max=347),
        transforms.Resize((64,64),antialias=True)
        ])
        self.dataset_x = self.transform(self.dataset_x).to(torch.uint8)
    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self,index):
        data_x = self.dataset_x[index]
        data_y = self.dataset_y[index]
        return data_x, data_y

    @property
    def name(self):
        return 'TCIR'

    @property
    def image_shape(self):
        return self.dataset_x.shape[1:]

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_dim(self):
        return 1

    @property
    def has_labels(self):
        return True

class MinMaxNorm(object):
    def __init__(self, min, max, scale=255):
        self.min = min
        self.max = max
        self.scale = scale
    def __call__(self, img):
        img = ((img - self.min)/(self.max - self.min))*self.scale
        return img

    def __repr__(self):
        return self.__class__.__name__+'()'

