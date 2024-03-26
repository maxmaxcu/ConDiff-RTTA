import os
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

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

class MinMaxNorm_Renorm(object):
    def __init__(self, min, max, scale=255):
        self.min = min
        self.max = max
        self.scale = scale
    def __call__(self, img):
        img = (img/self.scale)*(self.max - self.min) + self.min
        return img
    
    def __repr__(self):
        return self.__class__.__name__+'()'
        

class TCIRDataset_TTA(data.Dataset):
    def __init__(self,path='',phase='valid', config=None):
        dict = {'train':'2003_2013','valid':'2014_2014','test':'2015_2016'}
        self.dataset_x = torch.tensor(np.load(os.path.join(path,f"x_{dict[phase]}.npy"))[:,:,:-1,:-1],dtype=torch.float32)
        self.dataset_y = torch.tensor(np.load(os.path.join(path,f"y_{dict[phase]}.npy")),dtype=torch.float32).unsqueeze(-1)
        self.transform_discr = transforms.Compose([
            transforms.Normalize(mean=(269.0213,),std=(26.61631,)),
            transforms.Resize((64,64),antialias=True)
        ])
        self.transform_diff = transforms.Compose([
            MinMaxNorm(min=76,max=347),
            transforms.Resize((64,64),antialias=True)
        ])
        self.dataset_x_discr = self.transform_discr(self.dataset_x)
        self.dataset_x_diff = self.transform_diff(self.dataset_x) / 127.5 - 1

        self.dataset_y = self.dataset_y[config.input.testdata_start_index:config.input.testdata_end_index]
        self.dataset_x_discr = self.dataset_x_discr[config.input.testdata_start_index:config.input.testdata_end_index]
        self.dataset_x_diff = self.dataset_x_diff[config.input.testdata_start_index:config.input.testdata_end_index]
        self.dataset_x=None
        
    def __len__(self):
        return len(self.dataset_y)

    def __getitem__(self,index):
        data_x_discr = self.dataset_x_discr[index]
        data_x_diff = self.dataset_x_diff[index]
        data_y = self.dataset_y[index]
        return data_x_discr, data_x_diff, data_y
