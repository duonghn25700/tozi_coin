import sys
sys.path.append("../../libs")
from lib import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from glob import glob
import json

sys.path.append("..\\..\\coins\\data")
data_dir = sys.path[-1]


def one_hot_encoder(label, num_classes):
    # coin_one_hot_encoder = np.zeros(num_classes, dtype=int)
    # coin_one_hot_encoder[int(label)-1]=1
    return int(label)

class CoinDataset(Dataset):
    def __init__(self, image_path, transforms=None, phase ='train'):
        self.image_path = image_path
        self.phase = phase
        self.transforms = transforms
    
    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        image = Image.open(image_path)

        image_transform = self.transforms(image, self.phase)

        label = image_path.split('\\')[-2]
        
        label = one_hot_encoder(label, len(self.image_path))

        return image_transform, label

    def __len__(self):
        return len(self.image_path)
