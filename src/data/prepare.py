import sys
from torchvision import transforms
sys.path.append("../../libs")
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from glob import glob
import json

sys.path.append("..\\..\\coins\\data")
data_dir = sys.path[-1]

# make path list
def make_path_list(data_, phase='train'):
    target_path = os.path.join(data_ , phase + "\\**\\*.jpg")
    path_list = []
    for path in glob(target_path):
        path_list.append(path)
    return path_list