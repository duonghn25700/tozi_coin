import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models, datasets, transforms
from PIL import Image
import json
import numpy as np
# import cv2
import matplotlib.pyplot as plt
from src.utils.inference import Predictor, BaseTranform
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



### Everything need to change
###############################################################################################
# Load model resnet18 as pretrained
## Change last Layer to 5
model = models.resnet50(pretrained=True)
model = model.to(device)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 216)

## Load model
Path = 'logs/coinmodel_resnet50_2.pt'
checkpoint = torch.load(Path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

# Class
with open("cat_to_name.json", 'r') as f:
  data = json.load(f)
print(data)
class_index = []
# for i in range(1, len(data)+1):
#   class_index.append(data[str(i)])
# print(class_index)
###############################################################################################

for path in os.listdir('coins/data/val'):
    class_index.append(data[path])

# Prediction
predictor = Predictor(class_index)

# Parameters to Transform
resize = 256
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

### Prediction from Folder
# Path : Images need to predicted
image_file_path = 'test/'
i = 1
transform = BaseTranform(resize, mean, std)
for file in os.listdir(image_file_path):
    img = Image.open(image_file_path + file)
    img_transformed = transform(img)
    img_transformed = img_transformed.unsqueeze_(0)
    out = model(img_transformed)
    result = predictor.predict_max(out)
    plt.subplot(5, 3, i)
    plt.imshow(img)
    plt.title(result)
    plt.axis('off')
    i += 1
#     print(result)
plt.show()

#####################################