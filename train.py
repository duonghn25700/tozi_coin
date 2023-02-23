from libs.lib import *
from src.data.mytransform import *
from src.data.dataset import *
from src.data.prepare import make_path_list
import toml
import copy
## Step 1 : mytransform
resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

## Step 2 : Make dataset
train_data_list = make_path_list(data_=".\\coins\\data", phase='train')
val_data_list = make_path_list(data_=".\\coins\\data", phase='val')

# print(data_list)
train_dataset = CoinDataset(train_data_list, transforms=ImageTransforms(resize, mean, std), phase='train')
val_dataset = CoinDataset(val_data_list, transforms=ImageTransforms(resize, mean, std), phase='val')

# index = 200
# print(len(train_dataset))
# img, label = train_dataset.__getitem__(index)
# print(img.shape)

## Step 3 : Make dataloader
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

dataloader_dict = {'train':train_dataloader, 'val':val_dataloader}
batch_iteration = iter(dataloader_dict['train'])

## Step 4 : Loss, Network, Optimizer
## Load pretrain model
coin_net = models.vgg16(pretrained=True)
coin_net.classifier[6] = nn.Linear(in_features=4096, out_features=212)
# print(coin_net)

# Loss
criterion = nn.CrossEntropyLoss()

# params to update
params_to_update = []
update_param_name = ['classifier.6.weight', 'classifier.6.bias']

for name, param in coin_net.named_parameters():
    if name in update_param_name:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

# Observe that all parameters are being optimized
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
# print(params_to_update)

def train_model(model, dataloader_dict, criterion, optimizer, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

         # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            if epoch == 0 and phase == 'train':
                continue
            # Iterate over data.
            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader_dict[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

num_epochs = 2
train_model(coin_net, dataloader_dict, criterion, optimizer, num_epochs=num_epochs)