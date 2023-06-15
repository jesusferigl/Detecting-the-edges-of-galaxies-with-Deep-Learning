import copy
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




hyperparams = {
    'architecture' : 'resnet18',
    'batch_size' : 32,
    'data_path' : 'Imagenes_con_NaN/HIJV/',
    'epochs' : 2,
    'learning_rate' : 0.001,
    'train_masks' : glob.glob('mascaras_train/*'),
    'weight_decay' : 1e-5,
    'weights' : [0.075, 0.925]
}




class MultiviewNet(torch.nn.Module):

    def __init__(self, name_second_network):

        super().__init__()

        self.conv_1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_3 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_4 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_4 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.modelo = smp.Unet(encoder_name = name_second_network, encoder_weights = 'imagenet', in_channels=36, classes=2)
    
    def forward(self, x1, x2, x3, x4):

        output_after_conv_1_1 = self.conv_1_1(x1)
        output_after_conv_2_1 = self.conv_2_1(output_after_conv_1_1)
        output_after_conv_1_2 = self.conv_1_2(x2)
        output_after_conv_2_2 = self.conv_2_2(output_after_conv_1_2)
        output_after_conv_1_3 = self.conv_1_3(x3)
        output_after_conv_2_3 = self.conv_2_3(output_after_conv_1_3)
        output_after_conv_1_4 = self.conv_1_4(x4)
        output_after_conv_2_4 = self.conv_2_4(output_after_conv_1_4)
        output_after_red_1 = torch.cat([output_after_conv_2_1, output_after_conv_2_2, output_after_conv_2_3, output_after_conv_2_4], axis = 1)
        output = self.modelo(output_after_red_1)

        return output


# In[7]:


class MyDataset(Dataset):

    def __init__(self, path_visual_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):

        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        y = np.array(self.resize(y))

        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)

        y = torch.from_numpy(y).long()
        x_HJI = torch.from_numpy(x_HJI).float()
        x_HJV = torch.from_numpy(x_HJV).float()
        x_HIV = torch.from_numpy(x_HIV).float()
        x_JIV = torch.from_numpy(x_JIV).float()

        return x_HJI, x_HJV, x_HIV, x_JIV, y
    
    def __len__(self):

        return len(self.path_masks)
        
class MyDatasetNormalRotationAndFlip(Dataset):

    def __init__(self, path_visual_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):

        angle = np.random.choice([0, 90, 180, 270])
        flip = np.random.choice(['0', 'v', 'h'])

        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        y = np.array(self.resize(y))

        if angle == 90:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_CLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_CLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_CLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_CLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)

        elif angle == 180:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_180)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_180)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_180)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_180)
            y = cv2.rotate(y, cv2.ROTATE_180)

        elif angle == 270:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip == 'v':

            x_HJI = cv2.flip(x_HJI, 0)
            x_HJV = cv2.flip(x_HJV, 0)
            x_HIV = cv2.flip(x_HIV, 0)
            x_JIV = cv2.flip(x_JIV, 0)
            y = cv2.flip(y, 0)

        elif flip == 'h':

            x_HJI = cv2.flip(x_HJI, 1)
            x_HJV = cv2.flip(x_HJV, 1)
            x_HIV = cv2.flip(x_HIV, 1)
            x_JIV = cv2.flip(x_JIV, 1)
            y = cv2.flip(y, 1)

        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)

        y = torch.from_numpy(y).long()
        noise_x_HJI = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJI.shape))).float()
        noise_x_HJV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJV.shape))).float()
        noise_x_HIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HIV.shape))).float()
        noise_x_JIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_JIV.shape))).float()
        x_HJI = torch.from_numpy(x_HJI).float() + noise_x_HJI
        x_HJV = torch.from_numpy(x_HJV).float() + noise_x_HJV
        x_HIV = torch.from_numpy(x_HIV).float() + noise_x_HIV
        x_JIV = torch.from_numpy(x_JIV).float() + noise_x_JIV

        return x_HJI, x_HJV, x_HIV, x_JIV, y
    
    def __len__(self):

        return len(self.path_masks)




train_masks, validation_masks = train_test_split(hyperparams['train_masks'], test_size=0.2, random_state=25)
train = MyDatasetNormalRotationAndFlip(hyperparams['data_path'], train_masks)
valid = MyDataset(hyperparams['data_path'], validation_masks)
trainloader = DataLoader(train, batch_size=hyperparams['batch_size'], shuffle=True)
validloader = DataLoader(valid, batch_size=hyperparams['batch_size'], shuffle=True)




print('Train dataloader size:', trainloader.dataset.__len__())
print('Validation dataloader size:', validloader.dataset.__len__())



neural_network = MultiviewNet(hyperparams['architecture']).to(device)
class_weights = torch.FloatTensor(hyperparams['weights']).cuda()
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(neural_network.parameters(), lr = hyperparams['learning_rate'], weight_decay = hyperparams['weight_decay'])




def accuracy(predb, yb):

    metric = 0

    for i in range(yb.shape[0]):

        metric += (predb[i,:,:,:].argmax(dim=0) == yb[i,:,:]).float().mean().item()

    return(metric/yb.shape[0])



def validation_metrics(neural_network):

    neural_network.eval()

    mean_loss_validation, mean_accuracy_validation, validation_steps = 0, 0, 0
    tpl, fnl, fpl = [], [], []

    with torch.no_grad():

        for data in validloader:

            inputs1, inputs2, inputs3, inputs4, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4)
            loss_validation = criterion(outputs, labels)
            mean_loss_validation += loss_validation.item()
            mean_accuracy_validation += accuracy(outputs, labels)
            validation_steps += 1

            for m in range(labels.shape[0]):

                mask = labels[m,:,:]
                output = outputs[m,:,:,:].argmax(dim=0)
                tpl.append(torch.sum(output[mask == 1] == 1).item())
                fnl.append(torch.sum(output[mask == 1] == 0).item())
                fpl.append(torch.sum(output[mask == 0] == 1).item())

    precision = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fpl)))
    recall = np.mean(np.array(tpl)/(np.array(tpl) + np.array(fnl)))
    dice = np.mean(2*np.array(tpl)/(2*np.array(tpl)+np.array(fpl)+np.array(fnl)))
    
    return dice


def perform_train(neural_network):

    t = time.time()
    best_model = None
    best_epoch = None
    best_dice = 0

    for epoch in range(hyperparams['epochs']):

        neural_network.train()

        mean_loss_train, mean_accuracy_train, train_steps = 0, 0, 0

        for data in trainloader:

            inputs1, inputs2, inputs3, inputs4, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
            optimizer.zero_grad()
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4)
            loss = criterion(outputs, labels)
            mean_loss_train += loss.item()
            mean_accuracy_train += accuracy(outputs, labels)
            train_steps += 1
            loss.backward()
            optimizer.step()

        dice = validation_metrics(neural_network)

        if dice > best_dice:

            best_model = copy.deepcopy(neural_network)
            best_epoch = epoch
            best_dice = dice

        print('Epoch ', epoch + 1, 'finished.')

    print('Finished Training')
    print('Time required:', time.time() - t)
    print('Best model obtained in epoch ', best_epoch, ' with a validation dice of ', best_dice)
    
    torch.save(best_model, 'best_model_MVW-2Depth-' + hyperparams['architecture'] + '-Visual')
    torch.save(neural_network, 'last_model_MVW-2Depth-' + hyperparams['architecture'] + '-Visual')


perform_train(neural_network)
