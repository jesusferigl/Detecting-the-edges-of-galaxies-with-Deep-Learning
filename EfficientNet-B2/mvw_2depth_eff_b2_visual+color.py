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
    'architecture' : 'timm-efficientnet-b2',
    'batch_size' : 32,
    'data_path_HIJV' : 'Imagenes_con_NaN/HIJV/',
    'data_path_color' : 'Imagenes_con_NaN/color/',
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
        self.conv_1_5 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_5 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_6 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_6 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_7 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_7 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_8 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_8 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_9 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_9 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_10 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_10 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.modelo = smp.Unet(encoder_name = name_second_network, encoder_weights = 'noisy-student', in_channels=90, classes=2)
    
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):

        output_after_conv_1_1 = self.conv_1_1(x1)
        output_after_conv_2_1 = self.conv_2_1(output_after_conv_1_1)
        output_after_conv_1_2 = self.conv_1_2(x2)
        output_after_conv_2_2 = self.conv_2_2(output_after_conv_1_2)
        output_after_conv_1_3 = self.conv_1_3(x3)
        output_after_conv_2_3 = self.conv_2_3(output_after_conv_1_3)
        output_after_conv_1_4 = self.conv_1_4(x4)
        output_after_conv_2_4 = self.conv_2_4(output_after_conv_1_4)
        output_after_conv_1_5 = self.conv_1_5(x5)
        output_after_conv_2_5 = self.conv_2_5(output_after_conv_1_5)
        output_after_conv_1_6 = self.conv_1_6(x6)
        output_after_conv_2_6 = self.conv_2_6(output_after_conv_1_6)
        output_after_conv_1_7 = self.conv_1_7(x7)
        output_after_conv_2_7 = self.conv_2_7(output_after_conv_1_7)
        output_after_conv_1_8 = self.conv_1_8(x8)
        output_after_conv_2_8 = self.conv_2_8(output_after_conv_1_8)
        output_after_conv_1_9 = self.conv_1_9(x9)
        output_after_conv_2_9 = self.conv_2_9(output_after_conv_1_9)
        output_after_conv_1_10 = self.conv_1_10(x10)
        output_after_conv_2_10 = self.conv_2_10(output_after_conv_1_10)
        output_after_red_1 = torch.cat([output_after_conv_2_1, output_after_conv_2_2, output_after_conv_2_3, output_after_conv_2_4, output_after_conv_2_5, output_after_conv_2_6, output_after_conv_2_7, output_after_conv_2_8, output_after_conv_2_9, output_after_conv_2_10], axis = 1)
        output = self.modelo(output_after_red_1)

        return output




class MyDataset(Dataset):

    def __init__(self, path_visual_images, path_color_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_color_images = path_color_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):
        
        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        x_GI = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-i.png')
        x_GR = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-r.png')
        x_GZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-z.png')
        x_IZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.i-z.png')
        x_RI = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.r-i.png')
        x_RZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.r-z.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        x_GI = np.array(self.resize(x_GI))/65535
        x_GR = np.array(self.resize(x_GR))/65535
        x_GZ = np.array(self.resize(x_GZ))/65535
        x_IZ = np.array(self.resize(x_IZ))/65535
        x_RI = np.array(self.resize(x_RI))/65535
        x_RZ = np.array(self.resize(x_RZ))/65535
        y = np.array(self.resize(y))
        
        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)
        
        x_HJI = torch.from_numpy(x_HJI).float()
        x_HJV = torch.from_numpy(x_HJV).float()
        x_HIV = torch.from_numpy(x_HIV).float()
        x_JIV = torch.from_numpy(x_JIV).float()
        x_GI = torch.from_numpy(x_GI).float().unsqueeze(dim = 0)
        x_GR = torch.from_numpy(x_GR).float().unsqueeze(dim = 0)
        x_GZ = torch.from_numpy(x_GZ).float().unsqueeze(dim = 0)
        x_IZ = torch.from_numpy(x_IZ).float().unsqueeze(dim = 0)
        x_RI = torch.from_numpy(x_RI).float().unsqueeze(dim = 0)
        x_RZ = torch.from_numpy(x_RZ).float().unsqueeze(dim = 0)
        y = torch.from_numpy(y).long()
        
        return x_HJI, x_HJV, x_HIV, x_JIV, x_GI, x_GR, x_GZ, x_IZ, x_RI, x_RZ, y
    
    def __len__(self):

        return len(self.path_masks)
        
class MyDatasetNormalRotationAndFlip(Dataset):

    def __init__(self, path_visual_images, path_color_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_color_images = path_color_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):

        angle = np.random.choice([0, 90, 180, 270])
        flip = np.random.choice(['0', 'v', 'h'])

        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        x_GI = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-i.png')
        x_GR = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-r.png')
        x_GZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.g-z.png')
        x_IZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.i-z.png')
        x_RI = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.r-i.png')
        x_RZ = Image.open(self.path_color_images + self.path_masks[index].split('/')[-1][:-6] + '.r-z.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        x_GI = np.array(self.resize(x_GI))/65535
        x_GR = np.array(self.resize(x_GR))/65535
        x_GZ = np.array(self.resize(x_GZ))/65535
        x_IZ = np.array(self.resize(x_IZ))/65535
        x_RI = np.array(self.resize(x_RI))/65535
        x_RZ = np.array(self.resize(x_RZ))/65535
        y = np.array(self.resize(y))

        if angle == 90:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_CLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_CLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_CLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_CLOCKWISE)
            x_GI = cv2.rotate(x_GI, cv2.ROTATE_90_CLOCKWISE)
            x_GR = cv2.rotate(x_GR, cv2.ROTATE_90_CLOCKWISE)
            x_GZ = cv2.rotate(x_GZ, cv2.ROTATE_90_CLOCKWISE)
            x_IZ = cv2.rotate(x_IZ, cv2.ROTATE_90_CLOCKWISE)
            x_RI = cv2.rotate(x_RI, cv2.ROTATE_90_CLOCKWISE)
            x_RZ = cv2.rotate(x_RZ, cv2.ROTATE_90_CLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)

        elif angle == 180:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_180)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_180)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_180)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_180)
            x_GI = cv2.rotate(x_GI, cv2.ROTATE_180)
            x_GR = cv2.rotate(x_GR, cv2.ROTATE_180)
            x_GZ = cv2.rotate(x_GZ, cv2.ROTATE_180)
            x_IZ = cv2.rotate(x_IZ, cv2.ROTATE_180)
            x_RI = cv2.rotate(x_RI, cv2.ROTATE_180)
            x_RZ = cv2.rotate(x_RZ, cv2.ROTATE_180)
            y = cv2.rotate(y, cv2.ROTATE_180)
            
        elif angle == 270:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_GI = cv2.rotate(x_GI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_GR = cv2.rotate(x_GR, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_GZ = cv2.rotate(x_GZ, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_IZ = cv2.rotate(x_IZ, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_RI = cv2.rotate(x_RI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_RZ = cv2.rotate(x_RZ, cv2.ROTATE_90_COUNTERCLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif flip == 'v':

            x_HJI = cv2.flip(x_HJI, 0)
            x_HJV = cv2.flip(x_HJV, 0)
            x_HIV = cv2.flip(x_HIV, 0)
            x_JIV = cv2.flip(x_JIV, 0)
            x_GI = cv2.flip(x_GI, 0)
            x_GR = cv2.flip(x_GR, 0)
            x_GZ = cv2.flip(x_GZ, 0)
            x_IZ = cv2.flip(x_IZ, 0)
            x_RI = cv2.flip(x_RI, 0)
            x_RZ = cv2.flip(x_RZ, 0)
            y = cv2.flip(y, 0)

        elif flip == 'h':

            x_HJI = cv2.flip(x_HJI, 1)
            x_HJV = cv2.flip(x_HJV, 1)
            x_HIV = cv2.flip(x_HIV, 1)
            x_JIV = cv2.flip(x_JIV, 1)
            x_GI = cv2.flip(x_GI, 1)
            x_GR = cv2.flip(x_GR, 1)
            x_GZ = cv2.flip(x_GZ, 1)
            x_IZ = cv2.flip(x_IZ, 1)
            x_RI = cv2.flip(x_RI, 1)
            x_RZ = cv2.flip(x_RZ, 1)
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
        noise_x_GI = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_GI.shape))).float()
        noise_x_GR = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_GR.shape))).float()
        noise_x_GZ = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_GZ.shape))).float()
        noise_x_IZ = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_IZ.shape))).float()
        noise_x_RI = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_RI.shape))).float()
        noise_x_RZ = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_RZ.shape))).float()
        x_HJI = torch.from_numpy(x_HJI).float() + noise_x_HJI
        x_HJV = torch.from_numpy(x_HJV).float() + noise_x_HJV
        x_HIV = torch.from_numpy(x_HIV).float() + noise_x_HIV
        x_JIV = torch.from_numpy(x_JIV).float() + noise_x_JIV
        x_GI = (torch.from_numpy(x_GI).float() + noise_x_GI).unsqueeze(dim = 0)
        x_GR = (torch.from_numpy(x_GR).float() + noise_x_GR).unsqueeze(dim = 0)
        x_GZ = (torch.from_numpy(x_GZ).float() + noise_x_GZ).unsqueeze(dim = 0)
        x_IZ = (torch.from_numpy(x_IZ).float() + noise_x_IZ).unsqueeze(dim = 0)
        x_RI = (torch.from_numpy(x_RI).float() + noise_x_RI).unsqueeze(dim = 0)
        x_RZ = (torch.from_numpy(x_RZ).float() + noise_x_RZ).unsqueeze(dim = 0)

        return x_HJI, x_HJV, x_HIV, x_JIV, x_GI, x_GR, x_GZ, x_IZ, x_RI, x_RZ, y
    
    def __len__(self):

        return len(self.path_masks)




train_masks, validation_masks = train_test_split(hyperparams['train_masks'], test_size=0.2, random_state=25)
train = MyDatasetNormalRotationAndFlip(hyperparams['data_path_HIJV'], hyperparams['data_path_color'], train_masks)
valid = MyDataset(hyperparams['data_path_HIJV'], hyperparams['data_path_color'], validation_masks)
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

            inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), data[8].to(device), data[9].to(device), data[10].to(device)
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10)
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

            inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), data[8].to(device), data[9].to(device), data[10].to(device)
            optimizer.zero_grad()
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10)
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
    
    torch.save(best_model, 'best_model_MVW-2Depth-' + hyperparams['architecture'] + '-Color')
    torch.save(neural_network, 'last_model_MVW-2Depth-' + hyperparams['architecture'] + '-Color')


perform_train(neural_network)
