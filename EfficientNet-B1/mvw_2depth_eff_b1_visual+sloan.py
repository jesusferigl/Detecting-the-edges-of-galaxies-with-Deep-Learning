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
    'architecture' : 'timm-efficientnet-b1',
    'batch_size' : 32,
    'data_path_HIJV' : 'Imagenes_con_NaN/HIJV/',
    'data_path_girz' : 'Imagenes_con_NaN/girz/',
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
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_9 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_10 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_10 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_11 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_11 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.conv_1_12 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 6, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU()
        )
        self.conv_2_12 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 9, kernel_size = (3,3), stride = (1,1), padding = (1,1), bias = True),
            torch.nn.BatchNorm2d(9),
            torch.nn.ReLU()
        )
        self.modelo = smp.Unet(encoder_name = name_second_network, encoder_weights = 'noisy-student', in_channels=108, classes=2)
    
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):

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
        output_after_conv_1_11 = self.conv_1_11(x11)
        output_after_conv_2_11 = self.conv_2_11(output_after_conv_1_11)
        output_after_conv_1_12 = self.conv_1_12(x12)
        output_after_conv_2_12 = self.conv_2_12(output_after_conv_1_12)
        output_after_red_1 = torch.cat([output_after_conv_2_1, output_after_conv_2_2, output_after_conv_2_3, output_after_conv_2_4, output_after_conv_2_5, output_after_conv_2_6, output_after_conv_2_7, output_after_conv_2_8, output_after_conv_2_9, output_after_conv_2_10, output_after_conv_2_11, output_after_conv_2_12], axis = 1)
        output = self.modelo(output_after_red_1)   
             
        return output


# In[15]:


class MyDataset(Dataset):

    def __init__(self, path_visual_images, path_girz_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_girz_images = path_girz_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):
        
        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        x_G = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.g.png')
        x_I = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.i.png')
        x_R = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.r.png')
        x_Z = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.z.png')
        x_IRG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.irg.png')
        x_ZIG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zig.png')
        x_ZIR = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zir.png')
        x_ZRG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zrg.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        x_G = np.array(self.resize(x_G))/65535
        x_I = np.array(self.resize(x_I))/65535
        x_R = np.array(self.resize(x_R))/65535
        x_Z = np.array(self.resize(x_Z))/65535
        x_IRG = np.array(self.resize(x_IRG))/255
        x_ZIG = np.array(self.resize(x_ZIG))/255
        x_ZIR = np.array(self.resize(x_ZIR))/255
        x_ZRG = np.array(self.resize(x_ZRG))/255
        y = np.array(self.resize(y))
        
        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)
        x_IRG = np.moveaxis(x_IRG, -1, 0)
        x_ZIG = np.moveaxis(x_ZIG, -1, 0)
        x_ZIR = np.moveaxis(x_ZIR, -1, 0)
        x_ZRG = np.moveaxis(x_ZRG, -1, 0)
        
        x_HJI = torch.from_numpy(x_HJI).float()
        x_HJV = torch.from_numpy(x_HJV).float()
        x_HIV = torch.from_numpy(x_HIV).float()
        x_JIV = torch.from_numpy(x_JIV).float()
        x_G = torch.from_numpy(x_G).float().unsqueeze(dim = 0)
        x_I = torch.from_numpy(x_I).float().unsqueeze(dim = 0)
        x_R = torch.from_numpy(x_R).float().unsqueeze(dim = 0)
        x_Z = torch.from_numpy(x_Z).float().unsqueeze(dim = 0)
        x_IRG = torch.from_numpy(x_IRG).float()
        x_ZIG = torch.from_numpy(x_ZIG).float()
        x_ZIR = torch.from_numpy(x_ZIR).float()
        x_ZRG = torch.from_numpy(x_ZRG).float()
        y = torch.from_numpy(y).long()
        
        return x_HJI, x_HJV, x_HIV, x_JIV, x_G, x_I, x_R, x_Z, x_IRG, x_ZIG, x_ZIR, x_ZRG, y
    
    def __len__(self):

        return len(self.path_masks)
        
class MyDatasetNormalRotationAndFlip(Dataset):

    def __init__(self, path_visual_images, path_girz_images, path_masks):

        self.path_visual_images = path_visual_images
        self.path_girz_images = path_girz_images
        self.path_masks = path_masks
        self.resize = torchvision.transforms.Resize((192, 192))
        
    def __getitem__(self, index):

        angle = np.random.choice([0, 90, 180, 270])
        flip = np.random.choice(['0', 'v', 'h'])

        x_HJI = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJI.png')
        x_HJV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HJV.png')
        x_HIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.HIV.png')
        x_JIV = Image.open(self.path_visual_images + self.path_masks[index].split('/')[-1][:-6] + '.JIV.png')
        x_G = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.g.png')
        x_I = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.i.png')
        x_R = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.r.png')
        x_Z = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.z.png')
        x_IRG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.irg.png')
        x_ZIG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zig.png')
        x_ZIR = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zir.png')
        x_ZRG = Image.open(self.path_girz_images + self.path_masks[index].split('/')[-1][:-6] + '.zrg.png')
        y = Image.open(self.path_masks[index])
        
        x_HJI = np.array(self.resize(x_HJI))/255
        x_HJV = np.array(self.resize(x_HJV))/255
        x_HIV = np.array(self.resize(x_HIV))/255
        x_JIV = np.array(self.resize(x_JIV))/255
        x_G = np.array(self.resize(x_G))/65535
        x_I = np.array(self.resize(x_I))/65535
        x_R = np.array(self.resize(x_R))/65535
        x_Z = np.array(self.resize(x_Z))/65535
        x_IRG = np.array(self.resize(x_IRG))/255
        x_ZIG = np.array(self.resize(x_ZIG))/255
        x_ZIR = np.array(self.resize(x_ZIR))/255
        x_ZRG = np.array(self.resize(x_ZRG))/255
        y = np.array(self.resize(y))

        if angle == 90:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_CLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_CLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_CLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_CLOCKWISE)
            x_G = cv2.rotate(x_G, cv2.ROTATE_90_CLOCKWISE)
            x_I = cv2.rotate(x_I, cv2.ROTATE_90_CLOCKWISE)
            x_R = cv2.rotate(x_R, cv2.ROTATE_90_CLOCKWISE)
            x_Z = cv2.rotate(x_Z, cv2.ROTATE_90_CLOCKWISE)
            x_IRG = cv2.rotate(x_IRG, cv2.ROTATE_90_CLOCKWISE)
            x_ZIG = cv2.rotate(x_ZIG, cv2.ROTATE_90_CLOCKWISE)
            x_ZIR = cv2.rotate(x_ZIR, cv2.ROTATE_90_CLOCKWISE)
            x_ZRG = cv2.rotate(x_ZRG, cv2.ROTATE_90_CLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_CLOCKWISE)

        elif angle == 180:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_180)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_180)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_180)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_180)
            x_G = cv2.rotate(x_G, cv2.ROTATE_180)
            x_I = cv2.rotate(x_I, cv2.ROTATE_180)
            x_R = cv2.rotate(x_R, cv2.ROTATE_180)
            x_Z = cv2.rotate(x_Z, cv2.ROTATE_180)
            x_IRG = cv2.rotate(x_IRG, cv2.ROTATE_180)
            x_ZIG = cv2.rotate(x_ZIG, cv2.ROTATE_180)
            x_ZIR = cv2.rotate(x_ZIR, cv2.ROTATE_180)
            x_ZRG = cv2.rotate(x_ZRG, cv2.ROTATE_180)
            y = cv2.rotate(y, cv2.ROTATE_180)

        elif angle == 270:

            x_HJI = cv2.rotate(x_HJI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HJV = cv2.rotate(x_HJV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_HIV = cv2.rotate(x_HIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_JIV = cv2.rotate(x_JIV, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_G = cv2.rotate(x_G, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_I = cv2.rotate(x_I, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_R = cv2.rotate(x_R, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_Z = cv2.rotate(x_Z, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_IRG = cv2.rotate(x_IRG, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_ZIG = cv2.rotate(x_ZIG, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_ZIR = cv2.rotate(x_ZIR, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_ZRG = cv2.rotate(x_ZRG, cv2.ROTATE_90_COUNTERCLOCKWISE)
            y = cv2.rotate(y, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip == 'v':

            x_HJI = cv2.flip(x_HJI, 0)
            x_HJV = cv2.flip(x_HJV, 0)
            x_HIV = cv2.flip(x_HIV, 0)
            x_JIV = cv2.flip(x_JIV, 0)
            x_G = cv2.flip(x_G, 0)
            x_I = cv2.flip(x_I, 0)
            x_R = cv2.flip(x_R, 0)
            x_Z = cv2.flip(x_Z, 0)
            x_IRG = cv2.flip(x_IRG, 0)
            x_ZIG = cv2.flip(x_ZIG, 0)
            x_ZIR = cv2.flip(x_ZIR, 0)
            x_ZRG = cv2.flip(x_ZRG, 0)
            y = cv2.flip(y, 0)

        elif flip == 'h':

            x_HJI = cv2.flip(x_HJI, 1)
            x_HJV = cv2.flip(x_HJV, 1)
            x_HIV = cv2.flip(x_HIV, 1)
            x_JIV = cv2.flip(x_JIV, 1)
            x_G = cv2.flip(x_G, 1)
            x_I = cv2.flip(x_I, 1)
            x_R = cv2.flip(x_R, 1)
            x_Z = cv2.flip(x_Z, 1)
            x_IRG = cv2.flip(x_IRG, 1)
            x_ZIG = cv2.flip(x_ZIG, 1)
            x_ZIR = cv2.flip(x_ZIR, 1)
            x_ZRG = cv2.flip(x_ZRG, 1)
            y = cv2.flip(y, 1)
            
        x_HJI = np.moveaxis(x_HJI, -1, 0)
        x_HJV = np.moveaxis(x_HJV, -1, 0)
        x_HIV = np.moveaxis(x_HIV, -1, 0)
        x_JIV = np.moveaxis(x_JIV, -1, 0)
        x_IRG = np.moveaxis(x_IRG, -1, 0)
        x_ZIG = np.moveaxis(x_ZIG, -1, 0)
        x_ZIR = np.moveaxis(x_ZIR, -1, 0)
        x_ZRG = np.moveaxis(x_ZRG, -1, 0)
        
        y = torch.from_numpy(y).long()
        noise_x_HJI = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJI.shape))).float()
        noise_x_HJV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HJV.shape))).float()
        noise_x_HIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_HIV.shape))).float()
        noise_x_JIV = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_JIV.shape))).float()
        noise_x_G = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_G.shape))).float()
        noise_x_I = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_I.shape))).float()
        noise_x_R = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_R.shape))).float()
        noise_x_Z = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_Z.shape))).float()
        noise_x_IRG = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_IRG.shape))).float()
        noise_x_ZIG = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_ZIG.shape))).float()
        noise_x_ZIR = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_ZIR.shape))).float()
        noise_x_ZRG = torch.from_numpy(abs(np.random.normal(loc=0, scale=np.random.choice([0.01, 0.05, 0.1]), size=x_ZRG.shape))).float()
        x_HJI = torch.from_numpy(x_HJI).float() + noise_x_HJI
        x_HJV = torch.from_numpy(x_HJV).float() + noise_x_HJV
        x_HIV = torch.from_numpy(x_HIV).float() + noise_x_HIV
        x_JIV = torch.from_numpy(x_JIV).float() + noise_x_JIV
        x_G = (torch.from_numpy(x_G).float() + noise_x_G).unsqueeze(dim = 0)
        x_I = (torch.from_numpy(x_I).float() + noise_x_I).unsqueeze(dim = 0)
        x_R = (torch.from_numpy(x_R).float() + noise_x_R).unsqueeze(dim = 0)
        x_Z = (torch.from_numpy(x_Z).float() + noise_x_Z).unsqueeze(dim = 0)
        x_IRG = (torch.from_numpy(x_IRG).float() + noise_x_IRG)
        x_ZIG = (torch.from_numpy(x_ZIG).float() + noise_x_ZIG)
        x_ZIR = (torch.from_numpy(x_ZIR).float() + noise_x_ZIR)
        x_ZRG = (torch.from_numpy(x_ZRG).float() + noise_x_ZRG)

        return x_HJI, x_HJV, x_HIV, x_JIV, x_G, x_I, x_R, x_Z, x_IRG, x_ZIG, x_ZIR, x_ZRG, y
    
    def __len__(self):

        return len(self.path_masks)


# In[16]:


train_masks, validation_masks = train_test_split(hyperparams['train_masks'], test_size=0.2, random_state=25)
train = MyDatasetNormalRotationAndFlip(hyperparams['data_path_HIJV'], hyperparams['data_path_girz'], train_masks)
valid = MyDataset(hyperparams['data_path_HIJV'], hyperparams['data_path_girz'], validation_masks)
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

            inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), data[8].to(device), data[9].to(device), data[10].to(device), data[11].to(device), data[12].to(device)
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12)
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

            inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device), data[7].to(device), data[8].to(device), data[9].to(device), data[10].to(device), data[11].to(device), data[12].to(device)
            optimizer.zero_grad()
            outputs = neural_network(inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10, inputs11, inputs12)
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
    
    torch.save(best_model, 'best_model_MVW-2Depth-' + hyperparams['architecture'] + '-Sloan')
    torch.save(neural_network, 'last_model_MVW-2Depth-' + hyperparams['architecture'] + '-Sloan')


perform_train(neural_network)
