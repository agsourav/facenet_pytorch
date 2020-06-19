import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse

def infer(model, image):    #image shape: (h,w,c)
    #image = torch.reshape(image, (229,229,3))
    #transform = torchvision.transforms.RandomCrop((229,229))
    #image = transform(image)
    img = image.permute(2,0,1)
    img = torch.unsqueeze(img, 0)
    print(img.shape)

    model.eval()
    with torch.no_grad():
        out = model(img.float())
    return out

def load_dataset(root, batchsize = 1, shuffle = False):
    train_transformer = transforms.Compose([
        transforms.Resize((229,229)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.ImageFolder(root = root, transform = train_transformer)
    trainloader = DataLoader(train_set, batch_size = batchsize, shuffle = shuffle)

    return trainloader

def arg_parse():
    parser = argparse.ArgumentParser(description='Facenet custom face recognition model')
    parser.add_argument('--rootdir', dest = 'rootdir', help = 'path of root directory of dataset',
    default= 'datasets/custom_images', type = str)
    parser.add_argument('--lr', dest = 'lr_rate',help = 'learning rate',
    default = 0.1)
    parser.add_argument('--bs', dest = 'batch', help = 'batch size',
    default = 2)
    parser.add_argument('--ep', dest = 'epochs', help = 'epochs',
    default = 1)
    parser.add_argument('--train', dest = 'train', help = '1 for training',
    default = 1)
    parser.add_argument('--check', dest = 'checkpoint', help = '1 to start from previous checkpoint/0 to initialise',
    default = 1)
    parser.add_argument('--train-module', dest = 'training_module', help = 'train inception model or classification',
    default = 'inception', type = str)
    return parser.parse_args()
    
