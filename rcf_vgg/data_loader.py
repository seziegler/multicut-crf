import torch
from torch.utils import data
import os
from os.path import join, abspath, splitext, split, isdir, isfile
from PIL import Image
import numpy as np
import cv2
from scipy.io import loadmat

def prepare_image_PIL(im):
    im = im[:,:,::-1] - np.zeros_like(im) # rgb to bgr
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im

def prepare_image_cv2(im):
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
    return im


class BSDS_RCFLoader_original(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/HED-BSDS_PASCAL', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<127.5)] = 2
            lb[lb >= 127.5] = 1
        else:
            img_file = self.filelist[index].rstrip()

        if self.split == "train":
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img

class BSDS_RCFLoader(data.Dataset):
    """
    Dataloader BSDS500
    """
    def __init__(self, root='data/bsds', split='train', transform=False):
        self.root = root
        self.split = split
        self.transform = transform
        if self.split == 'train':
            #self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
            self.filelist = join(self.root, 'train_list.lst')
        elif self.split == 'test':
            self.filelist = join(self.root, 'test_list_only_img.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()
        #self.thick_nms = thick_nms

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        
        #print('V3')
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()

            # get file name
            img_name = img_file[:-4]

            # get NMS mask
            
            nms = Image.open('./NMS_masks/processed/NMS_WS_masks/{}.png'.format(img_name))
            #nms = torch.tensor(np.array(nms)[:,:,0])
            nms = torch.tensor(np.array(nms))

            #print(nms.size())

            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)
            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb>0, lb<127.5)] = 2
            lb[lb >= 127.5] = 1
            
        
            
        else:
            
            img_file = self.filelist[index].rstrip()

            img_name = img_file[:-4]

            # get NMS mask
            
            nms = Image.open('./NMS_masks/processed/NMS_WS_masks/{}.png'.format(img_name))
            #nms = torch.tensor(np.array(nms)[:,:,0])
            nms = torch.tensor(np.array(nms))
           
        if self.split == "train":
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb, img_name, nms
        else:
            img = np.array(Image.open(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_PIL(img)
            return img, img_name, nms
