import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

class DatasetLoader(Dataset):
    def __init__(self,folder_path,source,target,scale,transform=None):
        self.folder_path = folder_path
        self.hr_folder = self.folder_path+target
        self.lr_folder = self.folder_path+source
        self.files_list_hr = os.listdir(self.hr_folder)#.sort()
        self.files_list_lr = os.listdir(self.lr_folder)#.sort()
        self.files_list_hr.sort()
        self.files_list_lr.sort()
        self.scale = scale
        self.min_width = 600
        self.transformation = transform
        print("hr list:",self.files_list_hr)
        print("lr list:",self.files_list_lr)
        #self.transform_images_hr = CenterCrop((2040,600))
        #self.transform_images_lr = CenterCrop((2040//scale,600//scale))
        print("hr len: ",len(self.files_list_hr)," lr len:",len(self.files_list_lr))
        self.no_of_files = len(self.files_list_hr)

    def __getitem__(self, id):
        file_name_hr = self.files_list_hr[id]
        file_name_lr = self.files_list_lr[id]
        #print("hr name:",file_name_hr," lr name:",file_name_lr)
        hr_img = cv2.imread(self.hr_folder + "/" + file_name_hr)
        lr_img = cv2.imread(self.lr_folder + "/" + file_name_lr)

        hr_img = self.transform(hr_img,1)
        lr_img = self.transform(lr_img,0)

        #print("aug lr:",lr_img.shape," aug hr:",hr_img.shape)
        lr_img, hr_img =self.RGB_np2Tensor(lr_img, hr_img)

        #print("to ten lr:",lr_img.shape," aug hr:",hr_img.shape)
        return lr_img, hr_img
    def __len__(self):
        return self.no_of_files

    def transform(self,img,hr):
        #this funtion reshapes images to a standard shape so that it can be tranied in batches in network
        h,w,c = img.shape
        if hr==0:
             width_re = self.min_width//self.scale
        else:
             width_re = self.min_width
        if h>w:
             temp_img = img[:,0:width_re,:]
        else:
             temp_img = img[0:width_re,:,:]

        return temp_img

    def RGB_np2Tensor(self,imgIn, imgTar):
        ts = (2,0, 1)
        if imgIn.shape[1]!=510:
             ts=(2,1,0)
      
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
        imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
        return imgIn, imgTar

    def augment(self,imgIn, imgTar):
        if random.random() < 0.3:
            imgIn = imgIn[:, ::-1, :]
            imgTar = imgTar[:, ::-1, :]
        if random.random() < 0.3:
            imgIn = imgIn[::-1, :, :]
            imgTar = imgTar[::-1, :, :]
        return imgIn, imgTar


class TestDatasetLoader(Dataset):
    def __init__(self,folder_path,source,target,scale,transform=None):
        self.folder_path = folder_path
        self.hr_folder = self.folder_path+target
        self.lr_folder = self.folder_path+source
        self.files_list_hr = os.listdir(self.hr_folder)#.sort()
        self.files_list_lr = os.listdir(self.lr_folder)#.sort()
        self.files_list_hr.sort()
        self.files_list_lr.sort()
        self.scale = scale
        self.min_width = 600
        self.transformation = transform
        print("hr list:",self.files_list_hr)
        print("lr list:",self.files_list_lr)
        #self.transform_images_hr = CenterCrop((2040,600))
        #self.transform_images_lr = CenterCrop((2040//scale,600//scale))
        print("hr len: ",len(self.files_list_hr)," lr len:",len(self.files_list_lr))
        self.no_of_files = len(self.files_list_hr)

    def __getitem__(self, id):
        file_name_hr = self.files_list_hr[id]
        file_name_lr = self.files_list_lr[id]
        #print("hr name:",file_name_hr," lr name:",file_name_lr)
        hr_img = cv2.imread(self.hr_folder + "/" + file_name_hr)
        lr_img = cv2.imread(self.lr_folder + "/" + file_name_lr)

        #hr_img = self.transform(hr_img,1)
        #lr_img = self.transform(lr_img,0)

        #print("aug lr:",lr_img.shape," aug hr:",hr_img.shape)
        lr_img, hr_img =self.RGB_np2Tensor(lr_img, hr_img)

        #print("to ten lr:",lr_img.shape," aug hr:",hr_img.shape)
        return lr_img, hr_img

    def __len__(self):
        return self.no_of_files

    def transform(self,img,hr):
        #this funtion reshapes images to a standard shape so that it can be tranied in batches in network
        h,w,c = img.shape
        if hr==0:
             width_re = self.min_width//self.scale
        else:
             width_re = self.min_width
        if h>w:
             temp_img = img[:,0:width_re,:]
        else:
             temp_img = img[0:width_re,:,:]

        return temp_img

    def RGB_np2Tensor(self,imgIn, imgTar):
        ts = (2,0, 1)
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
        imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(1.0)
        return imgIn, imgTar

