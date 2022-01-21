import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MyDateSet(Dataset):
    def __init__(self, imageFolder, labelFolder):
        self.imageFolder = imageFolder
        self.images = os.listdir(imageFolder)
        self.labelFolder = labelFolder
        self.labels = os.listdir(labelFolder)
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        lr = self.images[index]
        image_name = os.path.join(self.imageFolder, lr)
        hr = lr.replace("_3", "_6")
        # label_name = self.labels[index]
        label_name = os.path.join(self.labelFolder, hr)
        
        # image_read = cv2.imread(image_name)
        # label_read = cv2.imread(label_name)
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))
                        ])
        image_read = Image.open(image_name).convert('L')
        label_read = Image.open(label_name).convert('L')
        
        X = transform(image_read)
        Y = transform(label_read)

        return X, Y
