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
        image_name = self.images[index]
        image_name = os.path.join(self.imageFolder, image_name)
        label_name = self.labels[index]
        label_name = os.path.join(self.labelFolder, label_name)
        
        # image_read = cv2.imread(image_name)
        # label_read = cv2.imread(label_name)
        transform = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()
                        ])
        image_read = Image.open(image_name).convert('RGB')
        label_read = Image.open(label_name).convert('RGB')
        
        # X = torch.Tensor(image_read)
        # Y = torch.Tensor(label_read)

        X = transform(image_read)
        Y = transform(label_read)

        x=torch.reshape(X,(1,400,400))
        y=torch.reshape(Y,(1,400,400))
        return x, y
