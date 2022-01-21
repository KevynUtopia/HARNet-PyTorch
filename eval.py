import argparse
import glob
import random
import os
from PIL import Image
import numpy as np
import time
import datetime
import sys
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import itertools
import matplotlib.pyplot as plt
import pdb
import skimage.metrics
from tqdm import tqdm

def eval(model):
    lr = "./data/testset/6x6_256/"
    hr = "./data/testset/3x3_256/"
    num, psnr, ssim, mse, nmi= 0, 0, 0, 0, 0
    T_1 = transforms.Compose([ transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
                # transforms.Resize([128, 128])
                 ])
    T_2 = transforms.Compose([ transforms.ToTensor(),                         
                  transforms.Normalize((0.5), (0.5))])
    for i in tqdm(range(297)):
        lr_path = os.path.join(lr, str(i)+"_3.png")
        hr_path = os.path.join(hr, str(i)+"_6.png")
        if os.path.isfile(lr_path) and os.path.isfile(hr_path):
            lr_img = Image.open(lr_path).convert('L')
            hr_img = Image.open(hr_path).convert('L')
            
            lr_img = T_1(lr_img).cuda().unsqueeze(0)
            hr_img = T_2(hr_img).cuda().unsqueeze(0)
            
            sr_img = model(lr_img)

            yimg = sr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            gtimg = hr_img.cpu().detach().numpy().squeeze(0).squeeze(0)
            psnr += (skimage.metrics.peak_signal_noise_ratio(yimg, gtimg))
            ssim += (skimage.metrics.structural_similarity(yimg, gtimg))
            mse += (skimage.metrics.mean_squared_error(yimg, gtimg))
            nmi += (skimage.metrics.normalized_mutual_information(yimg, gtimg))
            num += 1
    print(" PSNR: %.4f SSIM: %.4f MSE: %.4f NMI: %.4f"%(psnr/num, ssim/num, mse/num, nmi/num))