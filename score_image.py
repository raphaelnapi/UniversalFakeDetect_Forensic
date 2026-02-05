import warnings
warnings.filterwarnings('ignore')

import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}
 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)


def preprocess_image(img_path: str, jpeg_quality: int, gaussian_sigma: float, arch: str):
    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
    
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])

    img = Image.open(img_path).convert("RGB")

    if gaussian_sigma is not None:
        img = gaussian_blur(img, gaussian_sigma)

    if jpeg_quality is not None:
        img = png2jpg(img, jpeg_quality)

    img_ret = transform(img)
    return img_ret.unsqueeze(0)


if __name__ == '__main__':
    print('=' * 70)
    print(' Universal Fake Detect (UFD)')
    print(' Single Image Inference — AI-Generated Content Assessment')
    print('-' * 70)
    print(' Model      : CLIP ViT-L/14 (frozen)')
    print(' Classifier : UFD pretrained head')
    print('=' * 70)
    print()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    print(':: OPTIONS')
    print(f'--IMG: {opt.img}')
    print(f'--ARCH: {opt.arch}')
    print(f'--CKPT: {opt.ckpt}')
    print(f'--JPEG_QUALITY: {opt.jpeg_quality}')
    print(f'--GAUSSIAN_SIGMA: {opt.gaussian_sigma}\n')

    print(':: Loading model ', end='')
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print (" [OK]")
    model.eval()
    model.cuda()
    set_seed()
    print(':: Preprocessing image...', end='')
    img = preprocess_image(opt.img, opt.jpeg_quality, opt.gaussian_sigma, opt.arch)
    print(' [OK]')
    print()

    score = model(img.cuda()).sigmoid().item()
    percent = float(score) * 100
    print(f'Score (Model {opt.arch}): {score}')
    print(f'Estimated AI Generation Probability: {percent:.2f}%')