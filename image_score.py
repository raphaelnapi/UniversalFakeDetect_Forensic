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
from PIL import Image, ImageDraw, ImageFont
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import logging

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

def preprocess_image(img_path: str, arch: str):
    img_mosaic = []
    patch_mosaic = []
    positions = []

    stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"

    transform = transforms.Compose([
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
    ])

    with Image.open(img_path).convert("RGB") as img:
        lim_x = max((img.size[0] // 224) * 224, 224)
        lim_y = max((img.size[1] // 224) * 224, 224)

        for x in range(0, lim_x, 224):
            for y in range(0, lim_y, 224):
                img_cropped = img.crop((x, y, x + 224, y + 224))
                img_transformed = transform(img_cropped)
                img_mosaic.append(img_cropped)
                patch_mosaic.append(img_transformed.unsqueeze(0))
                positions.append((x, y))

        remaining_width_box = (lim_x, 0, img.size[0], lim_y)
        remaining_height_box = (0, lim_y, img.size[0], img.size[1])
        remaining_width_image = img.crop(remaining_width_box)
        remaining_height_image = img.crop(remaining_height_box)

    return img_mosaic, patch_mosaic, positions, img.size, remaining_height_image, remaining_height_box, remaining_width_image, remaining_width_box

def draw_patch(img: Image, img_size: tuple, text: str):
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img_size[0], img_size[1])], outline='red', width=4)
    font = ImageFont.load_default(20)
    draw.text((20, 20), text, fill='yellow', font=font)

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
    parser.add_argument('--img', type=str, default=None, help='input image filepath')
    parser.add_argument('--out', type=str, default='out.png', help='output image filepath')
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--thres', type=int, default=50, help='Threshold (detection sensibility)')

    opt = parser.parse_args()

    logger = logging.getLogger("AI_Image_Analysis")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler("ufd_forensics.log", mode='a', encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f':: Start processing image: {opt.img}')

    print(':: OPTIONS')
    print(f'--IMG: {opt.img}')
    print(f'--ARCH: {opt.arch}')
    print(f'--CKPT: {opt.ckpt}')
    print(f'--THRES: {opt.thres}')
    print(f'--OUT: {opt.out}')

    logger.info(':: OPTIONS')
    logger.info(f'--IMG: {opt.img}')
    logger.info(f'--ARCH: {opt.arch}')
    logger.info(f'--CKPT: {opt.ckpt}')
    logger.info(f'--THRES: {opt.thres}')
    logger.info(f'--OUT: {opt.out}')
    print()

    print(':: Loading model ', end='')
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print(" [OK]")
    logger.info(':: Model loaded')
    model.eval()
    model.cuda()
    set_seed()
    print(':: Preprocessing image...', end='')
    img_mosaic, patch_mosaic, positions, img_size, remaining_height_image, remaining_height_box, remaining_width_image, remaining_width_box = preprocess_image(opt.img, opt.arch)
    print(' [OK]')
    logger.info(':: Image preprocessed')
    print()

    img_out = Image.new("RGB", img_size)

    for img, patch, (x, y) in zip(img_mosaic, patch_mosaic, positions):

        if img.size == (224, 224):
            score = model(patch.cuda()).sigmoid().item()
            percent = float(score) * 100
            if percent > opt.thres:
                print()
                print(f':: Patch ({x}, {y}, {x + 224}, {y + 224})')
                print(f'Score (Model {opt.arch}): {score}')
                print(f'Estimated AI Generation Probability: {percent:.2f}%')

                logger.info(f':: Patch ({x}, {y}, {x + 224}, {y + 224})')
                logger.info(f'Score (Model {opt.arch}): {score}')
                logger.info(f'Estimated AI Generation Probability: {percent:.2f}%')

                draw_patch(img, (224, 224), f'{percent:.2f}%')
            
        img_out.paste(img, (x, y))

    draw_patch(remaining_height_image, (remaining_height_box[2] - remaining_height_box[0], remaining_height_box[3] - remaining_height_box[1]), 'EXCLUDED')
    draw_patch(remaining_width_image, (remaining_width_box[2] - remaining_width_box[0], remaining_width_box[3] - remaining_width_box[1]), 'EXCLUDED')
    img_out.paste(remaining_height_image, remaining_height_box)
    img_out.paste(remaining_width_image, remaining_width_box)
    img_out.save(opt.out, 'png')