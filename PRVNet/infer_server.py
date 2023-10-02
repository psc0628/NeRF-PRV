import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torchvision.models as torchmodels

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

from engine_pretrain import train_one_epoch
import models.fcmae as fcmae
import models.convnextv2 as convnextv2

from PIL import Image
import random
from tqdm import tqdm

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import str2bool, wblue, wgreen, wred

import math

from train_regression import PVBNet

IMG_PATTERN = [
    [1],
    [0, 1],
    [0, 1, 3],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4],
]

patter_id = 2
min_label_value = 13
max_label_value = 58
device = 'cpu'

transform = transforms.Compose([
	transforms.CenterCrop(size=720),
	transforms.ToTensor()])

model_convnext = convnextv2.__dict__['convnextv2_tiny'](
		num_classes=1000,
		drop_path_rate=0.,
		head_init_scale=0.001,
	)

model = PVBNet(convnext_model=model_convnext)

checkpoint_path = './checkpoints/best_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
pretrained_encoder_state_dict = {}
for key in checkpoint.keys():
	pretrained_encoder_state_dict[key[7:]] = checkpoint[key]
model.load_state_dict(pretrained_encoder_state_dict)
model.to(device)

if __name__ == '__main__':
	while True:
		while os.path.isfile('./data/ready_c++.txt') == False:
			time.sleep(0.1)
		time.sleep(1)
		os.remove('./data/ready_c++.txt')

		imgs = []
		for img_idx in IMG_PATTERN[patter_id]:
			img_path = os.path.join('./data/images', f'{img_idx}.png')
			img = Image.open(img_path).convert('RGB')
			img = transform(img)
			imgs.append(img)
			# Single image Copy one
			if len(IMG_PATTERN[patter_id]) == 1:
				imgs.append(img)
		imgs = [img.unsqueeze(0).to(device) for img in imgs]

		with torch.no_grad():
			pred = model(imgs)  # 64*1: float
			pred = torch.nn.functional.sigmoid(pred)  # [0-1] float
			pred = min_label_value + (max_label_value - min_label_value) * pred  # [13-58] float
			pred = torch.round(pred)  # [13,58] int

		print('view budget is ' + str(pred))
		np.savetxt('./data/view_budget.txt', pred, fmt='%d')

		f = open('./data/ready_py.txt', 'a')
		f.close()

