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

import warnings

warnings.filterwarnings('ignore')

IMG_PATTERN = [
    [1],
    [0, 1],
    [0, 1, 3],
    [0, 1, 2, 3],
    [0, 1, 2, 3, 4],
]


class PVBPretrain(torch.nn.Module):
    def __init__(self, convnext_model, fc_dim=[1000, 500, 250, 100, 1]):
        super().__init__()
        # use base convnext if not specified
        self.encoder = convnext_model
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(fc_dim[0], fc_dim[1]),
            torch.nn.Linear(fc_dim[1], fc_dim[2]),
            torch.nn.Linear(fc_dim[2], fc_dim[3]),
            torch.nn.Linear(fc_dim[3], fc_dim[4]),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc_layer(out)
        return out


class PVBNet(torch.nn.Module):
    def __init__(self, convnext_model, fc_dim=[1000, 500, 250, 100, 1]):
        super().__init__()
        # use base convnext if not specified
        self.encoder = convnext_model
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(fc_dim[0] * 2, fc_dim[0]),
            torch.nn.Linear(fc_dim[0], fc_dim[1]),
            torch.nn.Linear(fc_dim[1], fc_dim[2]),
            torch.nn.Linear(fc_dim[2], fc_dim[3]),
            torch.nn.Linear(fc_dim[3], fc_dim[4]),
        )

    def forward(self, xs):
        """_summary_

        Args:
            xs (list): a list of images, len of list need to be greater than 1

        Returns:
            tensor: batch_size * num_classes, the prediction of the model
        """
        out = []
        for x in xs:
            out.append(self.encoder(x))
        out = torch.stack(out)
        mean = torch.mean(out, dim=0)
        variance = torch.var(out, dim=0)
        out = torch.cat([mean, variance], dim=-1)
        out = self.fc_layer(out)
        return out


class pvbPretrainDataset(Dataset):
    def __init__(self, dataset_root_dir, splits, transform=None, min_label_value=13, max_label_value=58,
                 viewspace_size=64, loss_type='CE'):
        super().__init__()

        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.min_label_value = min_label_value
        self.num_of_class = max_label_value - min_label_value + 1
        self.loss_type = loss_type

        self.training_files = []
        self.gt_files = []

        with open(splits, 'r') as f:
            for line in f.readlines():
                training_file = line.strip()
                for idx in range(0, viewspace_size):
                    self.training_files.append(
                        os.path.join(self.dataset_root_dir, training_file, f'rgbaClip_{idx}.png'))
                    self.gt_files.append(os.path.join(self.dataset_root_dir, training_file, f'view_budget.txt'))

        # # memory
        # self.imgs = []
        # self.gts = []
        #
        # for index in range(0, len(self.training_files)):
        #     gt_hit_bit = np.loadtxt(self.gt_files[index], dtype=int)
        #     gt_hit_bit = gt_hit_bit - self.min_label_value
        #     gt = []
        #     for i in range(0, self.num_of_class):
        #         if i == gt_hit_bit:
        #             gt.append(1)
        #         else:
        #             gt.append(0)
        #     gt = torch.from_numpy(np.asarray(gt)).float()
        #     self.gts.append(gt)
        #
        #     img = Image.open(self.training_files[index]).convert('RGB')
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     self.imgs.append(img)

    def __len__(self):
        return len(self.training_files)

    # # memory
    # def __getitem__(self, index):
    #     return self.imgs[index], self.gts[index]

    # disk
    def __getitem__(self, index):
        gt_hit_bit = np.loadtxt(self.gt_files[index], dtype=int)
        # gt_hit_bit = gt_hit_bit - self.min_label_value
        # gt = []
        # for i in range(0, self.num_of_class):
        #     if i == gt_hit_bit:
        #         gt.append(1)
        #     else:
        #         gt.append(0)
        # gt = torch.from_numpy(np.asarray(gt)).float()

        img = Image.open(self.training_files[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, gt_hit_bit


class pvbDataset(Dataset):
    def __init__(self, dataset_root_dir, splits, pattern=[0, 1, 2, 3, 4], transform=None, min_label_value=13,
                 max_label_value=58, loss_type='CE'):
        super().__init__()

        self.dataset_root_dir = dataset_root_dir
        self.transform = transform
        self.pattern = pattern
        self.min_label_value = min_label_value
        self.num_of_class = max_label_value - min_label_value + 1
        self.loss_type = loss_type

        with open(splits, 'r') as f:
            self.training_files = [line.strip() for line in f.readlines()]

        # self.training_files = self.training_files[0:int(len(self.training_files) * 0.05)]

        # # memory
        # self.imgs = []
        # self.gts = []
        #
        # for index in range(0, len(self.training_files)):
        #     gt_hit_bit = np.loadtxt(os.path.join(self.dataset_root_dir, self.training_files[index], 'view_budget.txt'),
        #                             dtype=int)
        #     gt_hit_bit = gt_hit_bit - self.min_label_value
        #     gt = []
        #     for i in range(0, self.num_of_class):
        #         if i == gt_hit_bit:
        #             gt.append(1)
        #         else:
        #             gt.append(0)
        #     gt = torch.from_numpy(np.asarray(gt)).float()
        #     self.gts.append(gt)
        #
        #     # if pattern is given, then choose images according to pattern
        #     imgs = []
        #     for img_idx in self.pattern:
        #         img_path = os.path.join(self.dataset_root_dir, self.training_files[index],
        #                                 f'rgbaClip_{img_idx}.png')
        #         img = Image.open(img_path).convert('RGB')
        #         if self.transform is not None:
        #             img = self.transform(img)
        #         imgs.append(img)
        #     self.imgs.append(imgs)

    def __len__(self):
        return len(self.training_files)

    # # memory
    # def __getitem__(self, index):
    #     return self.imgs[index], self.gts[index]

    # disk
    def __getitem__(self, index):
        gt_hit_bit = np.loadtxt(os.path.join(self.dataset_root_dir, self.training_files[index], 'view_budget.txt'),
                                dtype=int)
        # gt_hit_bit = gt_hit_bit - self.min_label_value
        # gt = []
        # for i in range(0, self.num_of_class):
        #     if i == gt_hit_bit:
        #         gt.append(1)
        #     else:
        #         gt.append(0)
        # gt = torch.from_numpy(np.asarray(gt)).float()

        # if pattern is given, then choose images according to pattern
        imgs = []
        for img_idx in self.pattern:
            img_path = os.path.join(self.dataset_root_dir, self.training_files[index],
                                    f'rgbaClip_{img_idx}.png')
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

        return imgs, gt_hit_bit


def save_checkpoint(model, args, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': args
    }
    torch.save(checkpoint, filepath)


def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation step')

    # Model parameters
    parser.add_argument('--model', default='convnextv2_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=720, type=int,
                        help='image input size')
    parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_embed_dim', type=int, default=512)
    parser.add_argument("--pre_train", action="store_true", help="Run with one image infer encoder.")
    parser.add_argument("--ImageNet", action="store_true", help="ImageNet pretrained weights.")
    parser.add_argument("--premodel_file", default='', type=str, help='pre_train model file path')
    parser.add_argument("--resnet101", action="store_true", help="Encoder as resnet101 pretrain.")
    parser.add_argument("--resnet50", action="store_true", help="Encoder as resnet50 pretrain.")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--loss_type', type=str, default="MSE", help="Run with xxx loss.")
    parser.add_argument('--focal_loss', action="store_true", help="Run with focal loss.")
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='gamma in focal loss')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/ConvNeXt-V2/data', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--viewspace_size', default=49, type=int, help='num of images for each object in dataset')
    parser.add_argument('--min_label_value', default=13, type=int, help='min num of required views')
    parser.add_argument('--max_label_value', default=58, type=int, help='max num of required views')
    parser.add_argument('--pattern_idx', default=4, type=int,
                        help='index of image pattern [0-4], num of initial views minus one [1-5]')

    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--distributed', default=False, type=bool, help='whether to use distributed training')
    return parser


def check_accuracy(model, data_loader, device, log_writer, args, epoch, loss_scaler, train_check):
    model.eval()
    correct = 0
    loss = 0
    l1_loss_value = 0
    perd_mean = 0
    total = 0

    l1_loss_value_list = []
    pred_value_list = []

    iteration = 0
    with torch.no_grad():
        for data in data_loader:
            imgs, label = data
            if args.pre_train:
                imgs = imgs.to(device)
            else:
                imgs = [img.to(device) for img in imgs]
            label = label.unsqueeze(1).to(device)  # 64*46

            label_view_budget = label  # 64*1: [0,46] int
            label = label.float()

            pred = model(imgs)  # 64*1: float
            pred = torch.nn.functional.sigmoid(pred) # 64*1: [0-1] float
            pred = args.min_label_value + (args.max_label_value - args.min_label_value) * pred # 64*1: [0-46] float

            loss += loss_scaler(pred, label)

            pred_view_budget = torch.round(pred) # 64*1: [0,46] int

            l1_loss_value += torch.abs(pred_view_budget - label_view_budget).sum()
            
            l1_loss_value_batch = torch.abs(pred_view_budget - label_view_budget)
            for value in l1_loss_value_batch:
                l1_loss_value_list.append(value.item())

            perd_mean += pred_view_budget.sum()
            for value in pred_view_budget:
                pred_value_list.append(value.item())

            correct += (pred_view_budget == label_view_budget).sum().item()
            total += label.size(0)

            # print(pred_view_budget.shape)
            # print(label_view_budget.shape)
            # print(l1_loss_value)
            # print(perd_mean)
            # print(correct)
            # print(total)

            iteration += 1
            if train_check:
                if iteration > 16:
                    break

    acc = correct / total
    l1_dis = l1_loss_value / total
    loss = loss / total

    l1_dis_var = 0
    for l1_loss in l1_loss_value_list:
        l1_dis_var += (l1_dis-l1_loss) * (l1_dis-l1_loss)
    l1_dis_var /= total - 1
    l1_dis_var = math.sqrt(l1_dis_var)

    perd_mean /= total
    pred_var = 0
    for pred in pred_view_budget:
        pred_var += (pred-perd_mean) * (pred-perd_mean)
    pred_var/= total - 1
    pred_var = math.sqrt(pred_var)

    print(f'Accuracy: {acc}, L1_distance: {l1_dis}, L1_std: {l1_dis_var}, pred_mean: {perd_mean}, perd_std: {pred_var}, Loss:{loss}')

    if log_writer is not None:
        if train_check:
            log_writer.add_scalar('train/acc', acc, epoch)
            log_writer.add_scalar('train/l1', l1_dis, epoch)
            log_writer.add_scalar('train/l1_std', l1_dis_var, epoch)
            log_writer.add_scalar('train/pred_mean', perd_mean, epoch)
            log_writer.add_scalar('train/perd_std', pred_var, epoch)
            log_writer.add_scalar('train/loss', loss, epoch)
        else:
            log_writer.add_scalar('val/acc', acc, epoch)
            log_writer.add_scalar('val/l1', l1_dis, epoch)
            log_writer.add_scalar('val/l1_std', l1_dis_var, epoch)
            log_writer.add_scalar('val/pred_mean', perd_mean, epoch)
            log_writer.add_scalar('val/perd_std', pred_var, epoch)
            log_writer.add_scalar('val/loss', loss, epoch)

    return l1_dis


def train_one_epoch(model, data_loader,
                    optimizer, device, epoch, loss_scaler,
                    log_writer, args):
    model.train(True)

    t = tqdm(
        data_loader,
        leave=True
    )
    data_iter = 0
    for data in t:
        optimizer.zero_grad()

        # if data_iter % args.update_freq == 0:
        #     utils.adjust_learning_rate(optimizer, data_iter / len(data_loader) + epoch, args)

        imgs, label = data
        if args.pre_train:
            imgs = imgs.to(device)
        else:
            imgs = [img.to(device) for img in imgs]
        label = label.unsqueeze(1).to(device)

        label = Variable(label.float().data, requires_grad=True)

        pred = model(imgs)

        pred = torch.nn.functional.sigmoid(pred)

        pred = args.min_label_value + (args.max_label_value - args.min_label_value) * pred

        loss = loss_scaler(pred, label)

        data_iter += 1

        loss.backward()
        optimizer.step()

        description = f"Epoch {epoch} | Loss {loss}"
        t.set_description_str(wblue(description))
        # t.set_postfix(loss=loss_value())


def main(args):
    # utils.init_distributed_mode(args)

    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_train = transforms.Compose([
        transforms.CenterCrop(size=args.input_size),
        # transforms.Resize(size=224),
        transforms.ToTensor()])

    transform_val = transforms.Compose([
        transforms.CenterCrop(size=args.input_size),
        # transforms.Resize(size=224),
        transforms.ToTensor()])

    if args.pre_train:
        train_dataset = pvbPretrainDataset(args.data_path, os.path.join(args.data_path, 'train_split.txt'),
                                           transform=transform_train, min_label_value=args.min_label_value,
                                           max_label_value=args.max_label_value, viewspace_size=args.viewspace_size,
                                           loss_type=args.loss_type)
        val_dataset = pvbPretrainDataset(args.data_path, os.path.join(args.data_path, 'val_split.txt'),
                                         transform=transform_val, min_label_value=args.min_label_value,
                                         max_label_value=args.max_label_value, viewspace_size=args.viewspace_size,
                                         loss_type=args.loss_type)
    else:
        train_dataset = pvbDataset(args.data_path, os.path.join(args.data_path, 'train_split.txt'),
                                   pattern=IMG_PATTERN[args.pattern_idx], transform=transform_train,
                                   min_label_value=args.min_label_value, max_label_value=args.max_label_value,
                                   loss_type=args.loss_type)
        val_dataset = pvbDataset(args.data_path, os.path.join(args.data_path, 'val_split.txt'),
                                 pattern=IMG_PATTERN[args.pattern_idx], transform=transform_val,
                                 min_label_value=args.min_label_value, max_label_value=args.max_label_value,
                                 loss_type=args.loss_type)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    # define the model
    model_convnext = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        head_init_scale=args.head_init_scale,
    )
    # model_convnext.to(device)
    # model_convnext = torch.nn.DataParallel(model_convnext).to(device)

    resnet = None
    if args.resnet101:
        resnet = torchmodels.resnet101(pretrained=True)
    if args.resnet50:
        resnet = torchmodels.resnet50(pretrained=True)

    model = None
    if args.pre_train:
        if args.resnet101 or args.resnet50:
            model = PVBPretrain(convnext_model=resnet)
        else:
            model = PVBPretrain(convnext_model=model_convnext)
    else:
        if args.resnet101 or args.resnet50:
            model = PVBNet(convnext_model=resnet)
        else:
            model = PVBNet(convnext_model=model_convnext)

    # imgs, labels = train_dataset[0]
    # for img in imgs:
    #     print(model(img.unsqueeze(0)))
    # return

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params:', n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(train_dataset) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    loss_scaler = None

    if args.loss_type == 'L1':
        loss_scaler = torch.nn.L1Loss()
    elif args.loss_type == 'MSE':
        loss_scaler = torch.nn.MSELoss()

    if not args.pre_train and args.premodel_file != '':
        checkpoint_path = args.premodel_file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
        if args.ImageNet:
            model.encoder.load_state_dict(checkpoint)
        else:
            pretrained_encoder_state_dict = {}
            for key in checkpoint.keys():
                if 'encoder' in key:
                    if args.model == 'convnextv2_tiny':
                        pretrained_encoder_state_dict[key[15:]] = checkpoint[key]
                    elif args.model == 'convnextv2_base':
                        pretrained_encoder_state_dict[key[8:]] = checkpoint[key]
            model.encoder.load_state_dict(pretrained_encoder_state_dict)

        # # Not Retrain encoder
        # for param in model.encoder.parameters():
        #     param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    l1_best = 10000
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # if log_writer is not None:
        #     log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        l1_curr = check_accuracy(model=model, data_loader=data_loader_val, device=device, log_writer=log_writer,
                                 args=args,
                                 epoch=epoch, loss_scaler=loss_scaler, train_check=False)
        if l1_curr <= l1_best:
            save_checkpoint(model=model, args=args, filepath=args.output_dir + '/best_checkpoint.pth')
            l1_best = l1_curr

        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                 loss_scaler=loss_scaler, epoch=epoch)

        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            check_accuracy(model=model, data_loader=data_loader_train, device=device, log_writer=log_writer, args=args,
                           epoch=epoch, loss_scaler=loss_scaler, train_check=True)
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                 'epoch': epoch,
        #                 'n_parameters': n_parameters}
        # if args.output_dir and utils.is_main_process():
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('l1_best '+str(l1_best))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

# python -W ignore train.py --model convnextv2_base --device cpu
# python train.py --model convnextv2_base --device cuda --pre_train
