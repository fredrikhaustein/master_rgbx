import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import generate_random_crop_pos, random_crop_pad_to_shape, normalize

class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, rgb, gt):
        rgb, gt = random_mirror(rgb, gt)
        if config.train_scale_array is not None:
            rgb, gt, scale = random_scale(rgb, gt, config.train_scale_array)

        rgb = normalize(rgb, self.norm_mean, self.norm_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)

        p_rgb = p_rgb.transpose(2, 0, 1)
        
        return p_rgb, p_gt

class ValPre(object):
    def __call__(self, rgb, gt):
        return rgb, gt

class TestPre(object):
    def __call__(self, rgb, gt):
        return rgb, gt

# Modify random_mirror and random_scale functions to exclude modal_x
def random_mirror(rgb, gt):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
    return rgb, gt

def random_scale(rgb, gt, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return rgb, gt, scale

def get_train_loader(engine, dataset):
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'val_source': config.val_source,
        'eval_source': config.eval_source,
        'class_names': config.class_names
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(data_setting, "train", train_preprocess)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)
    return train_loader, train_sampler

def get_val_loader(engine, dataset):
    # Similar modifications as get_train_loader
    data_setting = {
        'rgb_root': config.rgb_root_folder,
        'rgb_format': config.rgb_format,
        'gt_root': config.gt_root_folder,
        'gt_format': config.gt_format,
        'transform_gt': config.gt_transform,
        'class_names': config.class_names,
        'train_source': config.train_source,
        'val_source': config.val_source,
        'eval_source': config.eval_source,
        'class_names': config.class_names
    }
    val_preprocess = ValPre()
    val_dataset = dataset(data_setting, "val", val_preprocess)

    val_sampler = None
    is_shuffle = False
    batch_size = config.batch_size

    if engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = config.batch_size // engine.world_size

    val_loader = data.DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 shuffle=is_shuffle,
                                 pin_memory=True,
                                 sampler=val_sampler)
    return val_loader, val_sampler