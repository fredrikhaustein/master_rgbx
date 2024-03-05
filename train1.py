import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.logger import get_logger

from tensorboardX import SummaryWriter

logger = get_logger()
cudnn.benchmark = True
seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Initialize data loader
train_loader = get_train_loader(RGBXDataset)

# Initialize TensorBoard
tb_dir = config.tb_dir + '/{}'.format(time.strftime("%Y%m%d_%H-%M-%S", time.localtime()))
tb = SummaryWriter(log_dir=tb_dir)

# Initialize model, criterion, and optimizer
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
if torch.cuda.is_available():
    model = DataParallel(model).cuda()
params_list = []
params_list = group_weight(params_list, model, nn.BatchNorm2d, config.lr)
optimizer = torch.optim.AdamW(params_list, lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay) if config.optimizer == 'AdamW' else torch.optim.SGD(params_list, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

# Initialize learning rate policy
total_iteration = config.nepochs * len(train_loader)
lr_policy = WarmUpPolyLR(config.lr, config.lr_power, total_iteration, config.warm_up_epoch * len(train_loader))

# Training loop
logger.info('Begin training:')
for epoch in range(1, config.nepochs + 1):
    model.train()
    sum_loss = 0
    pbar = tqdm(train_loader, file=sys.stdout)
    for minibatch in pbar:
        imgs, gts, modal_xs = minibatch['data'].cuda(), minibatch['label'].cuda(), minibatch['modal_x'].cuda()
        loss = model(imgs, modal_xs, gts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        pbar.set_description(f'Epoch {epoch}/{config.nepochs}: Loss: {sum_loss / (pbar.n + 1):.4f}')
    tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
    # Checkpoint saving
    if epoch % config.checkpoint_step == 0 or epoch == config.nepochs:
        checkpoint_path = osp.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')
