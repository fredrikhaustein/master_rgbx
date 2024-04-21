import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader,get_val_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.discordCallback import post_to_discord
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.calculate_matrix import calculate_iou,calculate_iou_per_class

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1209069645685071872/HoOEDai_phExtY0rBdcnvQiIcYTEwS_XugDMBMbR9sgI2zs3fMHjRswARyCR5nUCl1cQ"
os.environ['MASTER_PORT'] = '169710'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)
    val_loader, val_sampler = get_val_loader(engine, RGBXDataset)
    print(config.dataset_path)
    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model=segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        iou_scores = []  # Store IoU scores for each batch

        # Initialize lists to store predictions and ground truths for the epoch
        all_preds = []
        all_gts = []
        iou_scores_per_class = [[] for _ in range(6)]

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            # print(minibatch)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']
            
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)


            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
            else:
                sum_loss += loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

            del loss
            pbar.set_description(print_str, refresh=False)


        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            message = f'Epoch {epoch}/{config.nepochs} training complete. Loss: {sum_loss / len(pbar)}'
            # post_to_discord(DISCORD_WEBHOOK_URL, message)


        # After training loop, we perform the validation
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_gts = []

        with torch.no_grad():
            for minibatch in val_loader:
                imgs = minibatch['data'].numpy()  # Assuming data comes in CPU numpy format
                modal_xs = minibatch['modal_x'].numpy()
                gts = minibatch['label'].cuda()  # Assuming labels are used for computing loss on GPU

                preds, refined_preds = [], []
                for img, modal_x in zip(imgs, modal_xs):
                    pred, _ = sliding_eval_rgbX(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device='cuda')
                    preds.append(pred)

                preds = np.stack(preds)  # Stack predictions into a numpy array
                preds_tensor = torch.from_numpy(preds).cuda()
                refined_preds_tensor = torch.from_numpy(refined_preds).cuda()

        # Compute loss here if applicable
        # loss = criterion(refined_preds_tensor, gts)  # Example loss computation
        # total_val_loss += loss.item()

        # Store predictions and ground truths for metrics computation
        all_val_preds.extend(preds)
        all_val_gts.extend(minibatch['label'].numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_precision = precision_score(all_val_gts, all_val_preds, average='macro')
        val_recall = recall_score(all_val_gts, all_val_preds, average='macro')
        val_f1 = f1_score(all_val_gts, all_val_preds, average='macro')

        # Log validation metrics
        tb.add_scalar('val_loss', avg_val_loss, epoch)
        tb.add_scalar('val_precision', val_precision, epoch)
        tb.add_scalar('val_recall', val_recall, epoch)
        tb.add_scalar('val_f1', val_f1, epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)