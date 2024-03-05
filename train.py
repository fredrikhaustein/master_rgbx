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
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from utils.calculate_matrix import calculate_iou,calculate_iou_per_class

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

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
            
            # print("Unique labels in gts:", gts.unique())
            # n_class = 2

            # valid_labels = (gts >= 0) & (gts < n_class)
            # if not valid_labels.all():
            #     invalid_labels = gts[~valid_labels]
            #     print(f"Invalid labels found: {invalid_labels.unique()}")
            #     raise ValueError("Labels outside of expected range detected.")

            # Assuming CrossEntropyLoss is used and 255 should be ignored
            # criterion = nn.CrossEntropyLoss(ignore_index=255)

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

            # with torch.no_grad():
            #     preds = torch.argmax(model(imgs, modal_xs), dim=1)  # Convert logits to predictions

            # Store predictions and ground truths
            # all_preds.append(preds.cpu().numpy())
            # all_gts.append(gts.cpu().numpy())

            # Assuming all_gts and all_preds are lists of arrays or lists, you first concatenate them into a single array
            # all_gts_array = np.concatenate([np.array(gt) for gt in all_gts])
            # all_preds_array = np.concatenate([np.array(pred) for pred in all_preds])

            # Now you can flatten them
            # all_gts_flat = all_gts_array.flatten()
            # all_preds_flat = all_preds_array.flatten()

            # Calculate IoU and store
            # iou = calculate_iou(preds.cpu(), gts.cpu(), n_classes=config.num_classes)  # Adjust n_classes as per your dataset
            # ious = calculate_iou_per_class(preds.cpu(), gts.cpu(), n_classes=config.num_classes)

            # Compute metrics precision, recall and f1 score
            # precision = precision_score(all_gts_flat, all_preds_flat, average='macro', labels=np.unique(all_gts))
            # recall = recall_score(all_gts_flat, all_preds_flat, average='macro', labels=np.unique(all_gts))
            # f1 = f1_score(all_gts_flat, all_preds_flat, average='macro', labels=np.unique(all_gts))

            
            # iou_scores.append(iou)

            # for i in range(len(optimizer.param_groups)):
                # optimizer.param_groups[i]['lr'] = lr

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

        # avg_iou = np.mean([iou for iou in iou_scores if not np.isnan(iou)])
        # avg_iou_per_class = [np.nanmean(cls_scores) if cls_scores else np.nan for cls_scores in iou_scores_per_class]

        # Define the file path for storing the metrics
        # metrics_file_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results_metrix_output/avg_iou_per_class.txt'

        # Ensure the directory exists
        # os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

        # Append avg_iou_per_class to the file with epoch information
        # with open(metrics_file_path, 'a') as file:  # Open in append mode
        #     file.write(f"Epoch: {epoch}\n")
        #     for class_index, iou in enumerate(avg_iou_per_class):
        #         file.write(f"Class {class_index}: {iou}\n")

        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            # tb.add_scalar('train_iou', avg_iou, epoch)
            # # Loop through each class's IoU and log it separately
            # for cls_index, cls_iou in enumerate(avg_iou_per_class):
            #     tb.add_scalar(f'train_iou_class_{cls_index}', cls_iou, epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                config.log_dir,
                                                config.log_dir_link)