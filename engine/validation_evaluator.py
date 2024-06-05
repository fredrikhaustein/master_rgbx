import os
import cv2
import numpy as np
import torch
from timm.models.layers import to_2tuple
from utils.transforms import pad_image_to_shape, normalize
from utils.crf import apply_crf

def sliding_val_rgbX(img, modal_x, class_num, multi_scales, scale_process_func, crop_size, stride_rate, device=None):
    crop_size = to_2tuple(crop_size)
    ori_rows, ori_cols, _ = img.shape
    processed_pred = np.zeros((ori_rows, ori_cols, class_num))

    for s in multi_scales:
        img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if len(modal_x.shape) == 2:
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        else:
            modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        new_rows, new_cols, _ = img_scale.shape
        processed_pred += scale_process_func(img_scale, modal_x_scale, class_num, (ori_rows, ori_cols),
                                             crop_size, stride_rate, device)
    
    pred = processed_pred.argmax(2)
    refined_pred = apply_crf(img, processed_pred)
    
    return pred, refined_pred

def scale_process_rgbX(img, modal_x, class_num, ori_shape, crop_size, stride_rate, device=None):
    new_rows, new_cols, c = img.shape
    long_size = new_cols if new_cols > new_rows else new_rows

    if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
        input_data, input_modal_x, margin = process_image_rgbX(img, modal_x, crop_size)
        score = val_func_process_rgbX(input_data, input_modal_x, class_num, device) 
        score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
    else:
        stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
        img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
        modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

        pad_rows = img_pad.shape[0]
        pad_cols = img_pad.shape[1]
        r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
        c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
        data_scale = torch.zeros(class_num, pad_rows, pad_cols).cuda(device)

        for grid_yidx in range(r_grid):
            for grid_xidx in range(c_grid):
                s_x = grid_xidx * stride[0]
                s_y = grid_yidx * stride[1]
                e_x = min(s_x + crop_size[0], pad_cols)
                e_y = min(s_y + crop_size[1], pad_rows)
                s_x = e_x - crop_size[0]
                s_y = e_y - crop_size[1]
                img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                if len(modal_x_pad.shape) == 2:
                    modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x]
                else:
                    modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x,:]

                input_data, input_modal_x, tmargin = process_image_rgbX(img_sub, modal_x_sub, crop_size)
                temp_score = val_func_process_rgbX(input_data, input_modal_x, class_num, device)
                
                temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                        tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                data_scale[:, s_y: e_y, s_x: e_x] += temp_score
        score = data_scale
        score = score[:, margin[0]:(score.shape[1] - margin[1]),
                margin[2]:(score.shape[2] - margin[3])]

    score = score.permute(1, 2, 0)
    data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

    return data_output

def val_func_process_rgbX(input_data, input_modal_x, class_num, device=None):
    input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
    input_data = torch.FloatTensor(input_data).cuda(device)

    input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
    input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)

    # Assume val_func is a callable model that has been defined elsewhere
    with torch.cuda.device(input_data.get_device()):
        val_func.eval()
        val_func.to(input_data.get_device())
        with torch.no_grad():
            score = val_func(input_data, input_modal_x)
            score = score[0]
            if is_flip:  # Assuming is_flip is a predefined variable
                input_data = input_data.flip(-1)
                input_modal_x = input_modal_x.flip(-1)
                score_flip = val_func(input_data, input_modal_x)
                score_flip = score_flip[0]
                score += score_flip.flip(-1)
            score = torch.exp(score)
    return score

def process_image_rgbX(img, modal_x, crop_size=None):
    p_img = img
    p_modal_x = modal_x

    if img.shape[2] < 3:
        im_b = p_img
        im_g = p_img
        im_r = p_img
        p_img = np.concatenate((im_b, im_g, im_r), axis=2)

    norm_mean = [0.485, 0.456, 0.406]  # Assuming normalization mean and std are predefined
    norm_std = [0.229, 0.224, 0.225]
    p_img = normalize(p_img, norm_mean, norm_std)
    if len(modal_x.shape) == 2:
        p_modal_x = normalize(p_modal_x, 0, 1)
    else:
        p_modal_x = normalize(p_modal_x, norm_mean, norm_std)

    if crop_size is not None:
        p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
        p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
        p_img = p_img.transpose(2, 0, 1)
        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)  # 3 H W
    
        return p_img, p_modal_x, margin

    p_img = p_img.transpose(2, 0, 1)  # 3 H W

    if len(modal_x.shape) == 2:
        p_modal_x = p_modal_x[np.newaxis, ...]
    else:
        p_modal_x = p_modal_x.transpose(2, 0, 1)

    return p_img, p_modal_x
