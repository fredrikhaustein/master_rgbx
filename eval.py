import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from config import config
from utils.crf import apply_crf, convert_predictions_to_probs
from utils.visulize_crf_pred import visualize_and_save_predictions
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img, get_class_colors,print_metrics, get_class_colors_six
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import TestPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred, refined_pred = self.sliding_eval_rgbX_invert(img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)
        
        # visualize_and_save_predictions(img, pred, refined_pred, '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/crf_results', get_class_colors_six())

        print(pred.shape)
        print(label.shape)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        # Additional code for evaluating 'refined_pred'
        hist_tmp_refined, labeled_tmp_refined, correct_tmp_refined = hist_info(config.num_classes, refined_pred, label)
        results_dict_refined = {'hist': hist_tmp_refined, 'labeled': labeled_tmp_refined, 'correct': correct_tmp_refined}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path + '_color')
            ensure_dir(self.save_path + '_refined')  # Directory for raw refined predictions
            ensure_dir(self.save_path + '_refined_color')  # Directory for color-coded refined predictions

            fn = name + '.png'

            # Save colored result for pred
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors_six()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path + '_color', fn))

            # Save raw result for pred
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

            # Save colored result for refined_pred
            refined_result_img = Image.fromarray(refined_pred.astype(np.uint8), mode='P')
            refined_result_img.putpalette(palette_list)  # Reuse the palette from pred
            refined_result_img.save(os.path.join(self.save_path + '_refined_color', fn))

            # Save raw result for refined_pred
            # cv2.imwrite(os.path.join(self.save_path + '_refined', fn), refined_pred)
            # logger.info('Save the refined image ' + fn)

        if self.show_image:
            # print(self.dataset)
            colors = self.dataset.get_class_colors()
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict, results_dict_refined

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        # print(count)
        # Compute overall IoU, mean IoU, etc.
        iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc, precision, recall, f1_score, mean_precision, mean_recall, mean_f1_score, overall_accuracy = compute_score(hist, correct, labeled)
        # result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, dataset.class_names, show_no_back=False)
        result_line = print_metrics(iou, recall, f1_score, precision, overall_accuracy,class_names=config.class_names, no_print=False)

        # Define the directory and file paths
        directory_path = 'results_metric_output'
        os.makedirs(directory_path, exist_ok=True)  # Ensure the directory exists
        metrics_file_path = os.path.join(directory_path, 'evaluation_vegetation_binary.txt')
        confusion_matrix_file_path = os.path.join(directory_path, 'confusion_vegetation_binary.npy')

        # Write metrics to file
        with open(metrics_file_path, 'w') as file:
            file.write(f"IoU per class: {iou}\n")
            file.write(f"Pixel accuracy: {pixel_acc}\n")
            file.write(f"Precision per class: {precision}\n")
            file.write(f"Recall per class: {recall}\n")
            file.write(f"F1 Score per class: {f1_score}\n")
            file.write(f"Mean IoU per class: {mean_IoU}\n")
            file.write(f"Pixel accuracy: {mean_pixel_acc}\n")
            file.write(f"Mean Precision per class: {mean_precision}\n")
            file.write(f"Mean Recall per class: {mean_recall}\n")
            file.write(f"Mean F1 Score per class: {mean_f1_score}\n")
            file.write(f"Mean IoU Impervious surfaces: {mean_IoU_no_back}\n")
            file.write(f"Frequency weighted IoU: {freq_IoU}\n")
            file.write(f"Overall accuracy: {overall_accuracy}\n")
        # Save the confusion matrix
        np.save(confusion_matrix_file_path, hist)

        # Also log the information
        # Log or print the metrics as needed
        logger.info(f"Mean IoU: {mean_IoU}")
        logger.info(f"Mean Pixel accuracy: {mean_pixel_acc}")
        logger.info(f"Mean precision: {mean_precision}")
        logger.info(f"Mean recall: {mean_recall}")
        logger.info(f"Mean F1 score: {mean_f1_score}")
        logger.info(f"Overall accuracy: {overall_accuracy}")

        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'val_source': config.val_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    test_pre = TestPre()
    dataset = RGBXDataset(data_setting, 'test', test_pre)
 
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                 config.norm_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)