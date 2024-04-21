# encoding: utf-8

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def hist_info(n_cl, pred, gt):
    # Check if the prediction and ground truth arrays have the same shape
    if pred.shape != gt.shape:
        print("Error: The shapes of prediction and ground truth do not match.")
        zero_confusion_matrix = np.zeros((n_cl, n_cl), dtype=int)
        return zero_confusion_matrix, 0, 0

    # Continue with the function if the shapes match
    k = (gt >= 0) & (gt < n_cl)
    labeled = np.sum(k)
    correct = np.sum(pred[k] == gt[k])
    confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int),
                                  minlength=n_cl ** 2).reshape(n_cl, n_cl)
    return confusionMatrix, labeled, correct

def compute_score(hist, correct, labeled):
    # Existing IoU calculations remain unchanged
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])  # Adjust according to need
    freq = hist.sum(1) / hist.sum()
    freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()
    classAcc = np.diag(hist) / hist.sum(axis=1)
    mean_pixel_acc = np.nanmean(classAcc)
    pixel_acc = correct / labeled

    # Compute Precision, Recall, F1 for each class
    precision = np.diag(hist) / (hist.sum(axis=0) + np.finfo(np.float32).eps)
    recall = np.diag(hist) / (hist.sum(axis=1) + np.finfo(np.float32).eps)
    f1_score = 2 * (precision * recall) / (precision + recall + np.finfo(np.float32).eps)

    # Compute mean Precision, Recall, F1
    mean_precision = np.nanmean(precision)
    mean_recall = np.nanmean(recall)
    mean_f1_score = np.nanmean(f1_score)

    overall_accuracy = np.diag(hist).sum() / hist.sum()

    return iou, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc, precision, recall, f1_score, mean_precision, mean_recall, mean_f1_score,overall_accuracy
