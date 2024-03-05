import numpy as np

# Function to calculate IoU
def calculate_iou(preds, labels, n_classes):
    iou_per_class = []
    for cls in range(n_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflow
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou_per_class.append(float('nan'))  # Avoid division by zero
        else:
            iou_per_class.append(float(intersection) / max(union, 1))
    return np.nanmean(iou_per_class)  # Return the average IoU over all classes

# Function to calculate IoU per class
def calculate_iou_per_class(preds, labels, n_classes):
    iou_per_class = []
    for cls in range(n_classes):
        pred_inds = preds == cls
        print(pred_inds)
        target_inds = labels == cls
        print(target_inds)
        intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflow
        print(intersection)
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        print(union)
        if union == 0:
            iou_per_class.append(float('nan'))  # Avoid division by zero
        else:
            iou_per_class.append(float(intersection) / max(union, 1))
    return iou_per_class  # Return IoU for each class
