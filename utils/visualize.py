import numpy as np
import cv2
import scipy.io as sio

def set_img_color(colors, background, img, pred, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt==background)] = 255
    return img

def show_prediction(colors, background, img, pred, gt):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final

def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    #set_img_color(colors, background, im1, clean, gt)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final

def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1,3)) * 255).tolist()[0])

    return colors

def get_ade_colors():
    colors = sio.loadmat('./color150.mat')['colors']
    colors = colors[:,::-1,]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0,[0,0,0])

    return colors

def get_class_colors():
    """
    Returns a list of RGB colors for the classes in the dataset.

    Modify the colors as needed for your specific dataset.
    """
    return [
        (255, 0, 0),    # Class 0 color (e.g., Red)
        (0, 255, 0),    # Class 1 color (e.g., Green)
        (0, 0, 255),    # Class 2 color (e.g., Blue)
        # Add more colors for additional classes
    ]

def get_class_colors_six():
    """
    Returns a list of RGB colors for six classes in the dataset.

    Modify the colors as needed for your specific dataset.

    "background", "road", "driveway", "parkingspot", "sport", "playground"
    """
    return [
        (255, 0, 0),    # Class 0 color (e.g., Red) background
        (0, 255, 0),    # Class 1 color (e.g., Green) road
        (0, 0, 255),    # Class 2 color (e.g., Blue) driveway
        (255, 255, 0),  # Class 3 color (e.g., Yellow) parkingspot
        (255, 0, 255),  # Class 4 color (e.g., Magenta) sport
        (0, 255, 255),  # Class 5 color (e.g., Cyan) playground
        # Add more colors if you have more than six classes
    ]


def print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    if show_no_back:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'mean_IU_no_back', mean_IoU_no_back*100,
                                                                                                                'freq_IoU', freq_IoU*100, 'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    else:
        lines.append('----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'freq_IoU', freq_IoU*100, 
                                                                                                    'mean_pixel_acc', mean_pixel_acc*100, 'pixel_acc',pixel_acc*100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line

def print_metrics(iou, recall, f1_scores, precision, overall_accuracy,freq_IoU, overestimation, class_names=None, no_print=False):
    n = len(iou)  # Assuming iou, recall, f1_scores, and precision have the same length
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i+1, class_names[i])
        lines.append('%-8s\tIoU: %.3f%%\tRecall: %.3f%%\tF1 Score: %.3f%%\tPrecision: %.3f%%' % (cls, iou[i] * 100, recall[i] * 100, f1_scores[i] * 100, precision[i] * 100))

    # Compute mean values for each metric
    mean_iou = np.nanmean(iou)
    mean_recall = np.nanmean(recall)
    mean_f1 = np.nanmean(f1_scores)
    mean_precision = np.nanmean(precision)

    # Append mean values
    lines.append('----------')
    lines.append('Mean Values\tIoU: %.3f%%\tRecall: %.3f%%\tF1 Score: %.3f%%\tPrecision: %.3f%%' % (mean_iou * 100, mean_recall * 100, mean_f1 * 100, mean_precision * 100))
    lines.append(f"Overall accuracy: {overall_accuracy}")
    lines.append(f"Frequency weighted IoU: {freq_IoU}")
    lines.append(f"Overestimation: {overestimation}")
    # Join all lines into a single string
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line



