import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import torch.nn.functional as F

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def convert_predictions_to_probs(pred):
    """Convert model's logits to probabilities using softmax for NumPy arrays."""
    pred_probs = softmax(pred)
    return pred_probs

def apply_crf(image, predictions, num_classes=2):
    # The current shape is (height, width, num_classes), 
    # but we need to reshape predictions to (num_classes, height * width)
    
    # Transpose the predictions from (height, width, num_classes) to (num_classes, height, width)
    predictions = np.transpose(predictions, (2, 0, 1))
    # print(predictions)
    
    # Now we can reshape to (num_classes, height * width) correctly
    unary = unary_from_softmax(predictions.reshape((num_classes, -1)))
    unary = np.ascontiguousarray(unary)
    height, width = predictions.shape[1], predictions.shape[2]
    
    # Initialize DenseCRF
    d = dcrf.DenseCRF2D(width, height, num_classes)
    d.setUnaryEnergy(unary)
    
    # Optional: Add pairwise potentials. These typically need to be tuned to your specific problem
    d.addPairwiseGaussian(sxy=(2, 2), compat=1)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=image, compat=10)
    
    # Perform CRF inference
    Q = d.inference(25)
    
    # The CRF inference returns the probabilities in the same order as the unary input
    # Reshape output back to (num_classes, height, width)
    Q = np.array(Q).reshape((num_classes, height, width))
    
    # Take the argmax over the class axis to get the final class labels per pixel
    map = np.argmax(Q, axis=0)
    
    return map