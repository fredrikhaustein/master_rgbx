import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# Assuming apply_crf function is defined somewhere
# def apply_crf(original_image, pred_probs):
#     ... return refined_pred

def convert_predictions_to_image(pred, class_colors):
    """Converts prediction to an RGB image using class colors."""
    pred_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(class_colors):
        pred_image[pred == class_id] = color
    return pred_image

def visualize_and_save_predictions(original_image_path, pred, refined_pred, save_path, class_colors):
    """Visualizes original and refined predictions, and saves them."""
    # Ensure original_image is in RGB format if it's not already
    if original_image.shape[2] == 3:
        pass  # Assuming original_image is already an RGB image
    else:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Convert predictions to RGB images
    pred_image = convert_predictions_to_image(pred, class_colors)
    refined_pred_image = convert_predictions_to_image(refined_pred, class_colors)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_image)
    axes[1].set_title('Before CRF')
    axes[1].axis('off')
    
    axes[2].imshow(refined_pred_image)
    axes[2].set_title('After CRF')
    axes[2].axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    # Save the images
    save_original_path = os.path.join(save_path, "original_image.png")
    save_pred_path = os.path.join(save_path, "prediction_before_crf.png")
    save_refined_path = os.path.join(save_path, "prediction_after_crf.png")
    
    Image.fromarray(original_image).save(save_original_path)
    Image.fromarray(pred_image).save(save_pred_path)
    Image.fromarray(refined_pred_image).save(save_refined_path)
    print(f"Images saved to {save_path}")