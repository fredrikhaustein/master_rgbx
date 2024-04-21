import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def display_and_save_random_images(aerial_img_folder, mask_folder, additional_img_folder, output_folder, alpha=0.4, num_images=4):
    """
    Display and save random sets of images: an aerial image, its segmentation mask, the aerial image with the mask overlay,
    and an additional image (e.g., NDVI) side by side. Each set is chosen randomly from the mask folder.

    Parameters:
    - aerial_img_folder: Path to the folder containing aerial images.
    - mask_folder: Path to the folder containing segmentation masks.
    - additional_img_folder: Path to the folder containing additional images (e.g., NDVI images).
    - output_folder: Path to the folder where output images will be saved.
    - alpha: Transparency of the overlay mask. Default is 0.4.
    - num_images: Number of random images to display. Default is 4.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    available_images = [os.path.splitext(f)[0] for f in os.listdir(mask_folder) if f.endswith('.png')]
    selected_images = random.sample(available_images, min(num_images, len(available_images)))

    for idx, image_name in enumerate(selected_images):
        aerial_img_path = os.path.join(aerial_img_folder, image_name + '.png')
        mask_path = os.path.join(mask_folder, image_name + '.png')
        additional_img_path = os.path.join(additional_img_folder, image_name + '.png')

        aerial_img = cv2.imread(aerial_img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        additional_img = cv2.imread(additional_img_path)

        if len(mask.shape) == 2 or mask.shape[2] == 1:
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        else:
            mask_color = mask

        overlay = cv2.addWeighted(aerial_img, 1, mask_color, alpha, 0)

        plt.figure(figsize=(20, 10))
        titles = ['Aerial Image', 'Segmentation Mask', 'Image with Mask Overlay', 'Additional Image']
        images = [cv2.cvtColor(aerial_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), cv2.cvtColor(additional_img, cv2.COLOR_BGR2RGB)]

        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        
        # Save the figure
        plt.savefig(os.path.join(output_folder, f"{image_name}.png"))
        plt.close()  # Close the figure to free memory


# Example usage
aerial_img_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_2/RGBFolder'
mask_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/results_multiclass_ignore_0_color'
additional_img_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_2/ModalXFolder'
display_and_save_random_images(aerial_img_folder, mask_folder, additional_img_folder, "/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_plot")
