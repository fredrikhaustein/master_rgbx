import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_and_save_comparison(ground_truth_folder, direct_folder, indirect_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the files in the direct method folder
    for file_name in os.listdir(direct_folder):
        if file_name.endswith('.png'):  # Adjust the extension if necessary
            # Extract the relevant part of the file name (e.g., "01_01" from "white_overlay_01_01.png")
            base_name = file_name.split('_')[-2] + '_' + file_name.split('_')[-1]

            # Construct the full path for each file
            ground_truth_path = os.path.join(ground_truth_folder, base_name)
            direct_path = os.path.join(direct_folder, file_name)
            indirect_path = os.path.join(indirect_folder, file_name)

            # Load the images
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            pred_direct = cv2.imread(direct_path, cv2.IMREAD_GRAYSCALE)
            pred_indirect = cv2.imread(indirect_path, cv2.IMREAD_GRAYSCALE)

            # Convert ground truth pixel values from 1 to 255
            ground_truth = np.where(ground_truth == 1, 255, ground_truth)

            # Create a comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(ground_truth, cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[1].imshow(pred_direct, cmap='gray')
            axes[1].set_title('Direct Method')
            axes[2].imshow(pred_indirect, cmap='gray')
            axes[2].set_title('Indirect Method')

            for ax in axes:
                ax.axis('off')

            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(output_folder, base_name)
            plt.savefig(output_path)
            plt.close(fig)
# Example usage
# ground_truth_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_17_fkb_impervious_surfaces/LabelFolder'
# direct_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_master_image_new_dataset/Dataset_17_with_model_from_dataset_18_manuall_fkb'
# indirect_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_master_image_new_dataset/Dataset_17_with_model_indirect_ndvi_01'
# output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_difference_ground_truth_pred/difference_ground_truth_pred_18_direct_vs_indirect_ndvi_01'

ground_truth_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_22_trondheim_test/LabelFolder'
direct_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results_new_dataset/dataset_22_trondheim_test_direct'
indirect_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results_new_dataset/dataset_22_trondheim_test_indirect'
output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_difference_ground_truth_pred/difference_ground_truth_test_trondheim_area'

plot_and_save_comparison(ground_truth_folder, direct_folder, indirect_folder, output_folder)
