from PIL import Image
import numpy as np

def convert_binary_png_and_save(input_path, output_path):
    """
    Read a PNG file that contains binary values (0 and 1), convert 1s to 255 while keeping 0s,
    and save the modified image back as a PNG file.

    Args:
    input_path (str): Path to the input PNG file.
    output_path (str): Path where the modified PNG file will be saved.

    Returns:
    None
    """
    # Load the image
    img = Image.open(input_path)
    img = img.convert('L')  # Ensure it's in grayscale mode

    # Convert the image to a numpy array
    data = np.array(img)

    # Convert 1s to 255
    data[data == 1] = 255

    # Create a new Image from the modified array
    new_img = Image.fromarray(data.astype(np.uint8))

    # Save the new image
    new_img.save(output_path)

    print(f'Modified image saved successfully to {output_path}')

# Example usage
input_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_20_fkb_vegitation_ndvi_01_indirect/LabelFolder/07_35.png'  # Path to the input PNG file
output_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/07_35_ndvi_01.png'  # Path where the modified PNG file will be saved

folder1 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_manually_labling_ground_truth_png/29_21.png"
folder1Output = "/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/05_12_manually.png"
folder2 = "/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/fkb_ground_truth_png/29_21.png"
folder2Output = "/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/05_12_fkb.png"


# convert_binary_png_and_save(input_path, output_path)  # Convert and save the image
convert_binary_png_and_save(input_path, output_path)  # Convert and save the image
# convert_binary_png_and_save(folder2, folder2Output)  # Convert and save the image


