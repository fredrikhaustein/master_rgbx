import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

def tiff_to_rgb_and_save(tiff_path, save_path, colormap=plt.cm.RdYlGn):
    """
    Read a TIFF file, convert its data (expected to range from -1 to 1) to an RGB image using a colormap,
    and save the RGB image to a specified path as a PNG file.

    Args:
    tiff_path (str): Path to the TIFF file.
    save_path (str): Path where the RGB image will be saved, including the filename.
    colormap (matplotlib.colors.Colormap): Colormap to use for the conversion.

    Returns:
    None
    """
    # Open the TIFF file
    with rasterio.open(tiff_path) as src:
        atif = src.read(1)  # Read the first band

    # Normalize the input data to the range [0, 1]
    norm = Normalize(vmin=-1, vmax=1)
    normalized_atif = norm(atif)

    # Apply the colormap
    mapped_color = colormap(normalized_atif)

    # Convert to RGB, discarding the alpha channel if it exists
    rgb_image = (mapped_color[:, :, :3] * 255).astype(np.uint8)

    # Save to PNG using PIL
    img = Image.fromarray(rgb_image)
    img.save(save_path)

    print(f'Image saved successfully to {save_path}')

# Example usage:
# Create a test array with values from -1 to 1
test_atif = "/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/datasets/Dataset_15_building_binary/ModalXFolder/07_35.tif"

# Path where the image will be saved
save_path = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/result_master_images/atif_image.png'

# Convert to RGB and save the image
tiff_to_rgb_and_save(test_atif, save_path)

print(f'Image saved successfully to {save_path}')