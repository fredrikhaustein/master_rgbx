import os
from osgeo import gdal,osr  
from PIL import Image
import numpy as np
def convert_png_to_geotiff(png_folder, tif_folder, output_folder):
    """
    Converts PNG images to GeoTIFF format using georeference data from corresponding TIFF files,
    ensuring all output files are set to the EPSG:4326 CRS.

    Parameters:
    - png_folder: Folder containing PNG files.
    - tif_folder: Folder containing corresponding TIFF files with georeference data.
    - output_folder: Folder where the output GeoTIFFs will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    # Spatial Reference System set to EPSG:4326
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(25832)

    # Loop through all PNG files in the PNG folder
    for png_filename in os.listdir(png_folder):
        if png_filename.endswith('.png'):
            base_name = os.path.splitext(png_filename)[0]
            tif_filename = base_name + '.tif'
            png_path = os.path.join(png_folder, png_filename)
            tif_path = os.path.join(tif_folder, tif_filename)
            output_path = os.path.join(output_folder, tif_filename)

            # Open the PNG file
            png_image = Image.open(png_path)
            png_array = np.array(png_image)

            # Open the corresponding TIFF file to get geospatial data
            tif_ds = gdal.Open(tif_path)
            if not tif_ds:
                print(f"Failed to open TIFF file: {tif_path}")
                continue

            # Create a new GeoTIFF file with the same geospatial info
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(output_path, png_image.width, png_image.height, 1, gdal.GDT_Byte)
            print(tif_ds.GetGeoTransform())
            print(srs.ExportToWkt())
            out_ds.SetGeoTransform(tif_ds.GetGeoTransform())  # Copy GeoTransform from source
            out_ds.SetProjection(srs.ExportToWkt())  # Set the projection to EPSG:4326

            # Write PNG data to the new GeoTIFF
            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(png_array)
            out_band.SetNoDataValue(0)  # Assuming '0' is the no-data value

            # Close datasets
            out_band = None
            out_ds = None
            tif_ds = None

            print(f"Converted {png_path} to GeoTIFF and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    png_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/results/results_image/result_augmented_building_color'
    tif_folder = '/cluster/home/fredhaus/imperviousSurfaces/DatasetMaster/Tile_aerial_image'
    output_folder = '/cluster/home/fredhaus/imperviousSurfaces/rgbx_seg/RGBX_Semantic_Segmentation/result_tif/direct_method'
    convert_png_to_geotiff(png_folder, tif_folder, output_folder)

# This script will convert PNG images to GeoTIFF format using georeference data from corresponding TIFF files.
# The PNG images are assumed to have the same dimensions as the TIFF images.
# The output GeoTIFF files will be saved in the specified output folder.
# The script uses the GDAL library to read and write geospatial data.