import os
import argparse
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image

from pyproj import Transformer

import rasterio
from rasterio.windows import Window
from rasterio.crs import CRS


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class OrthomosaicTiler:
    def __init__(self, input_path, output_dir, tile_size, output_format, csv_path=None, csv_epsg=None):
        self.input_path = input_path
        self.input_name = os.path.basename(input_path).split(".")[0]
        self.tile_size = tile_size
        self.output_format = output_format.lower()
        self.csv_path = csv_path
        self.csv_epsg = csv_epsg
        self.points = []

        # Ensure output directory exists
        self.output_dir = os.path.join(output_dir, self.input_name)
        os.makedirs(self.output_dir, exist_ok=True)

        if csv_path:
            self.read_csv()

    def read_csv(self):
        with rasterio.open(self.input_path) as src:
            dst_crs = src.crs
            src_crs = CRS.from_epsg(self.csv_epsg) if self.csv_epsg else CRS.from_epsg(4326)
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                lon, lat = row['Long'], row['Lat']
                x, y = transformer.transform(lon, lat)
                self.points.append((x, y))

    def tile(self):
        """
        Tile the orthomosaic GeoTIFF file into smaller tiles.
        """
        with rasterio.open(self.input_path) as src:
            width = src.width
            height = src.height
            nodata = src.nodata
            crs_wkt = src.crs.to_wkt()

            # Prepare arguments for multiprocessing
            args = []
            for i in range(0, width, self.tile_size):
                for j in range(0, height, self.tile_size):
                    window = Window(i, j, self.tile_size, self.tile_size)
                    transform = src.window_transform(window)
                    args.append((i, j, window, transform, nodata, crs_wkt))

            # Use multiprocessing to process tiles in parallel
            with Pool(cpu_count() // 2) as pool:
                pool.starmap(self.process_tile, args)

    def process_tile(self, i, j, window, transform, nodata, crs_wkt):
        """
        Process a single tile.
        """
        with rasterio.open(self.input_path) as src:
            tile = src.read(window=window)

            # Check if the tile contains valid data and a point (if CSV provided)
            if self.is_invalid_tile(tile, nodata):
                return

            if self.csv_path and not self.tile_contains_point(transform, window):
                return

            # Create a tile name
            tile_name = f"{self.input_name}_{i}_{j}_{self.tile_size}"
            tile_path = f"{self.output_dir}/{tile_name}"

            # Save the tile
            self.save_tile(tile, tile_path, src.profile, transform, crs_wkt)

    def is_invalid_tile(self, tile, nodata):
        """
        Check if the tile is invalid (contains more than half NaN, black, white, or nodata values).
        """
        total_pixels = tile.size

        # Count NaN values
        nan_count = np.isnan(tile).sum()

        # Count black pixels (0)
        black_count = np.sum(tile == 0)

        # Count white pixels (255)
        white_count = np.sum(tile == 255)

        # Count nodata pixels
        nodata_count = 0
        if nodata is not None:
            nodata_count = np.sum(tile == nodata)

        # Check if more than half of the pixels are NaN, black, white, or nodata
        invalid_pixel_count = nan_count + black_count + white_count + nodata_count
        if invalid_pixel_count > total_pixels / 2:
            return True

        return False

    def tile_contains_point(self, transform, window):
        """
        Check if the tile contains at least one point from the CSV file.
        """
        minx, miny = transform * (0, 0)
        maxx, maxy = transform * (window.width, window.height)

        for x, y in self.points:
            if minx <= x < maxx and miny <= y < maxy:
                return True
        return False

    def save_tile(self, tile, tile_path, profile, transform, crs_wkt):
        """
        Save the tile to a file.
        """
        if self.output_format == 'jpeg':
            self.save_as_jpeg(tile, tile_path, transform, crs_wkt)
        else:
            self.save_as_geotiff(tile, tile_path, profile, transform)

    def save_as_geotiff(self, tile, tile_path, profile, transform):
        """
        Save the tile as a GeoTIFF file.
        """
        profile.update({
            'height': tile.shape[1],
            'width': tile.shape[2],
            'transform': transform
        })

        with rasterio.open(f"{tile_path}.tif", 'w', **profile) as dst:
            dst.write(tile)

    def save_as_jpeg(self, tile, tile_path, transform, crs_wkt):
        """
        Save the tile as a JPEG file with sidecar files for georeferencing.
        """
        # Convert the tile to uint8 format for JPEG
        tile = np.moveaxis(tile, 0, -1)  # Move channels to last dimension
        tile = np.clip(tile, 0, 255).astype(np.uint8)

        # If the tile has an alpha channel, remove it
        if tile.shape[2] == 4:
            tile = tile[:, :, :3]

        # Save the JPEG file
        image = Image.fromarray(tile)
        image.save(f"{tile_path}.jpg", "JPEG")

        # Save the sidecar files for georeferencing
        self.save_sidecar_files(tile_path, transform, crs_wkt)

    def save_sidecar_files(self, tile_path, transform, crs_wkt):
        """
        Save the sidecar files for georeferencing.
        """
        # Save the .jgw file
        with open(f"{tile_path}.jgw", 'w') as f:
            f.write(f"{transform.a}\n")  # Pixel size in the x-direction
            f.write(f"{transform.b}\n")  # Rotation term (typically 0)
            f.write(f"{transform.d}\n")  # Rotation term (typically 0)
            f.write(f"{transform.e}\n")  # Pixel size in the y-direction (negative)
            f.write(f"{transform.c}\n")  # X-coordinate of the center of the upper left pixel
            f.write(f"{transform.f}\n")  # Y-coordinate of the center of the upper left pixel

        # Save the .prj file
        with open(f"{tile_path}.prj", 'w') as f:
            f.write(crs_wkt)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description='Tile a large orthomosaic GeoTIFF file.')

    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input orthomosaic GeoTIFF file')

    parser.add_argument('--output_dir', type=str,
                        default=f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/Data",
                        help='Directory to save the output tiles')

    parser.add_argument('--tile_size', type=int, default=2048,
                        help='Size of each tile in pixels (default: 2048)')

    parser.add_argument('--output_format', type=str, choices=['geotiff', 'jpeg'], default='geotiff',
                        help='Output format of the tiles (default: geotiff)')

    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to the CSV file containing points of interest')

    parser.add_argument('--csv_epsg', type=int, default=4326,
                        help='EPSG code for the coordinate system of the CSV file (default: 4326 for WGS84)')

    args = parser.parse_args()

    try:
        tiler = OrthomosaicTiler(input_path=args.input_path,
                                 output_dir=args.output_dir,
                                 tile_size=args.tile_size,
                                 output_format=args.output_format,
                                 csv_path=args.csv_path,
                                 csv_epsg=args.csv_epsg)
        tiler.tile()

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        print(f"ERROR: Could not tile orthomosaic.\n{e}")


if __name__ == "__main__":
    main()