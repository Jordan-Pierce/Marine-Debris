import os
import shutil
import argparse
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image

import arcpy
from osgeo import gdal
from pyproj import CRS, Transformer

# ----------------------------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------------------------
SRC_CRS_WKT = """PROJCS["NAD_1983_2011_StatePlane_Mississippi_East_FIPS_2301_Ft_US",GEOGCS["GCS_NAD_1983_2011",
DATUM["NAD_1983_2011",SPHEROID["GRS_1980",6378137.0,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG",
"1116"]],PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],UNIT["Degree",0.0174532925199433,AUTHORITY["EPSG","9102"]],
AUTHORITY["EPSG","6318"]],PROJECTION["Transverse_Mercator",AUTHORITY["Esri","43006"]],PARAMETER["False_Easting",
984250.0,AUTHORITY["Esri","100001"]],PARAMETER["False_Northing",0.0,AUTHORITY["Esri","100002"]],PARAMETER[
"Central_Meridian",-88.83333333333333,AUTHORITY["Esri","100010"]],PARAMETER["Scale_Factor",0.99995,AUTHORITY["Esri",
"100003"]],PARAMETER["Latitude_Of_Origin",29.5,AUTHORITY["Esri","100021"]],UNIT["Foot_US",0.3048006096012192,
AUTHORITY["EPSG","9003"]],AUTHORITY["EPSG","6507"]]"""

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
        self.output_csv = []

        # Ensure output directory exists
        self.output_dir = f"{output_dir}/{self.input_name}"
        self.with_dir = os.path.join(output_dir, f"{self.input_name}/with")
        self.without_dir = os.path.join(output_dir, f"{self.input_name}/without")
        os.makedirs(self.with_dir, exist_ok=True)
        os.makedirs(self.without_dir, exist_ok=True)

        self.read_csv()
        self.define_crs(self.input_path, SRC_CRS_WKT)

    def define_crs(self, tile_path, crs):
        spatial_ref = arcpy.SpatialReference()
        spatial_ref.loadFromString(crs)
        arcpy.DefineProjection_management(tile_path, spatial_ref)

    def read_csv(self):
        if not os.path.exists(self.csv_path):
            print(f"Warning: CSV file not found at {self.csv_path}. Skipping transformation.")
            return

        # Create CRS objects directly from WKT and EPSG code
        dst_crs = CRS.from_wkt(SRC_CRS_WKT)
        src_crs = CRS.from_epsg(4326)  # WGS84

        # Create a transformer object
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        # Determine file extension and read accordingly
        file_ext = os.path.splitext(self.csv_path)[1].lower()
        if file_ext == '.csv':
            df = pd.read_csv(self.csv_path, header=0)
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(self.csv_path, header=0)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        valid_points = []
        for _, row in df.iterrows():
            try:
                lon, lat = row['Lon'], row['Lat']
                x, y = transformer.transform(float(lon), float(lat))
                valid_points.append((x, y))
            except Exception as e:
                print(f"Warning: Could not convert coordinates for row {_}.\n{e}")

        self.points = valid_points
        df = df.iloc[:len(valid_points)]  # Ensure the DataFrame length matches the number of valid points
        df['X'] = [point[0] for point in self.points]
        df['Y'] = [point[1] for point in self.points]

        output_file = os.path.basename(self.csv_path).replace(file_ext, "_projected.csv")
        output_path = f"{self.output_dir}/{output_file}"
        df.to_csv(output_path, index=False)

        print(f"Transformed {len(self.points)} points successfully.")

    def get_tile_transform(self, transform, x_offset, y_offset):
        """
        Calculate the geotransform for the tile.
        """
        return (
            transform[0] + x_offset * transform[1],  # Top-left x
            transform[1],  # W-E pixel resolution
            transform[2],  # Rotation, 0 if image is "north up"
            transform[3] + y_offset * transform[5],  # Top-left y
            transform[4],  # Rotation, 0 if image is "north up"
            transform[5]  # N-S pixel resolution
        )

    def tile(self, use_multiprocessing=True):
        """
        Tile the orthomosaic GeoTIFF file into smaller tiles.
        """
        src_ds = gdal.Open(self.input_path)
        width = src_ds.RasterXSize
        height = src_ds.RasterYSize
        nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        crs_wkt = src_ds.GetProjectionRef()
        transform = src_ds.GetGeoTransform()

        if use_multiprocessing:
            self.tile_multiprocessing(width, height, nodata, crs_wkt, transform)
        else:
            self.tile_single_process(width, height, nodata, crs_wkt, transform)

    def tile_multiprocessing(self, width, height, nodata, crs_wkt, transform):
        """
        Tile the orthomosaic using multiprocessing.
        """
        # Prepare arguments for multiprocessing
        args = []
        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                tile_width = min(self.tile_size, width - i)
                tile_height = min(self.tile_size, height - j)
                window = (i, j, tile_width, tile_height)
                tile_transform = self.get_tile_transform(transform, i, j)
                args.append((i, j, window, tile_transform, nodata, crs_wkt))

        # Use multiprocessing to process tiles in parallel
        with Pool(cpu_count() // 2) as pool:
            pool.starmap(self.process_tile, args)

    def tile_single_process(self, width, height, nodata, crs_wkt, transform):
        """
        Tile the orthomosaic using a single process.
        """
        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                tile_width = min(self.tile_size, width - i)
                tile_height = min(self.tile_size, height - j)
                window = (i, j, tile_width, tile_height)
                tile_transform = self.get_tile_transform(transform, i, j)
                self.process_tile(i, j, window, tile_transform, nodata, crs_wkt)

    def process_tile(self, i, j, window, transform, nodata, crs_wkt):
        """
        Process a single tile.
        """
        src_ds = gdal.Open(self.input_path)
        tile = src_ds.ReadAsArray(window[0], window[1], window[2], window[3])

        # Check if the tile contains valid data
        if self.is_invalid_tile(tile, nodata):
            return

        # Create a tile name
        tile_name = f"{self.input_name}---{i}_{j}_{window[2]}_{window[3]}"
        tile_path = f"{self.with_dir}/{tile_name}"

        # Save the tile
        self.save_tile(tile, tile_path, transform, crs_wkt)

    def is_invalid_tile(self, tile, nodata):
        """
        Check if the tile is invalid (contains more than half NaN, black, white, or nodata values).
        """

        if tile is None:
            return True

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

    def save_sidecar_files(self, tile_path, transform, crs_wkt):
        """
        Save the sidecar files for georeferencing.
        """
        # Save the .jgw file
        with open(f"{tile_path}.jgw", 'w') as f:
            f.write(f"{transform[1]}\n")  # Pixel size in the x-direction
            f.write("0\n")  # Rotation term (typically 0)
            f.write("0\n")  # Rotation term (typically 0)
            f.write(f"{transform[5]}\n")  # Pixel size in the y-direction (negative)
            f.write(f"{transform[0]}\n")  # X-coordinate of the center of the upper left pixel
            f.write(f"{transform[3]}\n")  # Y-coordinate of the center of the upper left pixel

        # Save the .prj file
        with open(f"{tile_path}.prj", 'w') as f:
            f.write(crs_wkt)

    def save_tile(self, tile, tile_path, transform, crs_wkt):
        """
        Save the tile to a file.
        """
        try:
            if self.output_format == 'jpeg':
                self.save_as_jpeg(tile, tile_path, transform, crs_wkt)
                self.define_crs(f"{tile_path}.jpeg", SRC_CRS_WKT)
            else:
                self.save_as_geotiff(tile, tile_path, transform, crs_wkt)
                self.define_crs(f"{tile_path}.tif", SRC_CRS_WKT)

            # If tile doesn't contain any points, move it and its sidecar files to the "without" folder
            src_ds = gdal.Open(tile_path + f".{self.output_format}")
            transform = src_ds.GetGeoTransform()
            raster_x, raster_y = src_ds.RasterXSize, src_ds.RasterYSize
            contains_point = self.tile_contains_point(transform, (0, 0, raster_x, raster_y))
            if not contains_point:

                src_ds.FlushCache()
                src_ds = None

                if self.output_format == 'jpeg':
                    shutil.move(tile_path + ".jgw", self.without_dir)
                    shutil.move(tile_path + ".jpeg", self.without_dir)
                    shutil.move(tile_path + ".jpeg.aux.xml", self.without_dir)
                    shutil.move(tile_path + ".jpeg.xml", self.without_dir)
                    shutil.move(tile_path + ".prj", self.without_dir)
                else:
                    shutil.move(tile_path + ".tif", self.without_dir)
                    shutil.move(tile_path + ".tif.aux.xml", self.without_dir)

        except Exception as e:
            print(f"Could not save / move tile {os.path.basename(tile_path)}.\n{e}")

    def save_as_geotiff(self, tile, tile_path, transform, crs_wkt):
        """
        Save the tile as a lossless GeoTIFF file.
        """
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(f"{tile_path}.tif", tile.shape[2], tile.shape[1], tile.shape[0], gdal.GDT_Float32)
        dst_ds.SetGeoTransform(transform)
        dst_ds.SetProjection(crs_wkt)

        for band in range(tile.shape[0]):
            dst_ds.GetRasterBand(band + 1).WriteArray(tile[band])
            dst_ds.GetRasterBand(band + 1).SetNoDataValue(transform[2])

        dst_ds.FlushCache()
        dst_ds = None

    def tile_contains_point(self, transform, window):
        """
        Check if the tile contains at least one point from the CSV file.
        """
        minx = transform[0]
        miny = transform[3]
        maxx = minx + window[2] * transform[1]
        maxy = miny + window[3] * transform[5]

        for x, y in self.points:
            if minx <= x < maxx and miny >= y > maxy:
                return True
        return False

    def save_as_jpeg(self, tile, tile_path, transform, crs_wkt):
        """
        Save the tile as a high-quality JPEG file with sidecar files for georeferencing.
        """
        # Convert the tile to uint8 format for JPEG
        tile = np.moveaxis(tile, 0, -1)  # Move channels to last dimension
        tile = np.clip(tile, 0, 255).astype(np.uint8)

        # If the tile has an alpha channel, remove it
        if tile.shape[2] == 4:
            tile = tile[:, :, :3]

        # Save the JPEG file with highest quality
        image = Image.fromarray(tile)
        image.save(f"{tile_path}.jpeg", "JPEG", quality=95, optimize=True)

        # Save the sidecar files for georeferencing
        self.save_sidecar_files(tile_path, transform, crs_wkt)


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

    parser.add_argument('--tile_size', type=int, default=1024,
                        help='Size of each tile in pixels (default: 2048)')

    parser.add_argument('--output_format', type=str, choices=['geotiff', 'jpeg'], default='jpeg',
                        help='Output format of the tiles (default: geotiff)')

    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to the CSV file containing points of interest')

    parser.add_argument('--csv_epsg', type=int, default=4326,
                        help='EPSG code for the coordinate system of the CSV file (default: 4326 for WGS84)')

    parser.add_argument('--single_process', action='store_true',
                        help='Use single-process method instead of multiprocessing')

    args = parser.parse_args()

    try:
        tiler = OrthomosaicTiler(input_path=args.input_path,
                                 output_dir=args.output_dir,
                                 tile_size=args.tile_size,
                                 output_format=args.output_format,
                                 csv_path=args.csv_path,
                                 csv_epsg=args.csv_epsg)

        tiler.tile(use_multiprocessing=not args.single_process)

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        print(f"ERROR: Could not tile orthomosaic.\n{e}")


if __name__ == "__main__":
    main()