import os
import shutil
import argparse
import traceback
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import arcpy
from osgeo import gdal
from pyproj import CRS, Transformer

# Constants
SRC_CRS_WKT = """PROJCS["NAD_1983_2011_StatePlane_Mississippi_East_FIPS_2301_Ft_US",GEOGCS["GCS_NAD_1983_2011",
DATUM["NAD_1983_2011",SPHEROID["GRS_1980",6378137.0,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG",
"1116"]],PRIMEM["Greenwich",0.0,AUTHORITY["EPSG","8901"]],UNIT["Degree",0.0174532925199433,AUTHORITY["EPSG","9102"]],
AUTHORITY["EPSG","6318"]],PROJECTION["Transverse_Mercator",AUTHORITY["Esri","43006"]],PARAMETER["False_Easting",
984250.0,AUTHORITY["Esri","100001"]],PARAMETER["False_Northing",0.0,AUTHORITY["Esri","100002"]],PARAMETER[
"Central_Meridian",-88.83333333333333,AUTHORITY["Esri","100010"]],PARAMETER["Scale_Factor",0.99995,AUTHORITY["Esri",
"100003"]],PARAMETER["Latitude_Of_Origin",29.5,AUTHORITY["Esri","100021"]],UNIT["Foot_US",0.3048006096012192,
AUTHORITY["EPSG","9003"]],AUTHORITY["EPSG","6507"]]"""

class OrthomosaicTiler:
    def __init__(self, input_path, output_dir, tile_size, output_format, red_dot, csv_path=None, csv_epsg=None):
        self.input_path = input_path
        self.input_name = os.path.basename(input_path).split(".")[0]
        self.tile_size = tile_size
        self.output_format = output_format.lower()
        self.red_dot = red_dot
        self.csv_path = csv_path
        self.csv_epsg = csv_epsg
        self.points = []
        self.output_csv = []

        # Ensure output directory exists
        self.output_dir = f"{output_dir}/{self.input_name}/data"
        self.with_dir = os.path.join(self.output_dir, f"with")
        self.without_dir = os.path.join(self.output_dir, f"without")
        self.red_dot_dir = os.path.join(self.output_dir, f"red_dot")

        os.makedirs(self.with_dir, exist_ok=True)
        os.makedirs(self.without_dir, exist_ok=True)
        os.makedirs(self.red_dot_dir, exist_ok=True)

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

        df.columns = ['Type of debris', 'Date', 'Latitude', 'Longitude']

        valid_points = []
        for _, row in df.iterrows():
            try:
                # Convert coordinates to easting and northing
                lon, lat = float(row['Longitude']), float(row['Latitude'])
                easting, northing = transformer.transform(lon, lat)
                valid_points.append((easting, northing))
            except Exception as e:
                print(f"Warning: Could not convert coordinates for row {_}.\n{e}")

        self.points = valid_points
        df = df.iloc[:len(valid_points)]  # Ensure the DataFrame length matches the number of valid points
        df['Easting'] = [point[0] for point in self.points]
        df['Northing'] = [point[1] for point in self.points]

        output_file = os.path.basename(self.csv_path).replace(file_ext, "_projected.csv")
        output_path = f"{self.output_dir}/{output_file}"
        df.to_csv(output_path, index=False)

        print(f"Transformed {len(self.points)} points successfully.")

    def get_tile_transform(self, transform, x_offset, y_offset):
        return (
            transform[0] + x_offset * transform[1],  # Top-left x
            transform[1],  # W-E pixel resolution
            transform[2],  # Rotation, 0 if image is "north up"
            transform[3] + y_offset * transform[5],  # Top-left y
            transform[4],  # Rotation, 0 if image is "north up"
            transform[5]  # N-S pixel resolution
        )

    def tile(self, use_multiprocessing=True):
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
        args = []
        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                tile_width = min(self.tile_size, width - i)
                tile_height = min(self.tile_size, height - j)
                window = (i, j, tile_width, tile_height)
                tile_transform = self.get_tile_transform(transform, i, j)
                args.append((i, j, window, tile_transform, nodata, crs_wkt))

        with Pool(cpu_count() // 4) as pool:
            pool.starmap(self.process_tile, args)

    def tile_single_process(self, width, height, nodata, crs_wkt, transform):
        for i in range(0, width, self.tile_size):
            for j in range(0, height, self.tile_size):
                tile_width = min(self.tile_size, width - i)
                tile_height = min(self.tile_size, height - j)
                window = (i, j, tile_width, tile_height)
                tile_transform = self.get_tile_transform(transform, i, j)
                self.process_tile(i, j, window, tile_transform, nodata, crs_wkt)

    def process_tile(self, i, j, window, transform, nodata, crs_wkt):
        src_ds = gdal.Open(self.input_path)
        tile = src_ds.ReadAsArray(window[0], window[1], window[2], window[3])

        if self.is_invalid_tile(tile, nodata):
            return

        tile_name = f"{self.input_name}---{i}_{j}_{window[2]}_{window[3]}"
        tile_path = f"{self.with_dir}/{tile_name}"

        contains_point = self.tile_contains_point(transform, window)

        self.save_tile(tile, tile_path, transform, crs_wkt, contains_point)

        if contains_point and self.red_dot:
            points = self.get_tile_points(transform, window)
            red_dot_tile = self.superimpose_red_dots(tile, points)
            red_dot_tile_path = f"{self.red_dot_dir}/{tile_name}"
            self.save_tile(red_dot_tile, red_dot_tile_path, transform, crs_wkt, contains_point, is_red_dot=True)

    def is_invalid_tile(self, tile, nodata):
        if tile is None:
            return True

        total_pixels = tile.size
        nan_count = np.isnan(tile).sum()
        black_count = np.sum(tile == 0)
        white_count = np.sum(tile == 255)
        nodata_count = 0
        if nodata is not None:
            nodata_count = np.sum(tile == nodata)

        invalid_pixel_count = nan_count + black_count + white_count + nodata_count
        if invalid_pixel_count > total_pixels / 2:
            return True

        return False

    def tile_contains_point(self, transform, window):
        minx = transform[0]
        miny = transform[3]
        maxx = minx + window[2] * transform[1]
        maxy = miny + window[3] * transform[5]

        for x, y in self.points:
            if minx <= x < maxx and miny >= y > maxy:
                return True
        return False

    def get_tile_points(self, transform, window):
        minx = transform[0]
        miny = transform[3]
        maxx = minx + window[2] * transform[1]
        maxy = miny + window[3] * transform[5]

        points_in_tile = []
        for x, y in self.points:
            if minx <= x < maxx and miny >= y > maxy:
                # Convert coordinates to pixel values
                pixel_x = int((x - minx) / transform[1])
                pixel_y = int((y - miny) / transform[5])
                points_in_tile.append((pixel_x, pixel_y))

        return np.array(points_in_tile)

    def superimpose_red_dots(self, tile, points):
        tile = np.moveaxis(tile, 0, -1)
        tile = np.clip(tile, 0, 255).astype(np.uint8)

        if tile.shape[2] == 4:
            tile = tile[:, :, :3]

        image = Image.fromarray(tile)
        draw = ImageDraw.Draw(image)
        dot_size = 2

        for x, y in points:
            draw.ellipse((x - dot_size,
                          y - dot_size,
                          x + dot_size,
                          y + dot_size), fill='red')

        tile = np.moveaxis(np.array(image), -1, 0)

        return tile

    def save_sidecar_files(self, tile_path, transform, crs_wkt):
        with open(f"{tile_path}.jgw", 'w') as f:
            f.write(f"{transform[1]}\n")
            f.write("0\n")
            f.write("0\n")
            f.write(f"{transform[5]}\n")
            f.write(f"{transform[0]}\n")
            f.write(f"{transform[3]}\n")

        with open(f"{tile_path}.prj", 'w') as f:
            f.write(crs_wkt)

    def save_tile(self, tile, tile_path, transform, crs_wkt, contains_point, is_red_dot=False):
        try:
            if self.output_format == 'jpeg':
                self.save_as_jpeg(tile, tile_path, transform, crs_wkt)
                self.define_crs(f"{tile_path}.jpeg", SRC_CRS_WKT)
            else:
                self.save_as_geotiff(tile, tile_path, transform, crs_wkt)
                self.define_crs(f"{tile_path}.tif", SRC_CRS_WKT)

            if not contains_point and not is_red_dot:
                src_ds = gdal.Open(tile_path + f".{self.output_format}")
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
            print(f"Could not save / move tile {os.path.basename(tile_path)}\n{e}")

    def save_as_geotiff(self, tile, tile_path, transform, crs_wkt):
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(f"{tile_path}.tif", tile.shape[2], tile.shape[1], tile.shape[0], gdal.GDT_Float32)
        dst_ds.SetGeoTransform(transform)
        dst_ds.SetProjection(crs_wkt)

        for band in range(tile.shape[0]):
            dst_ds.GetRasterBand(band + 1).WriteArray(tile[band])
            dst_ds.GetRasterBand(band + 1).SetNoDataValue(transform[2])

        dst_ds.FlushCache()
        dst_ds = None

    def save_as_jpeg(self, tile, tile_path, transform, crs_wkt):
        tile = np.moveaxis(tile, 0, -1)
        tile = np.clip(tile, 0, 255).astype(np.uint8)

        if tile.shape[2] == 4:
            tile = tile[:, :, :3]

        image = Image.fromarray(tile)
        image.save(f"{tile_path}.jpeg", "JPEG", quality=95, optimize=True)

        self.save_sidecar_files(tile_path, transform, crs_wkt)

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

    parser.add_argument("--red_dot", action="store_true",
                        help="Superimpose a red dot on tiles containing points of interest")

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
                                 red_dot=args.red_dot,
                                 csv_path=args.csv_path,
                                 csv_epsg=args.csv_epsg)

        tiler.tile(use_multiprocessing=not args.single_process)

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        print(f"ERROR: Could not tile orthomosaic.\n{e}")

if __name__ == "__main__":
    main()