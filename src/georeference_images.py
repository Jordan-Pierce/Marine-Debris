import os
import re
import argparse
from exif import Image as ExifImage

import numpy as np
import utm

import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from PIL import Image


# --------------------------------------------------------------------------------------------------------------
# 
# --------------------------------------------------------------------------------------------------------------


def parse_image(image_path: str) -> dict:
    """
    Extract metadata from an image using the exif library.

    Args:
        image_path: The full path to the image file.

    Returns:
        A dictionary containing the relevant parsed EXIF data.
    """
    parsed_data = {}
    
    try:
        with open(image_path, 'rb') as image_file:
            img = ExifImage(image_file)
            
            # Basic file info
            parsed_data["File Name"] = os.path.basename(image_path)
            
            # Basic image metadata
            if hasattr(img, 'datetime_original'):
                parsed_data["Date/Time Original"] = img.datetime_original
            
            if hasattr(img, 'model'):
                parsed_data["Camera Model Name"] = img.model
                
            # Image dimensions - using pixel dimensions from EXIF
            if hasattr(img, 'pixel_x_dimension') and hasattr(img, 'pixel_y_dimension'):
                parsed_data["Exif Image Width"] = str(img.pixel_x_dimension)
                parsed_data["Exif Image Height"] = str(img.pixel_y_dimension)
            
            # Camera specs
            if hasattr(img, 'focal_length'):
                parsed_data["Focal Length"] = f"{img.focal_length:.1f}"
            
            if hasattr(img, 'focal_length_in_35mm_film'):
                parsed_data["Focal Length In 35mm Format"] = f"{img.focal_length_in_35mm_film:.1f}"
            
            # GPS data
            if all(hasattr(img, attr) for attr in ['gps_latitude', 'gps_latitude_ref', 'gps_longitude', 'gps_longitude_ref']):
                # Format GPS coordinates
                lat_str = f"{int(img.gps_latitude[0])} deg {int(img.gps_latitude[1])}' {img.gps_latitude[2]:.2f}\" {img.gps_latitude_ref}"
                lon_str = f"{int(img.gps_longitude[0])} deg {int(img.gps_longitude[1])}' {img.gps_longitude[2]:.2f}\" {img.gps_longitude_ref}"
                parsed_data["GPS Latitude"] = lat_str
                parsed_data["GPS Longitude"] = lon_str
            
            # Altitude data
            if hasattr(img, 'gps_altitude'):
                altitude = float(img.gps_altitude)
                if hasattr(img, 'gps_altitude_ref') and img.gps_altitude_ref == 1:  # 1 indicates below sea level
                    altitude = -altitude
                parsed_data["Absolute Altitude"] = str(altitude)
                parsed_data["Relative Altitude"] = str(altitude)  # Same as absolute if relative not available
            
            # Field of view can be calculated from focal length and sensor size
            # Using a typical value for DJI drones if not calculable
            parsed_data["Field Of View"] = "75.0"
            
            # Gimbal orientation - using default values for nadir view
            # These are typically stored in XMP metadata which exif library doesn't support
            parsed_data["Gimbal Roll Degree"] = "0.0"
            parsed_data["Gimbal Yaw Degree"] = "0.0"
            parsed_data["Gimbal Pitch Degree"] = "-90.0"  # Assuming nadir (downward) view

    except Exception as e:
        print(f"Error reading EXIF data from {image_path}: {str(e)}")
        return None
    
    return parsed_data


def dms_to_dd(dms_str, ref):
    """Converts a DMS (Degrees, Minutes, Seconds) string to Decimal Degrees."""
    try:
        # Using regex to find numbers: handles various formats
        parts = re.findall(r'(\d+\.?\d*)', dms_str)
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        
        dd = degrees + minutes / 60.0 + seconds / 3600.0
        
        if ref in ['S', 'W']:
            dd *= -1
        return dd
    except (ValueError, IndexError) as e:
        print(f"Error parsing DMS string '{dms_str}': {e}")
        return None


def create_geotiff(parsed_data, source_jpg_path, output_tiff_path, compress=True, quality=85):
    """
    Creates a georeferenced TIFF from a JPG using its parsed EXIF metadata.
    """
    # --- Step A: Prepare data from the parsed dictionary ---
    try:
        # Image dimensions
        img_width = int(parsed_data['Exif Image Width'])
        img_height = int(parsed_data['Exif Image Height'])

        # Camera position
        lat_dd = dms_to_dd(parsed_data['GPS Latitude'], parsed_data['GPS Latitude'].split()[-1])
        lon_dd = dms_to_dd(parsed_data['GPS Longitude'], parsed_data['GPS Longitude'].split()[-1])
        # Use Absolute Altitude for sea level reference, Relative for AGL
        altitude_m = float(parsed_data['Absolute Altitude']) 
        
        # Camera orientation (gimbal)
        yaw_deg = float(parsed_data['Gimbal Yaw Degree'])
        pitch_deg = float(parsed_data['Gimbal Pitch Degree'])
        roll_deg = float(parsed_data['Gimbal Roll Degree'])

        # Camera intrinsics
        fov_deg = float(parsed_data['Field Of View'].split()[0])

    except (KeyError, ValueError) as e:
        print(f"Missing or invalid essential metadata: {e}")
        return

    # --- Step B: Convert camera position to UTM coordinates ---
    cam_x, cam_y, zone_num, zone_letter = utm.from_latlon(lat_dd, lon_dd)
    cam_z = altitude_m

    # --- Step C: Calculate ground coordinates of image corners ---
    # Convert angles to radians for numpy
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    
    # Calculate ground footprint size assuming a flat plane at sea level (z=0)
    # This uses the altitude above the ground plane. If using absolute alt, this is an approximation.
    # For higher accuracy, you'd need a Digital Elevation Model (DEM).
    ground_height_m = 2 * cam_z * np.tan(np.radians(fov_deg / 2))
    aspect_ratio = img_width / img_height
    ground_width_m = ground_height_m * aspect_ratio

    # Define corners in the camera's coordinate system (X right, Y down, Z forward)
    # The image plane is effectively at a distance of `cam_z` from the ground
    half_w = ground_width_m / 2
    half_h = ground_height_m / 2
    
    # Define corners relative to the point directly below the camera
    corners_relative = np.array([
        [-half_w,  half_h, 0],  # Top-Left
        [ half_w,  half_h, 0],  # Top-Right
        [ half_w, -half_h, 0],  # Bottom-Right
        [-half_w, -half_h, 0]   # Bottom-Left
    ])

    # Create 3D rotation matrix (Yaw, Pitch, Roll)
    # Note: A pitch of -90 deg (downward) is the reference for a nadir image
    # We adjust pitch because our calculation is based on a nadir projection
    pitch_nadir_adjusted = np.radians(90 + pitch_deg)

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0,           0,          1]])
    
    R_pitch = np.array([[np.cos(pitch_nadir_adjusted), 0, np.sin(pitch_nadir_adjusted)],
                        [0,                          1, 0],
                        [-np.sin(pitch_nadir_adjusted),0, np.cos(pitch_nadir_adjusted)]])
                      
    R_roll = np.array([[1, 0,           0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
    
    R = R_yaw @ R_pitch @ R_roll # Combined rotation matrix

    # Rotate the corner points and add the camera's position
    rotated_corners = R @ corners_relative.T
    world_corners = rotated_corners.T + np.array([cam_x, cam_y, 0])

    # --- Step D: Create Ground Control Points (GCPs) ---
    pixel_corners = [
        (0, 0),                       # Top-Left
        (img_width, 0),               # Top-Right
        (img_width, img_height),      # Bottom-Right
        (0, img_height)               # Bottom-Left
    ]

    gcps = []
    for (col, row), (x, y, z) in zip(pixel_corners, world_corners):
        gcps.append(GroundControlPoint(row=row, col=col, x=x, y=y))

    # Calculate the affine transformation from the GCPs
    transform = from_gcps(gcps)

    # --- Step E: Write the georeferenced TIFF file (Corrected) ---
    try:
        with rasterio.open(source_jpg_path) as src:
            # Get the CRS for the UTM zone
            # Northern Hemisphere: 326xx, Southern Hemisphere: 327xx
            crs_epsg = f"EPSG:326{zone_num}" if lat_dd >= 0 else f"EPSG:327{zone_num}"
            
            # Read the source image's bands and metadata
            src_data = src.read()
            
            # Build the output profile from scratch
            profile = {
                'driver': 'GTiff',
                'height': img_height,
                'width': img_width,
                'count': src.count,      # Number of bands (e.g., 3 for RGB)
                'dtype': src.dtypes[0],  # Data type (e.g., uint8)
                'crs': crs_epsg,
                'transform': transform,
            }
            # If compression is enabled, update the profile
            if compress:
                # Check if the user wants high, lossy compression
                if quality < 100:
                    profile['compress'] = 'jpeg'
                    profile['jpeg_quality'] = quality
                    profile['photometric'] = 'YCBCR' # Recommended for JPEG compression
                else:
                    # Use a good lossless compression
                    profile['compress'] = 'lzw'

            # Write the new GeoTIFF file
            with rasterio.open(output_tiff_path, 'w', **profile) as dst:
                dst.write(src_data)
        
        print(f"✅ Successfully created georeferenced file: {output_tiff_path}")

    except Exception as e:
        print(f"Error writing GeoTIFF: {e}")
        
        
def extract_image_parameters(parsed_data: dict) -> tuple:
    """
    Extract common image parameters from parsed metadata.
    
    Args:
        parsed_data: Dictionary containing image metadata
        
    Returns:
        tuple: (lat_dd, lon_dd, altitude_m, yaw_deg, pitch_deg, roll_deg, fov_deg, img_width, img_height)
    """
    lat_dd = dms_to_dd(parsed_data['GPS Latitude'], parsed_data['GPS Latitude'].split()[-1])
    lon_dd = dms_to_dd(parsed_data['GPS Longitude'], parsed_data['GPS Longitude'].split()[-1])
    altitude_m = float(parsed_data['Absolute Altitude'])
    yaw_deg = float(parsed_data['Gimbal Yaw Degree'])
    pitch_deg = float(parsed_data['Gimbal Pitch Degree'])
    roll_deg = float(parsed_data['Gimbal Roll Degree'])
    fov_deg = float(parsed_data['Field Of View'].split()[0])
    img_width = int(parsed_data['Exif Image Width'])
    img_height = int(parsed_data['Exif Image Height'])
    
    return (lat_dd, lon_dd, altitude_m, yaw_deg, pitch_deg, roll_deg, fov_deg, img_width, img_height)


def calculate_ground_dimensions(altitude_m: float, fov_deg: float, img_width: int, img_height: int) -> tuple:
    """
    Calculate ground footprint dimensions.
    
    Args:
        altitude_m: Camera altitude in meters
        fov_deg: Field of view in degrees
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        tuple: (ground_width_m, ground_height_m)
    """
    ground_height_m = 2 * altitude_m * np.tan(np.radians(fov_deg / 2))
    aspect_ratio = img_width / img_height
    ground_width_m = ground_height_m * aspect_ratio
    return ground_width_m, ground_height_m


def calculate_corner_coordinates(ground_width_m: float, ground_height_m: float, lat_dd: float, lon_dd: float, 
                                 yaw_deg: float, altitude_m: float = 0) -> tuple:
    """
    Calculate corner coordinates in UTM and lat/lon.
    
    Args:
        ground_width_m: Ground width in meters
        ground_height_m: Ground height in meters
        lat_dd: Latitude in decimal degrees
        lon_dd: Longitude in decimal degrees
        yaw_deg: Yaw angle in degrees
        altitude_m: Z coordinate for 3D corners (default: 0)
        
    Returns:
        tuple: (utm_corners, latlon_corners, zone_info)
    """
    # Convert camera position to UTM
    cam_x, cam_y, zone_num, zone_letter = utm.from_latlon(lat_dd, lon_dd)
    
    # Calculate corners
    half_w = ground_width_m / 2
    half_h = ground_height_m / 2
    
    # Create corner points
    corners = np.array([
        [-half_w, half_h, altitude_m],
        [half_w, half_h, altitude_m],
        [half_w, -half_h, altitude_m],
        [-half_w, -half_h, altitude_m]
    ])
    
    # Rotate corners based on yaw
    rotation_matrix = np.array([
        [np.cos(np.radians(yaw_deg)), -np.sin(np.radians(yaw_deg)), 0],
        [np.sin(np.radians(yaw_deg)), np.cos(np.radians(yaw_deg)), 0],
        [0, 0, 1]
    ])
    
    rotated_corners = (rotation_matrix @ corners.T).T
    
    # Add camera position to get world coordinates
    world_corners = rotated_corners + np.array([cam_x, cam_y, 0])
    
    # Convert to lat/lon
    latlon_corners = []
    for x, y, _ in world_corners:
        lat, lon = utm.to_latlon(x, y, zone_num, zone_letter)
        latlon_corners.append((lon, lat))  # GeoJSON uses (lon, lat) order
        
    return world_corners, latlon_corners, (zone_num, zone_letter)


def image_contains_points(parsed_data: dict, gdf: gpd.GeoDataFrame) -> bool:
    """
    Check if any points from the GeoDataFrame fall within the image's footprint.
    
    Args:
        parsed_data: Dictionary containing image metadata
        gdf: GeoDataFrame containing points to check
        
    Returns:
        bool: True if any point falls within the image footprint
    """
    try:
        # Extract parameters
        params = extract_image_parameters(parsed_data)
        lat_dd, lon_dd, altitude_m, yaw_deg, _, _, fov_deg, img_width, img_height = params
        
        # Calculate ground dimensions
        ground_width_m, ground_height_m = calculate_ground_dimensions(altitude_m, fov_deg, img_width, img_height)
        
        # Get corner coordinates
        _, latlon_corners, _ = calculate_corner_coordinates(
            ground_width_m, ground_height_m, lat_dd, lon_dd, yaw_deg
        )
        
        # Create polygon from corners
        image_footprint = Polygon(latlon_corners)
        
        # Convert footprint to same CRS as GeoDataFrame if needed
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Check if any point intersects with the footprint
        return any(point.intersects(image_footprint) for point in gdf.geometry)
        
    except (KeyError, ValueError) as e:
        print(f"Error checking image footprint: {e}")
        return False
    

def process_excel_to_geojson(src_path: str) -> str:
    """
    Process Excel file from source directory into a GeoJSON file.
    
    Args:
        src_path: Path to directory containing the Excel file
        
    Returns:
        str: Path to the output GeoJSON file
        
    Raises:
        AssertionError: If there isn't exactly one Excel file
        ValueError: If required headers aren't found
    """
    # 1. Get the list of Excel files in the source directory
    xlsx_files = [os.path.join(src_path, f) for f in os.listdir(f"{src_path}/") if f.endswith(".xlsx")]
    assert len(xlsx_files) == 1, "There should be exactly one Excel file in the directory."

    xlsx_path = xlsx_files[0]
    metadata_df = pd.read_excel(xlsx_path, header=None)

    # 2. Find the row index that contains the actual headers
    header_row_index = -1
    for i, row in metadata_df.iterrows():
        if 'Latitude' in row.astype(str).values:
            header_row_index = i
            break

    if header_row_index == -1:
        raise ValueError("Could not find the header row containing 'Latitude'. Check the CSV file.")

    # 3. Set the located row as the new column headers
    metadata_df.columns = metadata_df.iloc[header_row_index]

    # Force columns to numeric, coercing errors to NaN
    metadata_df['Longitude'] = pd.to_numeric(metadata_df['Longitude'], errors='coerce')
    metadata_df['Latitude'] = pd.to_numeric(metadata_df['Latitude'], errors='coerce')

    # Drop any rows that failed conversion
    metadata_df.dropna(subset=['Longitude', 'Latitude'], inplace=True)

    # Create the GeoDataFrame
    geometry = [Point(xy) for xy in zip(metadata_df['Longitude'], metadata_df['Latitude'])]
    gdf = gpd.GeoDataFrame(metadata_df, geometry=geometry, crs="EPSG:4326")

    # Save the final output
    output_geojson_path = os.path.join(src_path, f"{os.path.splitext(os.path.basename(xlsx_path))[0]}.geojson")
    gdf.to_file(output_geojson_path, driver='GeoJSON')

    print(f"✅ Successfully cleaned data and created GeoJSON file at: {output_geojson_path}")
    return output_geojson_path
        
        
# -----------------------------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Georeference drone images and optionally process accompanying Excel data."
    )
    parser.add_argument(
        "-i", "--images",
        required=True,
        help="Directory containing the drone images to process",
        type=str
    )
    parser.add_argument(
        "-e", "--excel",
        help="Path to Excel file with additional metadata (optional)",
        type=str,
        required=False
    )
    parser.add_argument(
        "--compress",
        help="Enable TIFF compression (default: True)",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--quality",
        help="JPEG compression quality (1-100, default: 85)",
        type=int,
        default=85
    )
    parser.add_argument(
        "--filter",
        help="Only process images that contain points from the Excel data",
        action="store_true",
        default=False
    )

    args = parser.parse_args()

    # Convert relative paths to absolute
    image_dir = os.path.abspath(args.images)
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        return

    # Process images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    print(f"Found {len(image_files)} images to process...")

    # Load GeoJSON data if we're filtering and Excel file is provided
    gdf = None
    if args.filter and args.excel:
        print("\nProcessing Excel file for filtering...")
        excel_dir = os.path.dirname(os.path.abspath(args.excel))
        try:
            geojson_path = process_excel_to_geojson(excel_dir)
            gdf = gpd.read_file(geojson_path)
            print(f"Loaded {len(gdf)} points from GeoJSON for filtering")
        except Exception as e:
            print(f"Error processing Excel file for filtering: {e}")
            return

    # Process each image
    processed_count = 0
    for i_idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        output_path = ".".join(image_path.split(".")[:-1]) + ".tif"
        
        print(f"\nProcessing image {i_idx}/{len(image_files)}: {image_file}")
        
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping...")
            continue
        
        # Parse the metadata
        parsed_metadata = parse_image(image_path)
        if parsed_metadata is None:
            print(f"Failed to parse metadata for {image_file}. Skipping...")
            continue

        # Check if image contains points when filtering is enabled
        if args.filter and gdf is not None:
            if not image_contains_points(parsed_metadata, gdf):
                print("Image does not contain any points from Excel data. Skipping...")
                continue
            else:
                print("Image contains points from Excel data. Processing...")

        try:
            # Create georeferenced TIFF
            create_geotiff(
                parsed_metadata,
                image_path,
                output_path,
                compress=args.compress,
                quality=args.quality
            )
            processed_count += 1

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    print(f"\nSuccessfully processed {processed_count} out of {len(image_files)} images.")

    # Process Excel file if provided
    if args.excel:
        excel_path = os.path.abspath(args.excel)
        if not os.path.exists(excel_path):
            print(f"\nError: Excel file '{excel_path}' does not exist.")
            return
        
        excel_dir = os.path.dirname(excel_path)
        try:
            geojson_path = process_excel_to_geojson(excel_dir)
            print(f"\nSuccessfully processed Excel data to: {geojson_path}")
        except Exception as e:
            print(f"\nError processing Excel file: {e}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()


