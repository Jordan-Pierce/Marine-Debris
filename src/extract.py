import os
import glob
import argparse
import pandas as pd
import traceback

from torch.utils.data import DataLoader
from torchgeo.datasets import GeoDataset, Landsat, NASAMarineDebris, NAIP, RasterDataset, NonGeoDataset
from rasterio.crs import CRS
import rasterio
import torchgeo
from torchgeo.samplers import RandomGeoSampler


from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

import cv2
from tqdm import tqdm
import panoptes_client

def test(image_dir, output_dir):

    # landsat_dataset = Landsat(output_dir)
    # nasa_dataset = NASAMarineDebris(output_dir)
    # naip_dataset = NAIP(output_dir)
    #
    # print("landsat", landsat_dataset)



    #torchgeo.datasets.PatternNet(root=image_dir)

    images = []

    src_crs = CRS.from_epsg(6507)

    for image in os.listdir(image_dir):

        image = os.path.join(image_dir, image)

        new_path = f"{image_dir}/Test.tiff"

        images.append(image)

        # im = Image.open(image)
        # im.save(new_path, "TIFF")
        #
        # raster = rasterio.open(new_path)
        # print("raster info", raster.crs)

    #     with rasterio.open(image) as src:
    #         with rasterio.vrt.WarpedVRT(src, crs=src_crs) as vrt:
    #             data = vrt.read()
    #
    # print(src_crs)
    raster = RasterDataset(root=image_dir, crs=src_crs, res=10, transforms=None)
    #raster = RasterDataset(root=image_dir)
    #test = NonGeoDataset(root=image_dir)

    #print(raster)

        # exif = Image.open(image)._getexif()
        #
        # for tagid in exif:
        #
        #     tagname = TAGS.get(tagid, tagid)
        #
        #     value = exif.get(tagid)
        #
        #     print(f"{tagname}: {value}")

        #print(exif)

        #print(image)

        # dataset = GeoDataset(image)
        #
        # sampler = RandomGeoSampler(dataset, size=512, roi=None, length=10)
        #
        # dataloader = DataLoader(dataset, sampler)


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize non-reduced and reduced annotations")

    parser.add_argument("--csv", type=str,
                        help="The CSV with debris annotations")

    parser.add_argument("--image_dir", type=str,
                        default=f'/Kira/GitHub/Marine-Debris/Data/April 2024/2024_04_12_Deer Island/DJI_202404120832_001_DEERISLANDMD-AREA',
                        help="The image directory")

    parser.add_argument("--output_dir", type=str,
                        default='./Output',
                        help="The output directory")

    args = parser.parse_args()

    # Parse out arguments
    csv = args.csv
    image_dir = args.image_dir
    output_dir = args.output_dir

    # Turn both csv files into pandas dataframes
    #csv = pd.read_csv(csv)

    try:

        test(image_dir, output_dir)


        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
