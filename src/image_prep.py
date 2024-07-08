import os
import tempfile
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pystac
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url
from torchgeo.samplers import RandomGeoSampler


plt.rcParams["figure.figsize"] = (12, 12)


def collect_image_paths(root, extensions=(".JPG",)):
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, filename))
    print(f"Found {len(image_paths)} image paths.")
    return image_paths

class MyDataset(RasterDataset):
    filename_glob = "*.JPG"
    filename_regex = ".*"
    date_format = None
    is_image = True
    separate_files = False

    def __init__(self, root: str, extensions=(".JPG",)):
        super().__init__(root)
        self.images = collect_image_paths(root, extensions)

    def __getitem__(self, index):
        image_path = self.images[index]

        metadata = self.get_metadata(image_path)
        print(f"Metadata for {image_path}: {metadata}")

        image = Image.open(image_path)
        image = image.convert('RGB')  # Ensure 3-channel RGB
        image = torch.tensor(np.array(image))  # Convert to tensor
        return image

    def __len__(self):
        return len(self.images)


dataset = MyDataset(root="/Users/alliebattista/Documents/DeerIslandImages")
print(f"Number of images found: {len(dataset)}")
