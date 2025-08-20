# Marine-Debris
(for finding trash)

## Install

After installing [`miniconda`](https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Windows-x86_64.exe), run the 
following from Anaconda Prompt:

```bash
# cmd

# Create a conda environment
conda create --name debris python=3.10 -y

# Activate the environment
conda activate debris

# Install the following from conda-forge
conda install git -y

# Install the folling from pip:
pip install uv
```

Next we're going to pull this repository down so we can run scripts as needed:

```bash
# cmd

# Navigate to where you want this repository to be on your local machine. Example below: 
cd Documents/
mkdir GitHub
cd GitHub

# Clone the repository
git clone https://github.com/Jordan-Pierce/Marine-Debris.git

# Navigate into the respository folder
cd Marine-Debris

# Now install the packages using the setup.py file
uv pip install -e .
```

### Usage

For georeferencing images, the following will create "georeferenced" images from the images in the image folder. If 
you provide a path to the excel file, it will also convert this into a geojson file. You can also compress the file, 
if you provide the excel file and add the tag for filter, only images that have an annotated point will be saved.
```bash
# cmd

python src\georeference_images.py --images path/to/image/folder --excel path/to/excel_file --compress --filter
```

### Notes

`arcpy` is needed to run `tile_Orthomosaic.py`. It is recommended to use the Anaconda environment with ArcGIS Pro 
installed. The `arcpy` module is not available in the default Anaconda environment.