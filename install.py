import os
import sys
import shutil
import platform
import subprocess

# ----------------------------------------------
# OS
# ----------------------------------------------
osused = platform.system()

if osused not in ['Windows', 'Linux', 'Darwin']:
    raise Exception("This install script is only for Windows, Linux, or macOS")

# ----------------------------------------------
# Conda
# ----------------------------------------------
# Need conda to install NVCC if it isn't already
console_output = subprocess.getstatusoutput('conda --version')

# Returned 1; conda not installed
if console_output[0]:
    raise Exception("This install script is only for machines with Conda already installed")

conda_exe = shutil.which('conda')

# ----------------------------------------------
# Python version
# ----------------------------------------------
python_ver = int(sys.version_info[1])

# check python version
if python_ver != 8:
    raise Exception(f"Only Python 3.8 is supported.")

# ---------------------------------------------
# Requirements file
# ---------------------------------------------
requirements_file = 'requirements.txt'

# Check if requirements.txt exists
if not os.path.isfile(requirements_file):
    print(f"ERROR: {requirements_file} not found in the current directory.")
    sys.exit(1)

# ---------------------------------------------
# MSVC for Windows (skipped for macOS)
# ---------------------------------------------
if osused == 'Windows':
    try:
        print(f"NOTE: Installing msvc-runtime")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'msvc-runtime'])
    except Exception as e:
        print(f"There was an issue installing msvc-runtime\n{e}")
        sys.exit(1)

# ----------------------------------------------
# CUDA Toolkit version
# ----------------------------------------------
try:
    if osused == 'Darwin':
        raise Exception("CUDA Toolkit is not supported on macOS.")

    # Command for installing cuda nvcc
    conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-11.8.0", "cuda-nvcc", "-y"]

    # Run the conda command
    print("NOTE: Installing CUDA NVCC 11.8")
    subprocess.run(conda_command, check=True)

    # Command for installing cuda nvcc
    conda_command = [conda_exe, "install", "-c", f"nvidia/label/cuda-11.8.0", "cuda-toolkit", "-y"]

    # Run the conda command
    print("NOTE: Installing CUDA Toolkit 11.8")
    subprocess.run(conda_command, check=True)

except Exception as e:
    if osused != 'Darwin':
        print("ERROR: Could not install CUDA Toolkit")
        sys.exit(1)
    else:
        print("Skipping CUDA Toolkit installation on macOS")

# ----------------------------------------------
# Pytorch
# ----------------------------------------------
try:
    torch_package = 'torch==2.0.0'
    torchvision_package = 'torchvision==0.15.1'
    torch_extra_argument1 = '--extra-index-url'
    torch_extra_argument2 = 'https://download.pytorch.org/whl/cu118'

    if osused != 'Darwin':
        # Setting Torch, Torchvision versions for CUDA
        torch_package += '+cu118'
        torchvision_package += '+cu118'
        list_args = [sys.executable, "-m", "pip", "install", torch_package, torchvision_package, torch_extra_argument1, torch_extra_argument2]
    else:
        # Setting Torch, Torchvision versions for CPU
        list_args = [sys.executable, "-m", "pip", "install", torch_package, torchvision_package]

    # Installing Torch, Torchvision
    print("NOTE: Installing Torch 2.0.0")
    subprocess.check_call(list_args)

except Exception as e:
    print("ERROR: Could not install Pytorch")
    sys.exit(1)

# ----------------------------------------------
# Other dependencies
# ----------------------------------------------
# Read packages from requirements.txt
with open(requirements_file, 'r') as file:
    install_requires = [line.strip() for line in file if line.strip() and not line.startswith('#')]

# Installing all the packages
for package in install_requires:
    try:
        print(f"NOTE: Installing {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"There was an issue installing {package}\n{e}\n")
        print(f"If you're not already, please try using a conda environment with python 3.8")
        sys.exit(1)

print("Done.")
