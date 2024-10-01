# HyPyRameter
This repo hosts code that calculates spectral parameters on hyperspectral reflefctance data.

[![DOI](https://zenodo.org/badge/567958968.svg)](https://zenodo.org/doi/10.5281/zenodo.10801541)

## Overview
The ImageCubeParameters module is designed to handle hyperspectral images in ENVI format (.hdr/.img) and calculate spectral parameters. It also generates browse product summary images based on the input hyperspectral data.
The PointSpectraParameters module is designed to handle calculation of parameters from individual spectra from .sed files, a csv file, or a pandas data frame. 

## Installation
First, create a new environment to install hypyrameter:
```bash
conda create -n hypyrameter
conda activate hypyrameter
```
Then, make sure your channels are set in the correct order:
```bash
conda config --add channels michael--s--phillips
conda config --add channels conda-forge
```
This should set conda-forge above michael--s--phillips like this:
```bash
conda config --show channels
channels:
    - conda-forge
    - michael--s--phillips
    - your other channels
```
Then, run the following command to install hypyrameter:
```bash
conda install hypyrameter
```
### For Developers
Clone the repository
```bash
git clone git@github.com:Michael-S-Phillips/HyPyRameter.git
```
Navigate to the repository:
```bash
cd HyPyRameter
```
Install the required packages using conda
```bash
conda env create -f environment.yml
```
This will create a conda environment called 'hypyrameter'. 
Activate the environment:
```bash
conda activate hypyrameter
```

## Usage
### Image Cube Parameters
```python
#Import the module
from hypyrameter.paramCalculator import cubeParamCalculator
# Instantiate the class and select your input image and output directory
pc = cubeParamCalculator()
# Run the calculator and save the results
pc.run()
```

### Point Spectra Parameters
See the PointSpectraParameters.ipynb file for how to run the Point Spectra Parameters calculation.

### Usage with SCAT 
Alternatively, HyPyRameter can be used with the Spectral Cube Analysis Tool (SCAT). It is installed along with SCAT, which provides a GUI for running spectral parameters calculations. See the documentation for SCAT for details (https://github.com/Michael-S-Phillips/SCAT).

## Detailed Description of HyPyRameter Contents
### Specific Listings of the Types and Formats of Files Uploaded

#### Python Scripts:
- `interpNans.py`
- `iovf_generic_utils.py`
- `iovf_generic.py`
- `paramCalculator.py`
- `utils.py`

#### Jupyter Notebooks:
- `ImageCubeParamsExample.ipynb`
- `PointSpectraParamsExample.ipynb`

#### MATLAB Script:
- `ngMeet_denoiser.m`

### Tools Needed to Parse or Reuse This Material

#### Python:
- The primary language used for the scripts and notebooks. Python 3.x is required to run these scripts.
- Libraries used include `numpy`, `timeit`, `cv2`, `matplotlib`, `spectral`, `multiprocessing`, `tqdm`, `pandas`, `tkinter`, and `os`. See the environment.yml file for more information.

#### Jupyter Notebook:
- Used for interactive data analysis and visualization. Requires Jupyter Notebook or JupyterLab to run.

#### MATLAB:
- The `ngMeet_denoiser.m` script requires MATLAB to run. This is an optional denoising routine, it is not required to run HyPyRameter.

### Explanations of the Uses for Uploaded Code

#### `interpNans.py`:
- Contains the `interpNaNs` class for interpolating NaN values in data cubes.

#### `iovf_generic_utils.py`:
- Utility functions for the `iovf_generic` module.

#### `iovf_generic.py`:
- Contains the `iovf` class for denoising hyperspectral images. This is the 'iterative outlier voting filter' for spectral image cube denoising. This is a computationally intensive function, so use with caution. See [Phillips et al., 2023](https://www.sciencedirect.com/science/article/pii/S0019103523002890) for more details.

#### `paramCalculator.py`:
- Contains the `cubeParamCalculator` and `pointParamCalculator` classes for calculating spectral parameters from hyperspectral images and point spectra, respectively.
- Example methods include `denoiser`, `previewData`, and various parameter calculation methods like `R463`, `BD530_2`, etc.

#### `utils.py`:
- Utility functions for handling spectral data files, such as `getSedFiles` and `getSpecFiles` and core functions for calculating parameters.

#### `ImageCubeParamsExample.ipynb`:
- Example notebook demonstrating how to use the `cubeParamCalculator` class to calculate spectral parameters from hyperspectral image cubes.

#### `PointSpectraParamsExample.ipynb`:
- Example notebook demonstrating how to use the `pointParamCalculator` class to calculate spectral parameters from point spectra data.

#### `ngMeet_denoiser.m`:
- MATLAB script for denoising hyperspectral images using the [NGMeet](https://www.mathworks.com/help/images/ref/denoisengmeet.html) algorithm.

These scripts and notebooks are designed to facilitate the calculation of spectral parameters from hyperspectral data, providing tools for data preprocessing, denoising, and parameter extraction.

## License
This software is licensed under the [Creative Commons Attribution License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). The Creative Commons Attribution license allows re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited. Please cite the associated publication in the Planetary Science Journal (Phillips et al. in press). The URL and DOI will be added to this section when the publication is available.

