# HyPyRameter
This repo hosts code that calculates spectral parameters on hyperspectral data.

## Overview
The ImageCubeParameters module is designed to handle hyperspectral images in ENVI format (.hdr/.img) and calculate spectral parameters. It also generates browse product summary images based on the input hyperspectral data.
The PointSpectraParameters module is designed to handle calculation of parameters from individual spectra from .sed files

## Installation
First, make sure your channels are set in the correct order
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
Alternatively, HyPyRameter can be used with the Spectral Cube Analysis Tool (SCAT). It is installed along with SCAT, which provides a GUI for running spectral parameters calculations. See the documentation for SCAT for details (https://github.com/Michael-S-Phillips/SCAT).

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

