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

## License
This software is licensed under the [Creative Commons Attribution License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). The Creative Commons Attribution license allows re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited. Please cite the associated publication in the Planetary Science Journal (Phillips et al. in press). The URL and DOI will be added to this section when the publication is available.

