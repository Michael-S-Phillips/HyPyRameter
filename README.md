# HyPyRameter
This repo hosts code that calculates spectral parameters on hyperspectral data.

## Overview
The ImageCubeParameters module is designed to handle hyperspectral images in ENVI format (.hdr/.img) and calculate spectral parameters. It also generates browse product summary images based on the input hyperspectral data.
The PointSpectraParameters module is designed to handle calculation of parameters from individual spectra from .sed files

### Installation
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
from hypyrameter.ImageCubeParameters.paramCalculator import paramCalculator
# Instantiate the class and select your input image and output directory
pc = paramCalculator()
# Run the calculator and save the results
pc.run()
```

### Point Spectra Parameters
See the PointSpectraParameters.ipynb file for how to run the Point Spectra Parameters calculation.



