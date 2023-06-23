#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:40:28 2022

This script is designed to drive the paramCalculator class in the associated paramCalculator.py file.
The paramCalculator class generates spectral parameter prodcuts from hyperspectral image data.

The Spectral (https://www.spectralpython.net) package is used for image ingest. Specifically, the envi i/o subpackage is used. 
The paramCalculator class could be edited in future versions to accomodate different image formats.

@author: 
    Michael S. Phillips, JHU/APL
    Christian Tai Udovicic, NAU

"""
#%% import modules 
import os
import numpy as np
import cv2
import param_utils as u
from paramCalculator import paramCalculator
import spectral.io.envi as envi
from spectral.io.envi import EnviException as EnviException
from interpNans import interpNaNs

#%% data ingestion and paramCalculator initiation

# Path to your data directory
# data_path = os.path.abspath(os.path.join('../../data/HyPyRameter/data'))
data_path = os.path.abspath(os.path.join('/Volumes/Arrakis/HySpex_Iceland/HyPyRameter/data'))

# path to your ENVI .hdr file. For VNIR (400-1000 nm) files use vhdr, for SWIR (1000-2600 nm) files use shdr 
vhdr = data_path + '/iceland_vnir_drone_crop.hdr'
shdr = data_path + '/iceland_swir_tripod_crop.hdr'
jhdr = data_path + '/EMIT_L2A_RFL_001_20230329T145406_2308809_052_reflectance_cropped.hdr'
file_name = jhdr.split('.')[0].split('/')[-1]

# If you have a bad bands list, instantiate it here, it may be included in the header file of your image.
# bbl = np.hstack((np.linspace(75,88,(88-75+1)),np.linspace(167,189,(189-167+1))))
# bbl = [int(i) for i in bbl]

# p is the paramCalculator object
p = paramCalculator(data_path,file=jhdr, denoise=True)
print(f'list of valid parameters:\n\t{p.validParams}')

# interpolate through NaNs - this only works if the nan values are at consistent wavelengths across the whole cube
# interp = interpNaNs(p.f, p.f_bands)
# interp.linearInterp()
# p.f = interp.data_cube

# optionally preview your data
p.previewData()

# optionally transpose the data for better display
# p.f = np.transpose(p.f,axes=(1,0,2))

# savepath is where to save your output parameter images
savepath = data_path+'/parameters/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)

#%% Calculate valid parameters and browse products
'''
calculate spectral parameters within the valid wavelength range of the image cube, save as a .img and browse products as .png
'''
params_file_name = savepath+file_name+'_params.hdr'
meta = p.f_.metadata.copy()
meta['wavelength'] = p.validParams
meta['band names'] = p.validParams
meta['wavelength units'] = 'parameters'
meta['default bands'] = ['R637', 'R550', 'R463']
print(f'calculating valid parameters\n{p.validParams}')
params = p.calculateParams()
print('calculating valid browse products')
browseProducts = p.calculateBrowse(params,savepath)
p.saveParamCube(params, params_file_name, meta)
        
# %%
