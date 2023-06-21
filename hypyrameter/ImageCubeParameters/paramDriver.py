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
file_name = vhdr.split('.')[0].split('/')[-1]

# If you have a bad bands list, instantiate it here, it may be included in the header file of your image.
# bbl = np.hstack((np.linspace(75,88,(88-75+1)),np.linspace(167,189,(189-167+1))))
# bbl = [int(i) for i in bbl]

# p is the paramCalculator object
p = paramCalculator(data_path,file=vhdr)

# interpolate through NaNs - this only works if the nan values are at consistent wavelengths across the whole cube
# interp = interpNaNs(p.f, p.f_bands)
# interp.linearInterp()
# p.f = interp.data_cube

# optionally preview your data
p.previewData()

# optionally transpose the data for better display
# p.v = np.transpose(p.v,axes=(1,0,2))
# p.s = np.transpose(p.s,axes=(1,0,2))

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

try:
    envi.save_image(params_file_name, params,
                    metadata=meta, dtype=np.float32)
except EnviException as error:
    print(error)
    choice = input('file exists, would you like to overwite?\n\ty or n\n')
    choice = choice.lower()
    if choice == 'y':
        envi.save_image(params_file_name, params,
                        metadata=meta, dtype=np.float32, force=True)
    else:
        pass
        


#%% MNF block
'''
calculate minimum noise fraction transform images and save as PNG and ENVI .img
'''
# #SWIR
# s_mnf10 = p.SWIR_MNF()
# bs = [4,3,2] #because cv2 writes BGR for some reason
# sName = '/SWIR_MNF_234.png'
# cv2.imwrite(savepath+sName,s_mnf10[:,:,bs])
# sName = '/SWIR_MNF_234_8bit.png'
# cv2.imwrite(savepath+sName,u.browse2bit(s_mnf10[:,:,bs]))

# bandList = ['1','2','3','4','5','6','7','8','9','10']
# sMeta = p.s_.metadata.copy()
# sMeta['wavelength'] = bandList
# sMeta['wavelength units'] = 'MNF Band'
# sMeta['default bands'] = ['1', '2', '3']
# envi.save_image(savepath+'/SWIR_MNF.hdr', s_mnf10, metadata=sMeta,dtype=np.float32)


# #Vis
# v_mnf10 = p.VIS_MNF()
# bs = [4,3,2]
# vName = '/VNIR_MNF_234.png'
# cv2.imwrite(savepath+vName,v_mnf10[:,:,bs])
# vName = '/VNIR_MNF_234_8bit.png'
# cv2.imwrite(savepath+vName,u.browse2bit(v_mnf10[:,:,bs]))

# bandList = ['1','2','3','4','5','6','7','8','9','10']
# vMeta = p.v_.metadata.copy()
# vMeta['wavelength'] = bandList
# vMeta['wavelength units'] = 'MNF Band'
# vMeta['default bands'] = ['1', '2', '3']
# envi.save_image(savepath+'/VIS_MNF.hdr', v_mnf10, metadata=vMeta,dtype=np.float32)

#%% browse block
'''
generate one-off browse product images (as PNG files)
'''
savepath = data_path + '/BrowseProducts/'
if os.path.isdir(savepath) is False:
    os.mkdir(savepath)

# calculate and save browse products as .png
bp = p.MAF()
n = 'MAF'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.FM2()
n = 'FM2'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.FAL()
n = 'FAL'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.PAL()
n = 'PAL'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.PFM()
n = 'PFM'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.PHY()
n = 'PHY'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.CR2()
n = 'CR2'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.HYD()
n = 'HYD'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.CHL()
n = 'CHL'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.HYS()
n = 'HYS'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)


