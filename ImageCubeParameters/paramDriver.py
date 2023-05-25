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
import os
import numpy as np
import cv2
from param_utils import utility_functions as u
from paramCalculator import paramCalculator
import spectral.io.envi as envi
from interpNans import interpNaNs

#%% data ingestion and paramCalculator initiation

# Path to your data directory
data_path = os.path.abspath(os.path.join('/Users/phillms1/Documents/Work/RAVEN/HyPyRameter/data'))

# path to your ENVI .hdr file. For VNIR (400-1000 nm) files use vhdr, for SWIR (1000-2600 nm) files use shdr 
file_name = 'EMIT_L2A_RFL_001_20230329T145406_2308809_052_reflectance_cropped'
vhdr = False
shdr = False
jhdr = data_path + '/'+file_name+'.hdr'
# jhdr = data_path + '/EMIT_L2A_RFL_001_20230329T145406_2308809_052_reflectance_cropped.img.hdr'

# If you have a bad bands list, instantiate it here.
# bbl = np.hstack((np.linspace(75,88,(88-75+1)),np.linspace(167,189,(189-167+1))))

# p is the paramCalculator object
p = paramCalculator(data_path,vhdr,shdr,jhdr)

'''
there is a "denoise" option in the paramCalculator. It uses the non-global meets local 
low-rank tensor solution from this paper https://ieeexplore.ieee.org/document/9208755. 
Based on experience, I recommend caution if using this denoise routine because it can change
your data in unexpected and non-ideal ways. If your image has < 2 million pixels, 
(or if you have a system with a lot of memory and compute power) you can use
the iovf_generic noise remediation routine that is in the attached .py document with 
the same name. This works quite well and applies noise remediation with a light touch.
'''
# interpolate through NaNs - this only works if the nan values are at consistent wavelengths across the whole cube
# p.j = np.where(p.j<0,np.nan, p.j)
# interp = interpNaNs(p.j, p.j_bands)
# interp.linearInterp()
# p.j = interp.data_cube

# optionally preview your data
p.previewData()

# optionally transpose the data for better display
# p.v = np.transpose(p.v,axes=(1,0,2))
# p.s = np.transpose(p.s,axes=(1,0,2))

# savepath is where to save your output parameter images
savepath = data_path+'/parameters/'
if not os.path.isdir(savepath):
    os.mkdir(savepath)

#%% SWIR parameter block
'''
calculate spectral parameters from SWIR image data, save as .img and as .png
'''
from spectral.io.envi import EnviException as EnviException
#------------------------------------------------------------------------------
#------------- SWIR Parameters ------------------------------------------------
#------------------------------------------------------------------------------

# full path for the output parameter ENVI file
sParams_file_name = savepath+'/'+file_name+'_params.hdr'

if shdr != False:
    paramList = ['OLINDEX3','LCPINDEX2','HCPINDEX2','BD1400','BD1450', 'BD1900_2',
                 'BD1900r2','BD2100_2','BD2165','BD2190','BD2210_2','BD2250','BD2290',
                 'BD2355','BDCARB','D2200','D2300','IRR2','ISLOPE','MIN2250','MIN2295_2480',
                 'MIN2345_2537','R2529', 'R1506', 'R1080','SINDEX2','BD1200','BD2100_3','GypTrip',
                 'ILL']
    sMeta = p.s_.metadata.copy()
    sMeta['wavelength'] = paramList
    sMeta['band names'] = paramList
    sMeta['wavelength units'] = 'parameters'
    sMeta['default bands'] = ['R2529', 'R1506', 'R1080']
    print('calculating SWIR parameters')
    sParams=p.calculateSwirParams()
    try:
        envi.save_image(sParams_file_name, sParams,
                        metadata=sMeta, dtype=np.float32)
    except EnviException as error:
        print(error)
        choice = input('file exists, would you like to overwite?\n\ty or n\n')
        choice = choice.lower()
        if choice == 'y':
            envi.save_image(sParams_file_name, sParams,
                            metadata=sMeta, dtype=np.float32, force=True)
        else:
            pass
        

    #--------- SWIR Browse PNG ----------------------------------------------------
    i0 = paramList.index('OLINDEX3')
    i1 = paramList.index('LCPINDEX2')
    i2 = paramList.index('HCPINDEX2')
    MAF = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    MAF = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,MAF))),axis=2)
    n = '/MAF.png'
    cv2.imwrite(savepath+n, MAF)

    i0 = paramList.index('R2529')
    i1 = paramList.index('R1506')
    i2 = paramList.index('R1080')
    FAL = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    FAL = np.where(FAL>1,-0.01,FAL)
    # FAL = np.flip(u.browse2bit(FAL),axis=2)
    FAL = np.flip(u.browse2bit(u.stretchNBands(u,FAL)),axis=2)
    n = '/FAL.png'
    cv2.imwrite(savepath+n, FAL)

    i0 = paramList.index('BD2210_2')
    i1 = paramList.index('BD2190')
    i2 = paramList.index('BD2165')
    PAL = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    PAL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PAL))),axis=2)
    n = '/PAL.png'
    cv2.imwrite(savepath+n, PAL)

    i0 = paramList.index('D2200')
    i1 = paramList.index('D2300')
    i2 = paramList.index('BD1900r2')
    PHY = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    PHY = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PHY))),axis=2)
    n = '/PHY.png'
    cv2.imwrite(savepath+n, PHY)

    i0 = paramList.index('BD2355')
    i1 = paramList.index('D2300')
    i2 = paramList.index('BD2290')
    PFM = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    PFM = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PFM))),axis=2)
    n = '/PFM.png'
    cv2.imwrite(savepath+n, PFM)

    i0 = paramList.index('MIN2295_2480')
    i1 = paramList.index('MIN2345_2537')
    i2 = paramList.index('BDCARB')
    CR2 = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    CR2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CR2))),axis=2)
    n = '/CR2.png'
    cv2.imwrite(savepath+n, CR2)

    i0 = paramList.index('SINDEX2')
    i1 = paramList.index('BD2100_2')
    i2 = paramList.index('BD1900_2')
    HYD = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    HYD = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYD))),axis=2)
    n = '/HYD.png'
    cv2.imwrite(savepath+n, HYD)

    i0 = paramList.index('ISLOPE')
    i1 = paramList.index('BD1400')
    i2 = paramList.index('IRR2')
    CHL = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    CHL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CHL))),axis=2)
    n = '/CHL.png'
    cv2.imwrite(savepath+n, CHL)

    i0 = paramList.index('MIN2250')
    i1 = paramList.index('BD2250')
    i2 = paramList.index('BD1900r2')
    HYS = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    HYS = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYS))),axis=2)
    n = '/HYS.png'
    cv2.imwrite(savepath+n, HYS)

    i0 = paramList.index('BD1200')
    i1 = paramList.index('BD1450')
    i2 = paramList.index('BD1900r2')
    HY2 = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    HY2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYS))),axis=2)
    n = '/HY2.png'
    cv2.imwrite(savepath+n, HY2)
    
    i0 = paramList.index('BD1450')
    i1 = paramList.index('BD2100_3')
    i2 = paramList.index('BD1200')
    LIC = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    LIC = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,LIC))),axis=2)
    n = '/LIC.png'
    cv2.imwrite(savepath+n, LIC)
    
    i0 = paramList.index('GypTrip')
    i1 = paramList.index('BD2165')
    i2 = paramList.index('BD1900_2')
    GYP = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    GYP = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,GYP))),axis=2)
    n = '/GYP.png'
    cv2.imwrite(savepath+n, GYP)
    
    i0 = paramList.index('BDCARB')
    i1 = paramList.index('ILL')
    i2 = paramList.index('GypTrip')
    SED = u.buildSummary(sParams[:,:,i0], sParams[:,:,i1], sParams[:,:,i2])
    SED = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,SED))),axis=2)
    n = '/SED.png'
    cv2.imwrite(savepath+n, SED)

#%% VIS parameter block
'''
calculate spectral parameters from VIS image data, save as .img and as .png
'''
#------------------------------------------------------------------------------
#------------- VIS Parameters -------------------------------------------------
#------------------------------------------------------------------------------
vParams_file_name = savepath+'/' + file_name + '_params.hdr'

if vhdr != False:
    paramList = ['R637','R550','R463','SH460','BD530_2','BD670','D700','BD875','BD920_2','RPEAK1','BDI1000VIS','ELMSUL']
    vMeta = p.v_.metadata.copy()
    vMeta['wavelength'] = paramList
    vMeta['band names'] = paramList
    vMeta['wavelength units'] = 'parameters'
    vMeta['default bands'] = ['R637', 'R550', 'R463']
    print('calculating VIS parameters')
    vParams=p.calculateVisParams()
    envi.save_image(vParams_file_name,vParams,metadata=vMeta,dtype=np.float32,force=True)

    #-- VIS Browse PNG ------------------------------------------------------------
    i0 = paramList.index('BD530_2')
    i1 = paramList.index('BD875')
    i2 = paramList.index('BD920_2')
    FM2 = u.buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    FM2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,FM2))),axis=2)
    n = '/FM2.png'
    cv2.imwrite(savepath+n, FM2)
    
    i0 = paramList.index('BD530_2')
    i1 = paramList.index('BD670')
    i2 = paramList.index('BD875')
    HEM = u.buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    HEM = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HEM))),axis=2)
    n = '/HEM.png'
    cv2.imwrite(savepath+n, HEM)
    
    # i0 = paramList.index('RPEAK1')
    # i1 = paramList.index('ELMSUL')
    # i2 = paramList.index('BDI1000VIS')
    # CPL = buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    CPL = p.CPL()
    CPL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CPL))),axis=2)
    n = '/CPL.png'
    cv2.imwrite(savepath+n, CPL)
    
    # i0 = paramList.index('ELMSUL')
    # i1 = paramList.index('ELMSUL')
    # i2 = paramList.index('ELMSUL')
    # SUL = buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    SUL = p.SUL()
    SUL = np.flip(u.browse2bit(u.stretchNBands(u,SUL)),axis=2)
    n = '/SUL.png'
    cv2.imwrite(savepath+n, SUL)

    i0 = paramList.index('R637')
    i1 = paramList.index('R550')
    i2 = paramList.index('R463')
    TRU = u.buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    TRU = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,TRU))),axis=2)
    # TRU = np.flip(browse2bit(TRU),axis=2)
    n = '/TRU.png'
    cv2.imwrite(savepath+n, TRU)
    
#%% Joined Param block
'''
calculate spectral parameters from a joined image cube, save as .img and as .png
use with the vhdr option
'''
from spectral.io.envi import EnviException as EnviException

#------------------------------------------------------------------------------
#------------- VIS Parameters -------------------------------------------------
#------------------------------------------------------------------------------
jParams_file_name = savepath+'/' + file_name + '_params.hdr'

if jhdr != False:
    paramList = ['R637','R550','R463','SH460','BD530_2','BD670','D700','BD875','BD920_2','RPEAK1','BDI1000VIS','ELMSUL',
                 'OLINDEX3','LCPINDEX2','HCPINDEX2','BD1400','BD1450', 'BD1900_2',
                 'BD1900r2','BD2100_2','BD2165','BD2190','BD2210_2','BD2250','BD2290',
                 'BD2355','BDCARB','D2200','D2300','IRR2','ISLOPE','MIN2250','MIN2295_2480',
                 'MIN2345_2537','R2529', 'R1506', 'R1080','SINDEX2','BD1200','BD2100_3','GypTrip',
                 'ILL']
    jMeta = p.j_.metadata.copy()
    jMeta['wavelength'] = paramList
    jMeta['band names'] = paramList
    jMeta['wavelength units'] = 'parameters'
    jMeta['default bands'] = ['R637', 'R550', 'R463']
    print('calculating Joined parameters')
    vParams=p.calculateVisParams()
    sParams=p.calculateSwirParams()
    jParams=np.concatenate((vParams, sParams), axis=2)
    # envi.save_image(savepath+'/vis_parameters.hdr',vParams,metadata=vMeta,dtype=np.float32,force=True)
    try:
        envi.save_image(jParams_file_name, jParams,
                        metadata=jMeta, dtype=np.float32)
    except EnviException as error:
        print(error)
        choice = input('file exists, would you like to overwite?\n\ty or n\n')
        choice = choice.lower()
        if choice == 'y':
            envi.save_image(jParams_file_name, jParams,
                            metadata=jMeta, dtype=np.float32, force=True)
        else:
            pass
        

    #--------- SWIR Browse PNG ----------------------------------------------------
    i0 = paramList.index('OLINDEX3')
    i1 = paramList.index('LCPINDEX2')
    i2 = paramList.index('HCPINDEX2')
    MAF = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    MAF = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,MAF))),axis=2)
    n = '/MAF.png'
    cv2.imwrite(savepath+n, MAF)

    i0 = paramList.index('R2529')
    i1 = paramList.index('R1506')
    i2 = paramList.index('R1080')
    FAL = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    FAL = np.where(FAL>1,-0.01,FAL)
    # FAL = np.flip(u.browse2bit(FAL),axis=2)
    FAL = np.flip(u.browse2bit(u.stretchNBands(u,FAL)),axis=2)
    n = '/FAL.png'
    cv2.imwrite(savepath+n, FAL)

    i0 = paramList.index('BD2210_2')
    i1 = paramList.index('BD2190')
    i2 = paramList.index('BD2165')
    PAL = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    PAL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PAL))),axis=2)
    n = '/PAL.png'
    cv2.imwrite(savepath+n, PAL)

    i0 = paramList.index('D2200')
    i1 = paramList.index('D2300')
    i2 = paramList.index('BD1900r2')
    PHY = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    PHY = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PHY))),axis=2)
    n = '/PHY.png'
    cv2.imwrite(savepath+n, PHY)

    i0 = paramList.index('BD2355')
    i1 = paramList.index('D2300')
    i2 = paramList.index('BD2290')
    PFM = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    PFM = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,PFM))),axis=2)
    n = '/PFM.png'
    cv2.imwrite(savepath+n, PFM)

    i0 = paramList.index('MIN2295_2480')
    i1 = paramList.index('MIN2345_2537')
    i2 = paramList.index('BDCARB')
    CR2 = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    CR2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CR2))),axis=2)
    n = '/CR2.png'
    cv2.imwrite(savepath+n, CR2)

    i0 = paramList.index('SINDEX2')
    i1 = paramList.index('BD2100_2')
    i2 = paramList.index('BD1900_2')
    HYD = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    HYD = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYD))),axis=2)
    n = '/HYD.png'
    cv2.imwrite(savepath+n, HYD)

    i0 = paramList.index('ISLOPE')
    i1 = paramList.index('BD1400')
    i2 = paramList.index('IRR2')
    CHL = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    CHL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CHL))),axis=2)
    n = '/CHL.png'
    cv2.imwrite(savepath+n, CHL)

    i0 = paramList.index('MIN2250')
    i1 = paramList.index('BD2250')
    i2 = paramList.index('BD1900r2')
    HYS = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    HYS = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYS))),axis=2)
    n = '/HYS.png'
    cv2.imwrite(savepath+n, HYS)

    i0 = paramList.index('BD1200')
    i1 = paramList.index('BD1450')
    i2 = paramList.index('BD1900r2')
    HY2 = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    HY2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HYS))),axis=2)
    n = '/HY2.png'
    cv2.imwrite(savepath+n, HY2)
    
    i0 = paramList.index('BD1450')
    i1 = paramList.index('BD2100_3')
    i2 = paramList.index('BD1200')
    LIC = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    LIC = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,LIC))),axis=2)
    n = '/LIC.png'
    cv2.imwrite(savepath+n, LIC)
    
    i0 = paramList.index('GypTrip')
    i1 = paramList.index('BD2165')
    i2 = paramList.index('BD1900_2')
    GYP = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    GYP = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,GYP))),axis=2)
    n = '/GYP.png'
    cv2.imwrite(savepath+n, GYP)
    
    i0 = paramList.index('BDCARB')
    i1 = paramList.index('ILL')
    i2 = paramList.index('GypTrip')
    SED = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    SED = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,SED))),axis=2)
    n = '/SED.png'
    cv2.imwrite(savepath+n, SED)
    

    #-- VIS Browse PNG ------------------------------------------------------------
    i0 = paramList.index('BD530_2')
    i1 = paramList.index('BD875')
    i2 = paramList.index('BD920_2')
    FM2 = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    FM2 = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,FM2))),axis=2)
    n = '/FM2.png'
    cv2.imwrite(savepath+n, FM2)
    
    i0 = paramList.index('BD530_2')
    i1 = paramList.index('BD670')
    i2 = paramList.index('BD875')
    HEM = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    HEM = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,HEM))),axis=2)
    n = '/HEM.png'
    cv2.imwrite(savepath+n, HEM)
    
    # i0 = paramList.index('RPEAK1')
    # i1 = paramList.index('ELMSUL')
    # i2 = paramList.index('BDI1000VIS')
    # CPL = buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    CPL = p.CPL()
    CPL = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,CPL))),axis=2)
    n = '/CPL.png'
    cv2.imwrite(savepath+n, CPL)
    
    # i0 = paramList.index('ELMSUL')
    # i1 = paramList.index('ELMSUL')
    # i2 = paramList.index('ELMSUL')
    # SUL = buildSummary(vParams[:,:,i0], vParams[:,:,i1], vParams[:,:,i2])
    SUL = p.SUL()
    SUL = np.flip(u.browse2bit(u.stretchNBands(u,SUL)),axis=2)
    n = '/SUL.png'
    cv2.imwrite(savepath+n, SUL)

    i0 = paramList.index('R637')
    i1 = paramList.index('R550')
    i2 = paramList.index('R463')
    TRU = u.buildSummary(jParams[:,:,i0], jParams[:,:,i1], jParams[:,:,i2])
    TRU = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,TRU))),axis=2)
    # TRU = np.flip(browse2bit(TRU),axis=2)
    n = '/TRU.png'
    cv2.imwrite(savepath+n, TRU)


#%% MNF block
'''
calculate minimum noise fraction transform images and save as PNG and ENVI .img
'''
#SWIR
s_mnf10 = p.SWIR_MNF()
bs = [4,3,2] #because cv2 writes BGR for some reason
sName = '/SWIR_MNF_234.png'
cv2.imwrite(savepath+sName,s_mnf10[:,:,bs])
sName = '/SWIR_MNF_234_8bit.png'
cv2.imwrite(savepath+sName,u.browse2bit(s_mnf10[:,:,bs]))

bandList = ['1','2','3','4','5','6','7','8','9','10']
sMeta = p.s_.metadata.copy()
sMeta['wavelength'] = bandList
sMeta['wavelength units'] = 'MNF Band'
sMeta['default bands'] = ['1', '2', '3']
envi.save_image(savepath+'/SWIR_MNF.hdr', s_mnf10, metadata=sMeta,dtype=np.float32)


#Vis
v_mnf10 = p.VIS_MNF()
bs = [4,3,2]
vName = '/VNIR_MNF_234.png'
cv2.imwrite(savepath+vName,v_mnf10[:,:,bs])
vName = '/VNIR_MNF_234_8bit.png'
cv2.imwrite(savepath+vName,u.browse2bit(v_mnf10[:,:,bs]))

bandList = ['1','2','3','4','5','6','7','8','9','10']
vMeta = p.v_.metadata.copy()
vMeta['wavelength'] = bandList
vMeta['wavelength units'] = 'MNF Band'
vMeta['default bands'] = ['1', '2', '3']
envi.save_image(savepath+'/VIS_MNF.hdr', v_mnf10, metadata=vMeta,dtype=np.float32)

#%% browse block
'''
generate one-off browse product images (as PNG files)
'''
savepath = '/Users/phillms1/Documents/Work/RAVEN/RAVEN_parameters/hyspex_parameters/BrowseProducts/'
if os.path.isdir(savepath) is False:
    os.mkdir(savepath)

# calculate and save browse products as .png
bp = p.MAF()
n = 'MAF'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.FM2()
n = 'FM2'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.FAL()
n = 'FAL'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.PAL()
n = 'PAL'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,bp))),axis=2)
cv2.imwrite(savepath+n,img)
del(img)

bp = p.PFM()
n = 'PFM'+'.png'
img = np.flip(u.browse2bit(u.stretchNBands(u,u.cropNZeros(u,bp))),axis=2)
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


