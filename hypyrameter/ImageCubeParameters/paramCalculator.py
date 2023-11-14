#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 13:33:42 2022

@author: phillms1
"""
import numpy as np
import timeit
import cv2
from matplotlib import pyplot as plt 
import spectral.io.envi as envi
import multiprocessing as mp
from spectral import calc_stats,noise_from_diffs,mnf
from tqdm import tqdm
import pandas as pd
from spectral.io.envi import EnviException as EnviException
import tkinter as tk
from tkinter import filedialog
import os

import hypyrameter.ImageCubeParameters.param_utils as u
from hypyrameter.ImageCubeParameters.iovf_generic import iovf as iovf


class paramCalculator:
    """this class handles an input hyperspectral image (an envi .img) and returns a spectral parameter image cube (as an envi .img)

    Returns:
        object: class object to calculate spectral parameters
    """
    '''
    this class handles an input hyperspectral image (an envi .img) and returns 
    browse product summary images
    
    I/O
    file: /path/to/image.hdr
        this is a .hdr file, not the .img
        
    '''
    
    def __init__(self, crop=None, bbl=[None], flip = False, transpose = False, denoise=False, preview=False):
        """initiation of paramCalculator class

        Args:
            crop (list, optional): list with starting and ending row and column values for crop region, like [r0, r1, c0, c1]. Defaults to None.
            bbl (list, optional): bad bands list, integer index values. Defaults to [None].
            flip (bool, optional): option to flip the image. Defaults to False.
            transpose (bool, optional): option to transpose rows, columns and bands. Defaults to False.
            denoise (bool, optional): option to denoise the image. Defaults to False.
        """
        
        tic = timeit.default_timer()
        # Create a tkinter root window (it won't be shown)
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Use the file dialog to select a '.hdr' file
        print("Select input hyperspectral cube")
        self.file = filedialog.askopenfilename(filetypes=[("Select input hyperspectral cube", "*.hdr")])

        # Check if the user selected a file
        if hasattr(self, 'file'):
            # Check the file extension
            _, file_extension = os.path.splitext(self.file)
            if file_extension.lower() == ".hdr":
                print("Selected file:", self.file)
            else:
                print("Please select a '.hdr' file.")
        else:
            print("No file selected")
            pass

        print(f'loading {self.file} using spectral')
        self.f_ = envi.open(self.file)
        print('\tobjects loaded\n')
        
        # Use the directory dialog to select an output directory
        print("Select output directory")
        self.outdir = filedialog.askdirectory()

        # Check if the user selected a directory
        if hasattr(self, 'outdir'):
            print("Selected output directory:", self.outdir)
        else:
            print("No output directory selected")
        
        # load wave tables and data
        self.f_bands = [float(b) for b in self.f_.metadata['wavelength']]

        if 'default bands' in self.f_.metadata:
            self.f_preview_bands = [int(float(b)) for b in self.f_.metadata['default bands']]
        else:  
            self.f_preview_bands = [39, 23, 4] #if no default bands are set, grab reasonable bands for preview

        # loading data
        print('loading data')
        self.f = np.array(self.f_.load())
        # flip and transpose as necessary
        if flip:
            self.f = np.flip(self.f, axis=0)
        if transpose:
            self.f = np.transpose(self.f, (1,0,2))

        print('\tdata loaded')
        if crop is not None:
            r0,r1,c0,c1 = crop
            self.f = self.f[r0:r1,c0:c1,:]

        if bbl[0] is not None:
            for i in bbl:
                self.f[:,:,i] = np.nan

        if denoise:
            print('denoising cube')
            self.denoiser()

        self.cube = self.f
        # remove funky values
        self.cube = np.where(self.cube>1, np.nan, self.cube)
        self.cube = np.where(self.cube<-1, np.nan, self.cube)
        self.wvt = self.f_bands

        self.validParams = self.determineValidParams()
        
        toc = timeit.default_timer()-tic
        print(f'{np.round(toc/60,2)} minutes to load data')

        if preview is True: 
            self.previewData()

    # -------------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------------
    def denoiser(self):
        dn = iovf(self.f, '')
        dn.run()
        self.f = dn.flt
        file_name = self.file.split('/')[-1].split('.')[0]
        denoised_file_name = self.outdir+'/'+file_name+'_denoised.hdr'
        meta = self.f_.metadata.copy()
        try:
            envi.save_image(denoised_file_name, self.f,
                            metadata=meta, dtype=np.float32)
        except EnviException as error:
            print(error)
            choice = input('file exists, would you like to overwite?\n\ty or n\n')
            choice = choice.lower()
            if choice == 'y':
                envi.save_image(denoised_file_name, self.f,
                                metadata=meta, dtype=np.float32, force=True)
            else:
                pass

    def previewData(self):
        # preview data
        plt.title('Image Preview')
        if hasattr(self, 'f_preview_bands'):
            preview = self.f[:,:,self.f_preview_bands]
        else:
            choice = input('type 3 bands to display as a list\n\tlike this [55, 30, 10]')
            self.f_preview_bands = choice
            preview = self.f[:,:,self.f_preview_bands]
        preview = [(r-np.nanmin(r))/(np.nanmax(r)-np.nanmin(r)) for r in preview]
        plt.imshow(preview)
        plt.show()

    def determineValidParams(self):
        # get wavelength bounds of data
        b_min = np.min(self.f_bands)
        b_max = np.max(self.f_bands)

        # get wavelength bounds for each parameter...
        '''We need a better way to only grab the parameters that are valid for the data cube'''
        paramDict = paramCalculator.__dict__.copy()
        paramList = list(paramDict)[6:-6]
        print(f'all parameters:\n\t{paramList}')
        w_bounds = [paramDict[param](self,check=True) for param in paramList]

        # check against parameter values
        validParams = []
        for bounds, param in zip(w_bounds,paramList):
            # determine if min bound is valid within a tolerance level
            tol = 5
            if bounds[0] > (b_min-tol):
                min_valid = True
            else:
                min_valid = False
            # determine if max bound is valid
            if bounds[1] < (b_max + tol):
                max_valid = True
            else:
                max_valid = False
            
            # if one is invalid then reject the parameter, otherwise add it to the list
            if min_valid and max_valid:
                validParams.append(param)
        
        return validParams
    
    # should add denoise routine here once it is fixed
    # -------------------------------------------------------------------------
            
    # -------------------------------------------------------------------------
    # parameter library
    # -------------------------------------------------------------------------

    # Reflectance (R) parameters
    def R463(self, check = False):
        if check:
            img = (463,463)
        elif not check:
            img = u.getBand(self.f,self.f_bands,463)
        return img
        
    def R550(self, check = False):
        if check:
            img = (550,550)
        elif not check:
            img = u.getBand(self.f,self.f_bands,550)
        return img
        
    def R637(self, check = False):
        if check:
            img = (637,637)
        elif not check:
            img = u.getBand(self.f,self.f_bands,637)
        return img

    def R1080(self, check = False):
        if check:
            img = (1080,1080)
        elif not check:
            img = u.getBand(self.f,self.f_bands,1080)
        return img
    
    def R1506(self, check = False):
        if check:
            img = (1506,1506)
        elif not check:
            img = u.getBand(self.f,self.f_bands,1506)
        return img
    
    def R2529(self, check = False):
        if check:
            img = (2529,2529)
        elif not check:
            img = u.getBand(self.f,self.f_bands,2529)
        return img

    # Index parameters
    def HCPINDEX2(self, check = False):
        if check:
            img = (1690, 2530)
        elif not check:
            # extract data from image cube
            R2120 = u.getBand(self.cube, self.wvt,2120)
            R2140 = u.getBand(self.cube, self.wvt,2140)
            R2230 = u.getBand(self.cube, self.wvt,2230)
            R2250 = u.getBand(self.cube, self.wvt,2250)
            R2430 = u.getBand(self.cube, self.wvt,2430)
            R2460 = u.getBand(self.cube, self.wvt,2460)
            R2530 = u.getBand(self.cube, self.wvt,2530)
            R1690 = u.getBand(self.cube, self.wvt,1690)
        
            W2120 = u.getClosestWavelength(2120,self.wvt)
            W2140 = u.getClosestWavelength(2140,self.wvt)
            W2230 = u.getClosestWavelength(2230,self.wvt)
            W2250 = u.getClosestWavelength(2250,self.wvt)
            W2430 = u.getClosestWavelength(2430,self.wvt)
            W2460 = u.getClosestWavelength(2460,self.wvt)
            W2530 = u.getClosestWavelength(2530,self.wvt)
            W1690 = u.getClosestWavelength(1690,self.wvt)
        
        
            # compute the corrected reflectance interpolating 
            slope = (R2530 - R1690) / (W2530 - W1690)      
            intercept = R2530 - slope * W2530
        
            # weighted sum of relative differences
            Rc2120 = slope*W2120 + intercept
            Rc2140 = slope*W2140 + intercept
            Rc2230 = slope*W2230 + intercept
            Rc2250 = slope*W2250 + intercept
            Rc2430 = slope*W2430 + intercept
            Rc2460 = slope*W2460 + intercept
        
            img=((1-(R2120/Rc2120))*0.1) + ((1-(R2140/Rc2140))*0.1) + ((1-(R2230/Rc2230))*0.15) + ((1-(R2250/Rc2250))*0.3) + ((1-(R2430/Rc2430))*0.2) + ((1-(R2460/Rc2460))*0.15)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def LCPINDEX2(self, check = False):
        if check:
            img = (1690, 1870)
        elif not check:
            # extract data from image cube
            R1690 = u.getBand(self.cube, self.wvt, 1690)
            R1750 = u.getBand(self.cube, self.wvt, 1750)
            R1810 = u.getBand(self.cube, self.wvt, 1810)
            R1870 = u.getBand(self.cube, self.wvt, 1870)
            R1560 = u.getBand(self.cube, self.wvt, 1560)
            R2450 = u.getBand(self.cube, self.wvt, 2450)
        
            W1690 = u.getClosestWavelength(1690,self.wvt)
            W1750 = u.getClosestWavelength(1750,self.wvt)
            W1810 = u.getClosestWavelength(1810,self.wvt)
            W1870 = u.getClosestWavelength(1870,self.wvt)
            W1560 = u.getClosestWavelength(1560,self.wvt)
            W2450 = u.getClosestWavelength(2450,self.wvt)
        
            # compute the corrected reflectance interpolating 
            slope = (R2450 - R1560)/(W2450 - W1560)
            intercept = R2450 - slope * W2450
        
            # weighted sum of relative differences
            Rc1690 = slope*W1690 + intercept
            Rc1750 = slope*W1750 + intercept
            Rc1810 = slope*W1810 + intercept
            Rc1870 = slope*W1870 + intercept
        
            img=((1-(R1690/Rc1690))*0.2) + ((1-(R1750/Rc1750))*0.2) + ((1-(R1810/Rc1810))*0.3) + ((1-(R1870/Rc1870))*0.3)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def OLINDEX3(self, check = False):
        if check:
            img = (1210, 1862)
        elif not check:
            # extract data from image cube
            R1210 = u.getBand(self.cube, self.wvt,1210)
            R1250 = u.getBand(self.cube, self.wvt,1250)
            R1263 = u.getBand(self.cube, self.wvt,1263)
            R1276 = u.getBand(self.cube, self.wvt,1276)
            R1330 = u.getBand(self.cube, self.wvt,1330)
            R1750 = u.getBand(self.cube, self.wvt,1750)
            R1862 = u.getBand(self.cube, self.wvt,1862)
        
            # find closest Hyspex wavelength
            W1210 = u.getClosestWavelength(1210,self.wvt)
            W1250 = u.getClosestWavelength(1250,self.wvt)
            W1263 = u.getClosestWavelength(1263,self.wvt)
            W1276 = u.getClosestWavelength(1276,self.wvt)
            W1330 = u.getClosestWavelength(1330,self.wvt)
            W1750 = u.getClosestWavelength(1750,self.wvt)
            W1862 = u.getClosestWavelength(1862,self.wvt)
        
            # ; compute the corrected reflectance interpolating 
            slope = (R1862 - R1750)/(W1862 - W1750)   #;slope = ( R2120 - R1690 ) / ( W2120 - W1690 )
            intercept = R1862 - slope*W1862               #;intercept = R2120 - slope * W2120
        
            Rc1210 = slope * W1210 + intercept
            Rc1250 = slope * W1250 + intercept
            Rc1263 = slope * W1263 + intercept
            Rc1276 = slope * W1276 + intercept
            Rc1330 = slope * W1330 + intercept
        
            img = (((Rc1210-R1210)/(abs(Rc1210)))*0.1) + (((Rc1250-R1250)/(abs(Rc1250)))*0.1) + (((Rc1263-R1263)/(abs(Rc1263)))*0.2) + (((Rc1276-R1276)/(abs(Rc1276)))*0.2) + (((Rc1330-R1330)/(abs(Rc1330)))*0.4)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def SINDEX2(self, check = False):
        if check:
            img = (2120, 2400)
        elif not check:
            img = u.getBandDepthInvert(self.cube,self.wvt,2120, 2290, 2400,mw=7,hw=3)
        return img
    
    def GINDEX(self, check = False):
        if check:
            img = (1420, 1820)
        elif not check:
            t1 = u.getBandArea(self.cube,self.wvt,1420,1463)
            t2 = u.getBandArea(self.cube,self.wvt,1463,1515)
            t3 = u.getBandArea(self.cube,self.wvt,1515,1576)
            b1 = t1*t2*t3
            b2 = self.BD1750()
            img = b1+b2
        return img
    
    # -----------------------------------------------------------------------------------------------
    # Band Depth (BD) Parameters
    def BD530_2(self, check = False):
        if check:
            img = (440, 614)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 440, 530, 614)
        return img
    
    def BD670(self, check = False):
        if check:
            img = (620, 745)
        elif not check:
            # this is a custom parameter
            img = u.getBandDepth(self.cube, self.wvt, 620, 670, 745)
        return img
    
    def BD875(self, check = False):
        if check:
            img = (747, 980)
        elif not check:
            # this is a custom parameter
            img = u.getBandDepth(self.cube, self.wvt, 747, 875, 980)
        return img
    
    def BD905(self, check = False):
        if check:
            img = (750, 1300)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,750,905,1300)
        return img
    
    def BD920_2(self, check = False):
        if check:
            img = (807, 984)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,807,920,984)
        return img
    
    def BD1200(self, check = False):
        if check:
            img = (1115, 1260)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 1115, 1200, 1260)
        return img
    
    def BD1300(self, check = False):
        if check:
            img = (1080, 1750)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 1260, 1320, 1750, mw=15)
        return img
    
    def BD1400(self, check = False):
        if check:
            img = (1330, 1467)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 1330, 1395, 1467, mw=3)
        return img
    
    def BD1450(self, check = False):
        if check:
            img = (1340, 1535)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 1340, 1450, 1535, mw=3)
        return img
    
    def BD1750(self, check = False):
        if check:
            img = (1688, 1820)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 1688, 1750, 1820)
        return img
    
    def BD1900_2(self, check = False):
        if check:
            img = (1850, 2067)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,1850, 1930, 2067)
        return img
    
    def BD1900r2(self, check = False):
        if check:
            img = (1908, 2132)
        elif not check:
            # img = u.getBandArea(self.cube, self.wvt, 1800, 2000)
            # extract individual channels, replacing CRISM_NANs with IEEE_NaNs
            R1908 = u.getBand(self.cube,self.wvt,1908, kwidth = 1) 
            R1914 = u.getBand(self.cube,self.wvt,1914, kwidth = 1) 
            R1921 = u.getBand(self.cube,self.wvt,1921, kwidth = 1) 
            R1928 = u.getBand(self.cube,self.wvt,1928, kwidth = 1) 
            R1934 = u.getBand(self.cube,self.wvt,1934, kwidth = 1) 
            R1941 = u.getBand(self.cube,self.wvt,1941, kwidth = 1) 
            R1862 = u.getBand(self.cube,self.wvt,1862, kwidth = 1) 
            R1869 = u.getBand(self.cube,self.wvt,1869, kwidth = 1) 
            R1875 = u.getBand(self.cube,self.wvt,1875, kwidth = 1) 
            R2112 = u.getBand(self.cube,self.wvt,2112, kwidth = 1) 
            R2120 = u.getBand(self.cube,self.wvt,2120, kwidth = 1) 
            R2126 = u.getBand(self.cube,self.wvt,2126, kwidth = 1) 
            
            R1815 = u.getBand(self.cube, self.wvt, 1815);
            R2132 = u.getBand(self.cube, self.wvt, 2132); 
            
            # retrieve the CRISM wavelengths nearest the requested values
            W1908 = u.getClosestWavelength(1908, self.wvt)
            W1914 = u.getClosestWavelength(1914, self.wvt)
            W1921 = u.getClosestWavelength(1921, self.wvt)
            W1928 = u.getClosestWavelength(1928, self.wvt)
            W1934 = u.getClosestWavelength(1934, self.wvt)
            W1941 = u.getClosestWavelength(1941, self.wvt)
            W1862 = u.getClosestWavelength(1862, self.wvt)#
            W1869 = u.getClosestWavelength(1869, self.wvt)
            W1875 = u.getClosestWavelength(1875, self.wvt)
            W2112 = u.getClosestWavelength(2112, self.wvt)
            W2120 = u.getClosestWavelength(2120, self.wvt)
            W2126 = u.getClosestWavelength(2126, self.wvt)
            W1815 = u.getClosestWavelength(1815, self.wvt)#  
            W2132 = u.getClosestWavelength(2132, self.wvt) 
            
            # compute the interpolated continuum values at selected wavelengths between 1815 and 2530
            slope = (R2132 - R1815)/(W2132 - W1815)
            CR1908 = R1815 + slope *(W1908 - W1815)
            CR1914 = R1815 + slope *(W1914 - W1815)
            CR1921 = R1815 + slope *(W1921 - W1815) 
            CR1928 = R1815 + slope *(W1928 - W1815)
            CR1934 = R1815 + slope *(W1934 - W1815)
            CR1941 = R1815 + slope *(W1941 - W1815) 
            
            CR1862 = R1815 + slope*(W1862 - W1815)
            CR1869 = R1815 + slope*(W1869 - W1815)
            CR1875 = R1815 + slope*(W1875 - W1815)    
            CR2112 = R1815 + slope*(W2112 - W1815)
            CR2120 = R1815 + slope*(W2120 - W1815)
            CR2126 = R1815 + slope*(W2126 - W1815)
            img= 1.0-((R1908/CR1908+R1914/CR1914+R1921/CR1921+R1928/CR1928+R1934/CR1934+R1941/CR1941)/(R1862/CR1862+R1869/CR1869+R1875/CR1875+R2112/CR2112+R2120/CR2120+R2126/CR2126))
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img

    def BD2100_2(self, check = False):
        if check:
            img = (1930, 2250)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,1930, 2132, 2250,lw=3,hw=3)
        return img
    
    def BD2100_3(self, check = False):
        if check:
            img = (2016, 2220)
        elif not check:
            img = u.getBandDepth(self.cube, self.wvt, 2016, 2100, 2220)
        return img
    
    def BD2165(self, check = False):
        if check:
            img = (2120, 2230)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2120,2165,2230,mw=3,hw=3) #(kaolinite group)
        return img
    
    def BD2190(self, check = False):
        if check:
            img = (2120, 2250)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2120,2185,2250,mw=3,hw=3) #(Beidellite, Allophane)
        return img
    
    def BD2210_2(self, check = False):
        if check:
            img = (2165, 2290)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2165,2210,2290) #(kaolinite group)
        return img
    
    def BD2250(self, check = False):
        if check:
            img = (2120, 2340)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2120, 2245, 2340,mw=7,hw=3) 
        return img
    
    def BD2265(self, check = False):
        if check:
            img = (2120, 2340)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2120, 2265, 2340, mw=3,hw=5) 
        return img
    
    def BD2290(self, check = False):
        if check:
            img = (2250, 2350)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2250, 2290, 2350) #(fe/mg phyllo group)
        return img
    
    def BD2355(self, check = False):
        if check:
            img = (2300, 2450)
        elif not check:
            img = u.getBandDepth(self.cube,self.wvt,2300, 2355, 2450) #(fe/mg phyllo group)
        return img  
      
    def BDCARB(self, check = False):
        if check:
            img = (2230, 2600)
        elif not check:
            # alternative formulation
            # b1 = u.getBandDepth(self.cube,self.wvt,2220,2375)
            # b2 = u.getBandDepth(self.cube,self.wvt,2400,2500)
            # img = b1+b2

            # extract channels, replacing CRISM_NAN with IEEE NAN
            R2230 = u.getBand(self.cube, self.wvt, 2230)
            R2320 = u.getBand(self.cube, self.wvt, 2320)
            R2330 = u.getBand(self.cube, self.wvt, 2330)
            R2390 = u.getBand(self.cube, self.wvt, 2390)
            R2520 = u.getBand(self.cube, self.wvt, 2520)
            R2530 = u.getBand(self.cube, self.wvt, 2530)
            R2600 = u.getBand(self.cube, self.wvt, 2600)
        
            # identify nearest wavelengths
            WL1 = u.getClosestWavelength(2230,self.wvt)
            WC1 = (u.getClosestWavelength(2330,self.wvt)+u.getClosestWavelength(2320,self.wvt))*0.5
            WH1 = u.getClosestWavelength(2390,self.wvt)
            a =  (WC1 - WL1)/(WH1 - WL1)  # a gets multipled by the longer (higher wvln)  band
            b = 1.0-a                     # b gets multiplied by the shorter (lower wvln) band
        
            WL2 =  u.getClosestWavelength(2390,self.wvt)
            WC2 = (u.getClosestWavelength(2530,self.wvt) + u.getClosestWavelength(2520,self.wvt))*0.5
            WH2 =  u.getClosestWavelength(2600,self.wvt)
            c = (WC2 - WL2)/(WH2 - WL2)   # c gets multipled by the longer (higher wvln)  band
            d = 1.0-c                           # d gets multiplied by the shorter (lower wvln) band
        
            # compute bdcarb
            img = 1.0 - (np.sqrt((((R2320 + R2330)*0.5)/(b*R2230 + a*R2390))*(((R2520 + R2530)*0.5)/(d*R2390 + c*R2600))))
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
# -----------------------------------------------------------------------------------------------
# Band Area (BA) parameters
    def BA1200(self, check = False):
        if check:
            img = (1115, 1260)
        elif not check:
            img = u.getBandArea(self.cube, self.wvt, 1115, 1260)
        return img
    
    def BA1450(self, check = False):
        if check:
            img = (1340, 1535)
        elif not check:
            img = u.getBandArea(self.cube, self.wvt, 1340, 1535)
        return img
    
    def BA1900(self, check = False):
        if check:
            img = (1850, 2067)
        elif not check:
            img = u.getBandArea(self.cube, self.wvt, 1850, 2067)
        return img
    
# -----------------------------------------------------------------------------------------------
# Depth (D) parameters
    def D460(self, check = False):
        if check:
            img = (420, 520)
        elif not check:
            img = u.getBandDepthInvert(self.cube, self.wvt, 420, 460, 520)
        return img
    
    def D700(self, check = False):
        if check:
            img = (630, 830)
        elif not check:
            # a custom parameter for chlorophyll
            # extract individual channels
            R630 = u.getBand(self.cube,self.wvt,630)
            R740 = u.getBand(self.cube,self.wvt,740)
            R760 = u.getBand(self.cube,self.wvt,760)
            R770 = u.getBand(self.cube,self.wvt,770)
            R690 = u.getBand(self.cube,self.wvt,690,kwidth=3)
            R710 = u.getBand(self.cube,self.wvt,710,kwidth=3)
            R720 = u.getBand(self.cube,self.wvt,720,kwidth=3)
            R830 = u.getBand(self.cube,self.wvt,830)
            
            # get closestgetClosestWavelengthngth
            W630 = u.getClosestWavelength(630,self.wvt)
            W740 = u.getClosestWavelength(740,self.wvt)
            W760 = u.getClosestWavelength(760,self.wvt)
            W770 = u.getClosestWavelength(770,self.wvt)
            W690 = u.getClosestWavelength(690,self.wvt)
            W710 = u.getClosestWavelength(710,self.wvt)
            W720 = u.getClosestWavelength(720,self.wvt)
            W830 = u.getClosestWavelength(830,self.wvt)
            
            # compute the interpolated continuum values at selected wavelengths between 630 and 830
            slope= (R830 - R630)/(W830 - W630)
            CR740 = R630 + slope*(W740 - W630)
            CR760 = R630 + slope*(W760 - W630)
            CR770 = R630 + slope*(W770 - W630)
            CR690 = R630 + slope*(W690 - W630)
            CR710 = R630 + slope*(W710 - W630)
            CR720 = R630 + slope*(W720 - W630)
        
            # compute d700 with IEEE NaN values in place of CRISM NaN
            img = 1 - (((R690/CR690) + (R710/CR710) + (R720/CR720))/((R740/CR740) + (R760/CR760) + (R770/CR770)))
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin) 
        return img
    
    def D2200(self, check = False):
        if check:
            img = (1815, 2430)
        elif not check:
            # extract individual channels
            R1815 = u.getBand(self.cube, self.wvt,1815, kwidth=5)
            R2165 = u.getBand(self.cube, self.wvt,2165)
            R2210 = u.getBand(self.cube, self.wvt,2210, kwidth=5)
            R2230 = u.getBand(self.cube, self.wvt,2230, kwidth=5)
            R2430 = u.getBand(self.cube, self.wvt,2430, kwidth=5)
        
        
            # retrieve wavelengths nearest the requested values
            W1815 = u.getClosestWavelength(1815, self.wvt)
            W2165 = u.getClosestWavelength(2165, self.wvt) 
            W2210 = u.getClosestWavelength(2210, self.wvt)
            W2230 = u.getClosestWavelength(2230, self.wvt)
            W2430 = u.getClosestWavelength(2430, self.wvt)
            
            # compute the interpolated continuum values at selected wavelengths between 1815 and 2430
            slope = (R2430 - R1815)/(W2430 - W1815)
            CR2165 = R1815 + slope*(W2165 - W1815)    
            CR2210 = R1815 + slope*(W2210 - W1815)
            CR2230 = R1815 + slope*(W2230 - W1815)
        
            # compute d2200 with IEEE NaN values
            img = 1 - (((R2210/CR2210) + (R2230/CR2230))/(2*(R2165/CR2165)))
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def D2300(self, check = False):
        if check:
            img = (1815, 2530)
        elif not check:
            # extract individual channels
            R1815 = u.getBand(self.cube,self.wvt,1815)
            R2120 = u.getBand(self.cube,self.wvt,2120)
            R2170 = u.getBand(self.cube,self.wvt,2170)
            R2210 = u.getBand(self.cube,self.wvt,2210)
            R2290 = u.getBand(self.cube,self.wvt,2290,kwidth=3)
            R2320 = u.getBand(self.cube,self.wvt,2320,kwidth=3)
            R2330 = u.getBand(self.cube,self.wvt,2330,kwidth=3)
            R2530 = u.getBand(self.cube,self.wvt,2530)
            
            # retrieve wavelengths nearest the requested values
            W1815 = u.getClosestWavelength(1815,self.wvt)
            W2120 = u.getClosestWavelength(2120,self.wvt)
            W2170 = u.getClosestWavelength(2170,self.wvt)
            W2210 = u.getClosestWavelength(2210,self.wvt)
            W2290 = u.getClosestWavelength(2290,self.wvt)
            W2320 = u.getClosestWavelength(2320,self.wvt)
            W2330 = u.getClosestWavelength(2330,self.wvt)
            W2530 = u.getClosestWavelength(2530,self.wvt)
            
            # compute the interpolated continuum values at selected wavelengths between 1815 and 2530
            slope = (R2530 - R1815)/(W2530 - W1815)
            CR2120 = R1815 + slope*(W2120 - W1815)
            CR2170 = R1815 + slope*(W2170 - W1815)
            CR2210 = R1815 + slope*(W2210 - W1815)
            CR2290 = R1815 + slope*(W2290 - W1815)
            CR2320 = R1815 + slope*(W2320 - W1815)
            CR2330 = R1815 + slope*(W2330 - W1815)
        
            # compute d2300 with IEEE NaN values
            img = 1 - (((R2290/CR2290) + (R2320/CR2320) + (R2330/CR2330))/((R2120/CR2120) + (R2170/CR2170) + (R2210/CR2210)))
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin) 
        return img
    
# -----------------------------------------------------------------------------------------------
# Minimum (MIN) parameters
    def MIN2295_2480(self, check = False):
        if check:
            img = (2165, 2570)
        elif not check:
            img1 = u.getBandDepth(self.cube, self.wvt,2165,2295,2364)
            img2 = u.getBandDepth(self.cube,self.wvt,2364,2480,2570)
            img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
            img3[:,:,0] = img1
            img3[:,:,1] = img2
            img = np.min(img3,axis=2)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def MIN2250(self, check = False):
        if check:
            img = (2165, 2350)
        elif not check:
            img1 = u.getBandDepth(self.cube, self.wvt,2165, 2210, 2350)
            img2 = u.getBandDepth(self.cube,self.wvt,2165, 2265, 2350)
            img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
            img3[:,:,0] = img1
            img3[:,:,1] = img2
            img = np.min(img3,axis=2)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def MIN2345_2537(self, check = False):
        if check:
            img = (2250, 2602)
        elif not check:
            img1 = u.getBandDepth(self.cube, self.wvt,2250, 2345, 2430)
            img2 = u.getBandDepth(self.cube,self.wvt,2430, 2537, 2602)
            img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
            img3[:,:,0] = img1
            img3[:,:,1] = img2
            img = np.min(img3,axis=2)
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
# -----------------------------------------------------------------------------------------------
# All other parameters (ratios, slopes, peaks)
    def RPEAK1(self, check = False):
        if check:
            img = (442, 963)
        elif not check:
            # old, but functional, RPEAK1
            rp_wv = [442,533,600,710,740,775,800,833,860,892,925,963]
            rp_i = [self.wvt.index(u.getClosestWavelength(i,self.wvt)) for i in rp_wv]
            rp_w = [u.getClosestWavelength(i,self.wvt) for i in rp_wv]
            rp_ = self.cube[:,:,rp_i]
            x_ = np.linspace(rp_w[0],rp_w[-1],num=521)
            flatShape=(np.shape(rp_)[0]*np.shape(rp_)[1],np.shape(rp_)[2])       
            rp_l = np.zeros(flatShape[0])#[]#np.empty(flatShape[0])
            rp_r = np.zeros(flatShape[0])#[]#np.empty(flatShape[0])
            rp_flat = np.reshape(rp_,flatShape)
            is_finite_non_zero = np.logical_and(np.isfinite(rp_flat), rp_flat != 0.0)
            goodIndeces = np.where(is_finite_non_zero)
            goodIndx = np.unique(goodIndeces[0])
            poly=[]
            print('\tpreparing polynomial arguments')
            # parallel attempt
            args = [(rp_w,rp_flat[i,:],5) for i in tqdm(goodIndx)]
            print('\n\tcalculating polynomials')
            with mp.Pool(6) as pool:
                for p in pool.imap(u.getPoly,args):
                    poly.append(p)
                
            print('\treturning peak reflectance and wavelengths of peak reflectance')
            for j,i in tqdm(enumerate(goodIndx)):
                rp_l[i] = x_[list(poly[j](x_)).index(np.nanmax(poly[j](x_)))]/1000 
                rp_r[i] = np.nanmax(poly[j](x_)) 
            
            print('\tre-shaping arrays')
            shape2d = (np.shape(rp_)[0],np.shape(rp_)[1])
            rp_l=np.reshape(rp_l,shape2d)
            rp_r=np.reshape(rp_r,shape2d)
            self.rpeak_reflectance = rp_r
            img = rp_l
        return img
    
    def BDI1000VIS(self, rp_r=None, check = False):
        if check:
            img = (833, 989)
        elif not check:
            if rp_r is None:
                rp_l = self.RPEAK1()
                rp_r = self.rpeak_reflectance
                
            # multispectral version
            # bdi_wv = [833,860,892,925,951,984,989] 
            # vi = [self.wvt.index(u.getClosestWavelength(i,self.wvt)) for i in bdi_wv]
            # wv_um = [u.getClosestWavelength(i,self.wvt)/1000 for i in bdi_wv]
            # wv_ = np.linspace(wv_um[0],wv_um[-1],num=201)
            
            vi0 = self.wvt.index(u.getClosestWavelength(833,self.wvt))
            vi1 = self.wvt.index(u.getClosestWavelength(989,self.wvt))
            n = vi1-vi0 + 1
            vi = np.linspace(vi0,vi1,n,dtype=int)
            wv_um = [self.wvt[i]/1000 for i in vi]
            bdi1000_cube = self.cube[:,:,vi]
            bdi_norm = np.empty(np.shape(bdi1000_cube))
            print('\tnormalizing input data')
            for b in tqdm(range(len(vi))):
                bdi_norm[:,:,b] = bdi1000_cube[:,:,b]/rp_r
            
            flatShape=(np.shape(bdi_norm)[0]*np.shape(bdi_norm)[1],np.shape(bdi_norm)[2])       
            bdi1000vis_value = np.zeros(flatShape[0])
            bdi_norm_flat = np.reshape(bdi_norm,flatShape)
            args = []
            bdi1000vis_value = []
            print('\tpreparing polynomial arguments')
            for i in tqdm(range(flatShape[0])):
                spec_vec = bdi_norm_flat[i,:]
                keepIndx = np.where(~np.isnan(spec_vec))[0]
                wv_um_ = [wv_um[q] for q in keepIndx]
                spec_vec_ = [spec_vec[q] for q in keepIndx]
                if not spec_vec_: 
                    spec_vec_ = np.linspace(0, len(wv_um), len(wv_um))
                    wv_um_ = wv_um
                args.append((wv_um_,spec_vec_,4))
            print('\treturning integrated polynomial values')
            with mp.Pool(6) as pool:
                for integ in pool.imap(u.getPolyInt,args):
                        bdi1000vis_value.append(integ)
            
            print('\treshaping array')
            bdi1000vis_value = np.reshape(bdi1000vis_value,(np.shape(bdi_norm)[0],np.shape(bdi_norm)[1]))
            img = bdi1000vis_value

        return img
    
    def ISLOPE(self, check = False):
        if check:
            img = (1815, 2530)
        elif not check:
            # extract individual bands
            R1815 = u.getBand(self.cube, self.wvt, 1815)
            R2530 = u.getBand(self.cube, self.wvt, 2530)
        
            W1815 = u.getClosestWavelength(1815,self.wvt)
            W2530 = u.getClosestWavelength(2530,self.wvt)
        
            # want in units of reflectance / um
            img = 1000.0 * ( R1815 - R2530 )/ ( W2530 - W1815 )
            nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
            img = np.where(img>-np.inf,img,nmin)
        return img
    
    def IRR2(self, check = False):
        if check:
            img = (2210, 2530)
        elif not check:
            img = u.getBandRatio(self.cube,self.wvt,2530,2210)
        return img
    
    # -------------------------------------------------------------------------
    # Run Parameter calculations
    # -------------------------------------------------------------------------
    def calculateParams(self):
        tic = timeit.default_timer()
        # loop through paramList, add result to a tuple. keep track of valid parameters
        paramDict = paramCalculator.__dict__.copy()
        intermediate_list = []
        for param in self.validParams:
            print(f'calculating: {param}')
            if param == 'BDI1000VIS' and 'RPEAK1' in self.validParams:
                intermediate_list.append(paramDict[param](self, rp_r=self.rpeak_reflectance))
            else:
                intermediate_list.append(paramDict[param](self))

        p_tuple = tuple(intermediate_list)

        img = np.dstack(p_tuple)
        toc = timeit.default_timer()-tic
        print(f'calculation took {round(toc/60,2)} minutes')
        return img
    
    def calculateBrowse(self, stype = 'linear', perc = 2, factor = 2.5):
        bf = pd.read_excel('hypyrameter/ImageCubeParameters/browseDefinitions.xlsx')
        # get valid browse products
        # Filter the DataFrame of browse products based on valid parameters
        filtered_bf = bf[bf['Param1'].isin(self.validParams) & bf['Param2'].isin(self.validParams) & bf['Param3'].isin(self.validParams)]
        # Get the list of BrowseProducts where all three parameters appear
        self.validBrowseProducts = filtered_bf['BrowseProduct'].tolist()
        print(f'valid browse products:\n{self.validBrowseProducts}')
        # calculate valid browse products
        for bp in self.validBrowseProducts:
            # Retrieve the parameters for the current browse product
            parameters = bf.loc[bf['BrowseProduct'] == bp, ['Param1', 'Param2', 'Param3']].values[0]
            i0 = self.validParams.index(parameters[0])
            i1 = self.validParams.index(parameters[1])
            i2 = self.validParams.index(parameters[2])
            browseProduct = u.buildSummary(self.params[:,:,i0], self.params[:,:,i1], self.params[:,:,i2])
            browseProduct = np.flip(u.browse2bit(u.stretchNBands(u.cropNZeros(browseProduct),stype=stype, perc=perc, factor=factor)),axis=2)
            n = '/' + bp + '.png'
            cv2.imwrite(self.outdir+n, browseProduct)
    
    def saveParamCube(self):
        file_name = self.file.split('/')[-1].split('.')[0]
        params_file_name = self.outdir+'/'+file_name+'_params.hdr'
        meta = self.f_.metadata.copy()
        meta['wavelength'] = self.validParams
        meta['band names'] = self.validParams
        meta['wavelength units'] = 'parameters'
        meta['default bands'] = ['R637', 'R550', 'R463']
        try:
            envi.save_image(params_file_name, self.params,
                            metadata=meta, dtype=np.float32)
        except EnviException as error:
            print(error)
            choice = input('file exists, would you like to overwite?\n\ty or n\n')
            choice = choice.lower()
            if choice == 'y':
                envi.save_image(params_file_name, self.params,
                                metadata=meta, dtype=np.float32, force=True)
            else:
                pass
    
    def run(self):
        print(f'calculating valid parameters\n{self.validParams}')
        self.params = self.calculateParams()
        self.saveParamCube()
        print(f'calculating valid browse products\n')
        self.browseProducts = self.calculateBrowse()
        
    
    