#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:59:05 2022

@author: phillms1
"""

import numpy as np
import hypyrameter.PointSpectraParameters.utils as u
import pandas as pd
from hypyrameter.PointSpectraParameters.readSed import getSedFiles

class oreXpressParamCalculator:
    '''
    this class handles an input .sed or .csv files containing wavelength and reflectance data and returns browse product 
    summary values. Alternatively, you can supply a pandas data frame where the rows are reflectance data and the columns are
    spectra. The first column of the data frame must contain the wavelengths for the spectra. Either a data path or a data frame 
    should be provided, not both.
    '''
    
    def __init__(self, data_path = False, df = False):
        
        if data_path:
            ext = data_path.split('.')[-1]
            if ext == 'csv':
                df = pd.read_csv(data_path)
            elif ext == '.sed':
                df = getSedFiles(data_path)
            else:
                raise ValueError('data_path must be a .sed or .csv file')

        self.wvt = df.iloc[:,0]
        self.spectra = df.iloc[:,1:]

        self.validParams = self.determineValidParams()

        self.specNames = list(self.spectra.columns)
    
    def determineValidParams(self):
        # get wavelength bounds of data
        b_min = np.min(self.wvt)
        b_max = np.max(self.wvt)

        # get wavelength bounds for each parameter...
        param_dict = oreXpressParamCalculator.__dict__.copy()
        param_list = list(param_dict)[4:-3]
        w_bounds = [param_dict[param](self,check=True) for param in param_list]

        # check against parameter values
        validParams = []
        for bounds, param in zip(w_bounds,param_list):
            # determine if min bound is valid within a tolerance level
            tol = 32
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
        
        print(f'valid parameters:\n\t{validParams}')
        return validParams
    
    # -------------------------------------------------------------------------
    # parameter library
    # -------------------------------------------------------------------------
    # INDEX parameters
    def HCPINDEX2(self, check=False):
        if check:
            paramValue = (1690,2530)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract data from image spectrum
            R1690 = u.getRvalue(spectrum, wvt,1690, kwidth=7)
            R2120 = u.getRvalue(spectrum, wvt,2120, kwidth=5)
            R2140 = u.getRvalue(spectrum, wvt,2140, kwidth=7)
            R2230 = u.getRvalue(spectrum, wvt,2230, kwidth=7)
            R2250 = u.getRvalue(spectrum, wvt,2250, kwidth=7)
            R2430 = u.getRvalue(spectrum, wvt,2430, kwidth=7)
            R2460 = u.getRvalue(spectrum, wvt,2460, kwidth=7)
            R2530 = u.getRvalue(spectrum, wvt,2530, kwidth=7)
        
            W1690 = u.getClosestWavelength(1690,wvt)
            W2120 = u.getClosestWavelength(2120,wvt)
            W2140 = u.getClosestWavelength(2140,wvt)
            W2230 = u.getClosestWavelength(2230,wvt)
            W2250 = u.getClosestWavelength(2250,wvt)
            W2430 = u.getClosestWavelength(2430,wvt)
            W2460 = u.getClosestWavelength(2460,wvt)
            W2530 = u.getClosestWavelength(2530,wvt)
        
        
            # compute the corrected reflectance interpolating 
            slope = (R2530 - R1690)/(W2530 - W1690)      
            intercept = R2530 - slope*W2530
        
            # weighted sum of relative differences
            Rc2120 = slope*W2120 + intercept
            Rc2140 = slope*W2140 + intercept
            Rc2230 = slope*W2230 + intercept
            Rc2250 = slope*W2250 + intercept
            Rc2430 = slope*W2430 + intercept
            Rc2460 = slope*W2460 + intercept
        
            paramValue=((1-(R2120/Rc2120))*0.1) + ((1-(R2140/Rc2140))*0.1) + ((1-(R2230/Rc2230))*0.15) + ((1-(R2250/Rc2250))*0.3) + ((1-(R2430/Rc2430))*0.2) + ((1-(R2460/Rc2460))*0.15)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def LCPINDEX2(self, check=False):
        if check:
            paramValue = (1690, 2450)
        elif not check:
            spectrum=self.spectrum
            wvt=self.wvt
            # extract data from image spectrum
            R1690 = u.getRvalue(spectrum, wvt, 1690)
            R1750 = u.getRvalue(spectrum, wvt, 1750)
            R1810 = u.getRvalue(spectrum, wvt, 1810)
            R1870 = u.getRvalue(spectrum, wvt, 1870)
            R1560 = u.getRvalue(spectrum, wvt, 1560)
            R2450 = u.getRvalue(spectrum, wvt, 2450)
        
            W1690 = u.getClosestWavelength(1690,wvt)
            W1750 = u.getClosestWavelength(1750,wvt)
            W1810 = u.getClosestWavelength(1810,wvt)
            W1870 = u.getClosestWavelength(1870,wvt)
            W1560 = u.getClosestWavelength(1560,wvt)
            W2450 = u.getClosestWavelength(2450,wvt)
        
            # compute the corrected reflectance interpolating 
            slope = (R2450 - R1560)/(W2450 - W1560)
            intercept = R2450 - slope * W2450
        
            # weighted sum of relative differences
            Rc1690 = slope*W1690 + intercept
            Rc1750 = slope*W1750 + intercept
            Rc1810 = slope*W1810 + intercept
            Rc1870 = slope*W1870 + intercept
        
            paramValue=((1-(R1690/Rc1690))*0.2) + ((1-(R1750/Rc1750))*0.2) + ((1-(R1810/Rc1810))*0.3) + ((1-(R1870/Rc1870))*0.3)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def OLINDEX3(self, check=False):
        if check:
            paramValue = (1210, 1862)
        elif not check:
            spectrum=self.spectrum
            wvt=self.wvt
            # extract data from image spectrum
            R1210 = u.getRvalue(spectrum, wvt,1210)
            R1250 = u.getRvalue(spectrum, wvt,1250)
            R1263 = u.getRvalue(spectrum, wvt,1263)
            R1276 = u.getRvalue(spectrum, wvt,1276)
            R1330 = u.getRvalue(spectrum, wvt,1330)
            R1750 = u.getRvalue(spectrum, wvt,1750)
            R1862 = u.getRvalue(spectrum, wvt,1862)
        
            # find closest Hyspex wavelength
            W1210 = u.getClosestWavelength(1210,wvt)
            W1250 = u.getClosestWavelength(1250,wvt)
            W1263 = u.getClosestWavelength(1263,wvt)
            W1276 = u.getClosestWavelength(1276,wvt)
            W1330 = u.getClosestWavelength(1330,wvt)
            W1750 = u.getClosestWavelength(1750,wvt)
            W1862 = u.getClosestWavelength(1862,wvt)
        
            # ; compute the corrected reflectance interpolating 
            slope = (R1862 - R1750)/(W1862 - W1750)   #;slope = ( R2120 - R1690 ) / ( W2120 - W1690 )
            intercept = R1862 - slope*W1862               #;intercept = R2120 - slope * W2120
        
            Rc1210 = slope * W1210 + intercept
            Rc1250 = slope * W1250 + intercept
            Rc1263 = slope * W1263 + intercept
            Rc1276 = slope * W1276 + intercept
            Rc1330 = slope * W1330 + intercept
        
            paramValue = (((Rc1210-R1210)/(abs(Rc1210)))*0.1) + (((Rc1250-R1250)/(abs(Rc1250)))*0.1) + (((Rc1263-R1263)/(abs(Rc1263)))*0.2) + (((Rc1276-R1276)/(abs(Rc1276)))*0.2) + (((Rc1330-R1330)/(abs(Rc1330)))*0.4)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def SINDEX2(self, check = False):
        if check:
            paramValue = (2120, 2400)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepthInvert(spectrum,wvt,2120, 2290, 2400,mw=7,hw=3)
        return paramValue
    
    def GINDEX(self, check = False):
        if check:
            paramValue = (1420, 1820)
        elif not check:
            t1 = u.getRvalueDepth(self.spectrum,self.wvt,1420,1450,1463)
            t2 = u.getRvalueDepth(self.spectrum,self.wvt,1463,1490,1515)
            t3 = u.getRvalueDepth(self.spectrum,self.wvt,1515,1540,1576)
            b1 = t1*t2*t3
            b2 = self.BD1750()
            paramValue = b1+b2
        return paramValue
    
    # -----------------------------------------------------------------------------------------------
    # Band Depth and shoulder parameters
    def D460(self, check = False):
        if check:
            paramValue = (420, 520)
        elif not check:
            paramValue = u.getRvalueDepthInvert(self.spectrum, self.wvt, 420, 460, 520)
        return paramValue
    
    def BD530_2(self, check = False):
        if check:
            paramValue = (440, 614)
        elif not check:
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 440, 530, 614, lw=3, mw=3, hw=3)
        return paramValue
    
    def BD670(self, check = False):
        if check:
            paramValue = (620, 745)
        elif not check:
            # this is a custom parameter
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 620, 670, 745, lw=3, mw=3, hw=3)
        return paramValue
    
    def D700(self, check = False):
        if check:
            paramValue = (630, 830)
        elif not check:
            # a custom parameter for chlorophyll
            # extract individual channels
            R630 = u.getRvalue(self.spectrum,self.wvt,630)
            R740 = u.getRvalue(self.spectrum,self.wvt,740)
            R760 = u.getRvalue(self.spectrum,self.wvt,760)
            R770 = u.getRvalue(self.spectrum,self.wvt,770)
            R690 = u.getRvalue(self.spectrum,self.wvt,690,kwidth=3)
            R710 = u.getRvalue(self.spectrum,self.wvt,710,kwidth=3)
            R720 = u.getRvalue(self.spectrum,self.wvt,720,kwidth=3)
            R830 = u.getRvalue(self.spectrum,self.wvt,830)
            
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
            paramValue = 1 - (((R690/CR690) + (R710/CR710) + (R720/CR720))/((R740/CR740) + (R760/CR760) + (R770/CR770)))
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue

    def BD875(self, check = False):
        if check:
            paramValue = (747, 980)
        elif not check:
            # this is a custom parameter
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 747, 875, 980)
        return paramValue
        
    def BD905(self, check = False):
        if check:
            paramValue = (750, 1300)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,750,905,1300)
        return paramValue  
    
    def BD920_2(self, check = False):
        if check:
            paramValue = (807, 1200)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,807,920,1200)
        return paramValue   

    def BD1200(self, check = False):
        if check:
            paramValue = (1115, 1260)
        elif not check:
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 1115, 1200, 1260)
        return paramValue
    
    def BD1300(self, check = False): 
        if check:
            paramValue = (910, 1650)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,910,1275,1650)
        return paramValue   
    
    def BD1400(self, check = False):
        if check:
            paramValue = (1330, 1467)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,1330, 1395, 1467,mw=3)
        return paramValue
    
    def BD1450(self, check = False):
        if check:
            paramValue = (1340, 1535)
        elif not check:
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 1340, 1450, 1535, mw=3)
        return paramValue
    
    def BD1750(self, check = False):
        if check:
            paramValue = (1688, 1820)
        elif not check:
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 1688, 1750, 1820)
        return paramValue
    
    def BD1900r2(self, check = False):
        if check:
            paramValue = (1815, 2132)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract individual channels, replacing CRISM_NANs with IEEE_NaNs
            R1908 = u.getRvalue(spectrum,wvt,1908, kwidth = 1) 
            R1914 = u.getRvalue(spectrum,wvt,1914, kwidth = 1) 
            R1921 = u.getRvalue(spectrum,wvt,1921, kwidth = 1) 
            R1928 = u.getRvalue(spectrum,wvt,1928, kwidth = 1) 
            R1934 = u.getRvalue(spectrum,wvt,1934, kwidth = 1) 
            R1941 = u.getRvalue(spectrum,wvt,1941, kwidth = 1) 
            R1862 = u.getRvalue(spectrum,wvt,1862, kwidth = 1) 
            R1869 = u.getRvalue(spectrum,wvt,1869, kwidth = 1) 
            R1875 = u.getRvalue(spectrum,wvt,1875, kwidth = 1) 
            R2112 = u.getRvalue(spectrum,wvt,2112, kwidth = 1) 
            R2120 = u.getRvalue(spectrum,wvt,2120, kwidth = 1) 
            R2126 = u.getRvalue(spectrum,wvt,2126, kwidth = 1) 
            
            R1815 = u.getRvalue(spectrum, wvt, 1815)
            R2132 = u.getRvalue(spectrum, wvt, 2132)
            
            # retrieve the CRISM wavelengths nearest the requested values
            W1908 = u.getClosestWavelength(1908, wvt)
            W1914 = u.getClosestWavelength(1914, wvt)
            W1921 = u.getClosestWavelength(1921, wvt)
            W1928 = u.getClosestWavelength(1928, wvt)
            W1934 = u.getClosestWavelength(1934, wvt)
            W1941 = u.getClosestWavelength(1941, wvt)
            W1862 = u.getClosestWavelength(1862, wvt)#
            W1869 = u.getClosestWavelength(1869, wvt)
            W1875 = u.getClosestWavelength(1875, wvt)
            W2112 = u.getClosestWavelength(2112, wvt)
            W2120 = u.getClosestWavelength(2120, wvt)
            W2126 = u.getClosestWavelength(2126, wvt)
            W1815 = u.getClosestWavelength(1815, wvt)#  
            W2132 = u.getClosestWavelength(2132, wvt) 
            
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
            paramValue= 1.0-((R1908/CR1908+R1914/CR1914+R1921/CR1921+R1928/CR1928+R1934/CR1934+R1941/CR1941)/(R1862/CR1862+R1869/CR1869+R1875/CR1875+R2112/CR2112+R2120/CR2120+R2126/CR2126))
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def BD1900_2(self, check = False):
        if check:
            paramValue = (1850, 2067)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,1850, 1930, 2067)
        return paramValue
    
    def BD2100_2(self, check = False):
        if check:
            paramValue = (1930, 2250)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,1930, 2132, 2250,lw=3,hw=3)
        return paramValue
    
    def BD2100_3(self, check = False):
        if check:
            paramValue = (2016, 2220)
        elif not check:
            paramValue = u.getRvalueDepth(self.spectrum, self.wvt, 2016, 2100, 2220)
        return paramValue
    
    def BD2165(self, check = False):
        if check:
            paramValue = (2120, 2230)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,2120,2165,2230,mw=3,hw=3) #(kaolinite group)
        return paramValue
    
    def BD2190(self, check = False):
        if check:
            paramValue = (2120, 2250)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,2120,2185,2250,mw=3,hw=3) #(Beidellite, Allophane)
        return paramValue
    
    def D2200(self, check = False):
        if check:
            paramValue = (1815, 2430)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract individual channels
            R1815 = u.getRvalue(spectrum, wvt,1815, kwidth=7)
            R2165 = u.getRvalue(spectrum, wvt,2165)
            R2210 = u.getRvalue(spectrum, wvt,2210, kwidth=7)
            R2230 = u.getRvalue(spectrum, wvt,2230, kwidth=7)
            R2430 = u.getRvalue(spectrum, wvt,2430, kwidth=7)
        
        
            # retrieve the CRISM wavelengths nearest the requested values
            W1815 = u.getClosestWavelength(1815, wvt)
            W2165 = u.getClosestWavelength(2165, wvt) 
            W2210 = u.getClosestWavelength(2210, wvt)
            W2230 = u.getClosestWavelength(2230, wvt)
            W2430 = u.getClosestWavelength(2430, wvt)
            
            slope = (R2430 - R1815)/(W2430 - W1815)
        
            CR2165 = R1815 + slope*(W2165 - W1815)    
            CR2210 = R1815 + slope*(W2210 - W1815)
            CR2230 = R1815 + slope*(W2230 - W1815)
        
            # compute d2200 
            paramValue = 1 - (((R2210/CR2210) + (R2230/CR2230))/(2*(R2165/CR2165)))
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def BD2210_2(self, check = False):
        if check:
            paramValue = (2165, 2290)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,2165,2210,2290) #(kaolinite group)
        return paramValue
    
    def BD2250(self, check = False):
        if check:
            paramValue = (2120, 2340)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt, 2120, 2245, 2340, mw=7,hw=3) 
        return paramValue
    
    def BD2265(self, check = False):
        if check:
            paramValue = (2120, 2340)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt, 2120, 2265, 2340, mw=3,hw=5) 
        return paramValue
    
    def BD2290(self, check = False):
        if check:
            paramValue = (2250, 2350)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,2250, 2290, 2350) #(fe/mg phyllo group)
        return paramValue
    
    def D2300(self, check = False):
        if check:
            paramValue = (1815, 2530)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract individual channels
            R1815 = u.getRvalue(spectrum,wvt,1815)
            R2120 = u.getRvalue(spectrum,wvt,2120)
            R2170 = u.getRvalue(spectrum,wvt,2170)
            R2210 = u.getRvalue(spectrum,wvt,2210)
            R2290 = u.getRvalue(spectrum,wvt,2290,kwidth=3)
            R2320 = u.getRvalue(spectrum,wvt,2320,kwidth=3)
            R2330 = u.getRvalue(spectrum,wvt,2330,kwidth=3)
            R2530 = u.getRvalue(spectrum,wvt,2530)
            
            # get closestgetClosestWavelengthngth
            W1815 = u.getClosestWavelength(1815,wvt)
            W2120 = u.getClosestWavelength(2120,wvt)
            W2170 = u.getClosestWavelength(2170,wvt)
            W2210 = u.getClosestWavelength(2210,wvt)
            W2290 = u.getClosestWavelength(2290,wvt)
            W2320 = u.getClosestWavelength(2320,wvt)
            W2330 = u.getClosestWavelength(2330,wvt)
            W2530 = u.getClosestWavelength(2530,wvt)
            
            # compute the interpolated continuum values at selected wavelengths between 1815 and 2530
            slope = (R2530 - R1815)/(W2530 - W1815)
            CR2120 = R1815 + slope*(W2120 - W1815)
            CR2170 = R1815 + slope*(W2170 - W1815)
            CR2210 = R1815 + slope*(W2210 - W1815)
            CR2290 = R1815 + slope*(W2290 - W1815)
            CR2320 = R1815 + slope*(W2320 - W1815)
            CR2330 = R1815 + slope*(W2330 - W1815)
        
            # compute d2300
            paramValue = 1 - (((R2290/CR2290) + (R2320/CR2320) + (R2330/CR2330))/((R2120/CR2120) + (R2170/CR2170) + (R2210/CR2210)))
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def BD2355(self, check = False):
        if check:
            paramValue = (2300, 2450)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueDepth(spectrum,wvt,2300, 2355, 2450) #(fe/mg phyllo group)
        return paramValue
    
    def BDCARB(self, check = False):
        if check:
            paramValue = (2230, 2600)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract channels, replacing CRISM_NAN with IEEE NAN
            R2230 = u.getRvalue(spectrum, wvt, 2230)
            R2320 = u.getRvalue(spectrum, wvt, 2320)
            R2330 = u.getRvalue(spectrum, wvt, 2330)
            R2390 = u.getRvalue(spectrum, wvt, 2390)
            R2520 = u.getRvalue(spectrum, wvt, 2520)
            R2530 = u.getRvalue(spectrum, wvt, 2530)
            R2600 = u.getRvalue(spectrum, wvt, 2600)
        
            # identify nearest CRISM wavelengths
            WL1 = u.getClosestWavelength(2230,wvt)
            WC1 = (u.getClosestWavelength(2330,wvt)+u.getClosestWavelength(2320,wvt))*0.5
            WH1 = u.getClosestWavelength(2390,wvt)
            a =  (WC1 - WL1)/(WH1 - WL1)  # a gets multipled by the longer (higher wvln)  band
            b = 1.0-a                     # b gets multiplied by the shorter (lower wvln) band
        
            WL2 =  u.getClosestWavelength(2390,wvt)
            WC2 = (u.getClosestWavelength(2530,wvt) + u.getClosestWavelength(2520,wvt))*0.5
            WH2 =  u.getClosestWavelength(2600,wvt)
            c = (WC2 - WL2)/(WH2 - WL2)   # c gets multipled by the longer (higher wvln)  band
            d = 1.0-c                           # d gets multiplied by the shorter (lower wvln) band
        
            # compute bdcarb
            paramValue = 1.0 - (np.sqrt((((R2320 + R2330)*0.5)/(b*R2230 + a*R2390))*(((R2520 + R2530)*0.5)/(d*R2390 + c*R2600))))
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    # MIN parameters
    def MIN2295_2480(self, check = False):
        if check:
            paramValue = (2165, 2570)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue1 = u.getRvalueDepth(spectrum, wvt,2165,2295,2364)
            paramValue2 = u.getRvalueDepth(spectrum,wvt,2364,2480,2570)
            paramValue3 = [paramValue1,paramValue2]
            paramValue = np.min(paramValue3)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def MIN2250(self, check = False):
        if check:
            paramValue = (2165, 2350)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue1 = u.getRvalueDepth(spectrum, wvt,2165, 2210, 2350)
            paramValue2 = u.getRvalueDepth(spectrum,wvt,2165, 2265, 2350)
            paramValue3 = [paramValue1,paramValue2]
            paramValue = np.min(paramValue3)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def MIN2345_2537(self, check = False):
        if check:
            paramValue = (2250, 2602)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue1 = u.getRvalueDepth(spectrum, wvt,2250, 2345, 2430)
            paramValue2 = u.getRvalueDepth(spectrum,wvt,2430, 2537, 2602)
            paramValue3 = [paramValue1,paramValue2]
            paramValue = np.min(paramValue3)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue    
    
    # All othe parameters (ratios, slopes, peaks)
    def ISLOPE(self, check = False):
        if check:
            paramValue = (1815, 2530)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            # extract individual bands
            R1815 = u.getRvalue(spectrum, wvt, 1815)
            R2530 = u.getRvalue(spectrum, wvt, 2530)
        
            W1815 = u.getClosestWavelength(1815,wvt)
            W2530 = u.getClosestWavelength(2530,wvt)
        
            # want in units of reflectance / um
            paramValue = 1000.0 * ( R1815 - R2530 )/(W2530 - W1815)
            if paramValue is -np.inf:
                paramValue = np.nan
        return paramValue
    
    def RPEAK1(self, check = False):
        if check:
            return (442, 963)
        elif not check:
            spectrum = self.spectrum
            wvt = list(self.wvt)
            rp_wv = [442,533,600,710,740,775,800,833,860,892]
            rp_i = [wvt.index(u.getClosestWavelength(i,wvt)) for i in rp_wv]
            rp_w = [u.getClosestWavelength(i,wvt) for i in rp_wv]
            rp_ = spectrum.iloc[rp_i]
            x_ = np.linspace(rp_w[0],rp_w[-1],num=5000)
            rp_l = np.empty(np.shape(spectrum))
            rp_r = np.empty(np.shape(spectrum))
            coefs = np.polyfit(rp_w,rp_[:],5)
            poly = np.poly1d(coefs)
            y_ = list(poly(x_))
            rp_l = x_[y_.index(np.max(y_))]/1000 #wavelength value of peak reflectance
            rp_r = np.max(y_)
            self.rpeak_reflectance = rp_r
            return rp_l
   
    def BDI1000VIS(self,rp_r=False, check = False):
        if check:
            return (833, 989)
        elif not check:
            spectrum = self.spectrum
            wvt = list(self.wvt)
            if rp_r is False:
                rp_l = self.RPEAK1()
                rp_r = self.rpeak_reflectance
            # multispectral version
            # bdi_wv = [833,860,892,925,951,984,1023] 
            # vi = [wvt.index(u.getClosestWavelength(i,wvt)) for i in bdi_wv]
            # wv_um = [u.getClosestWavelength(i,wvt)/1000 for i in bdi_wv]
            
            vi0 = wvt.index(u.getClosestWavelength(833,wvt))
            vi1 = wvt.index(u.getClosestWavelength(1023,wvt))
            n = vi1-vi0 + 1
            vi = np.linspace(vi0,vi1,n,dtype=int)
            wv_um = [wvt[i]/1000 for i in vi]
            bdi1000_spectrum = spectrum.iloc[vi]
            bdi_norm = np.empty(np.shape(bdi1000_spectrum))
            for b in range(len(vi)):
                bdi_norm[b] = bdi1000_spectrum.iloc[b]/rp_r
            
            coefs = np.polyfit(wv_um,bdi_norm,3)
            poly = np.poly1d(coefs)
            pint = np.poly1d(poly.integ())
            paramValue = pint(wv_um[-1])-pint(wv_um[0])#it.(wv_um,1.0-spec_vec)
            return paramValue
   
    def IRR2(self, check = False):
        if check:
            paramValue = (2210, 2530)
        elif not check:
            spectrum = self.spectrum
            wvt = self.wvt
            paramValue = u.getRvalueRatio(spectrum,wvt,2530,2210)
        return paramValue
    
    # -------------------------------------------------------------------------
    # Run the parameters
    # -------------------------------------------------------------------------
    def run(self):
        # loop through paramList, add result to a tuple. keep track of valid parameters
        param_dict = oreXpressParamCalculator.__dict__.copy()
        parameter_array = np.empty((len(self.validParams),len(self.specNames)))
        j = 0 #spectrum index tracker
        for spectrum in self.specNames:
            i = 0 #parameter index tracker
            self.spectrum = self.spectra.iloc[:,j]
            for param in self.validParams:
                if param == 'BDI1000VIS' and 'RPEAK1' in self.validParams:
                    # avoid calculating RPEAK1 values twice
                    parameter_array[i,j] = param_dict[param](self, rp_r=self.rpeak_reflectance)
                elif isinstance(param_dict[param](self), tuple):
                    parameter_array[i,j] = param_dict[param](self)[0]
                else:
                    parameter_array[i,j] = param_dict[param](self)
                i += 1
            j += 1

        self.parameter_df = pd.DataFrame(parameter_array,columns=self.specNames)
        return self.parameter_df