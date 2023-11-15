#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:34:25 2022
utility functions for parameter calculations

@author: phillms1
"""
import numpy as np
from scipy.interpolate import CubicSpline as cs
import multiprocessing as mp
import math
import glob
import pandas as pd


# -----------------------------------------------------
# utility functions for image cube processing
# ----------------------------------------------------

# function for parallel processing
def getCubicSplineIntegral(args):
    """return the integral of cubic spline fit

    Args:
        args (tuple): tuple of arguments for cubic spline fit

    Returns:
        integrand: integral of cubic spline of defined wavelength region
    """
    wv_um_,spec_vec = args
    non_finite_values = [value for value in spec_vec if not math.isfinite(value)]
    if non_finite_values:
        integrand = np.nan
    else:
        splineFit = cs(wv_um_, spec_vec)
        integrand = splineFit.integrate(wv_um_[0], wv_um_[-1])
    return integrand
    
def getPolyInt(args):
    """returns integrated polynomial

    Args:
        args (tuple): tuple of arguments for polyfit

    Returns:
        integrand: integral of polynomial of defined wavelength region
    """
    wv_um_,spec_vec,d = args
    coefs = np.polyfit(wv_um_,spec_vec,d)
    poly=np.poly1d(coefs)
    pint = np.poly1d(poly.integ())
    integrand = pint(wv_um_[-1])-pint(wv_um_[0])
    return integrand
    
def getPoly(args):
    """get polynomial definition

    Args:
        args (tuple): tuple of arguments for polyfit

    Returns:
        poly: np 1d polynomial object
    """
    rp_w,rp_flat,d = args
    coefs = np.polyfit(rp_w,rp_flat,d)
    poly=np.poly1d(coefs)
    return poly


def getBandArea(cube, wvt, low, high, lw=5, hw=5):
    """retrieve band area

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): low wavelength median filter kernel width. Defaults to 5.
        hw (int, optional): high wavelength median filter kernel width. Defaults to 5.

    Returns:
        (array): band area image
    """
    y1 = getBand(cube, wvt, low, kwidth=lw)
    x1 = getClosestWavelength(low, wvt)
    y2 = getBand(cube, wvt, high, kwidth=hw)
    x2 = getClosestWavelength(high, wvt)
    m = (y2-y1)/(x2-x1) #m is the slope at all pixels
    b = y2-m*x2         #b is the intercept at all pixels
    woi = np.linspace(wvt.index(x1),wvt.index(x2),wvt.index(x2)-wvt.index(x1)+1,dtype=int).tolist()
    wol = [wvt[w] for w in woi]
    s0, s1 = y1.shape
    s2 = len(woi)
    h = np.zeros([s0,s1,s2],dtype=np.float32) #continuum subtracted cube
    # y = []
    for i in woi:
        y = (m*wvt[i]+b) #continuum value
        h[:,:,int(i-np.min(woi))] = getBand(cube,wvt,wvt[i],kwidth=1) - y
    # h = 1000*h/(x2-x1)
    h_flat = np.reshape(h,[s0*s1, s2])
    ba = []
    args = [(wol, h_flat[i,:]) for i in range(s0*s1)]
    with mp.Pool() as pool:
        for ci in pool.imap(getCubicSplineIntegral, args):
            ba.append(ci)
    ba = -1*np.reshape(ba, [s0,s1])
    return ba

def getBandAreaInvert(cube, wvt, low, high, lw=5, hw=5):
    """retrieve inverted band area

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        low (float): lowest wavelength anchor point
        high (float): highest wavelength anchor point
        lw (int, optional): kernel width for median filter. Defaults to 5.
        hw (int, optional): kernel width for median filter. Defaults to 5.

    Returns:
        (array): inverted band area image
    """
    # this isn't really band area... it's average height 
    y1 = getBand(cube, wvt, low, kwidth=lw)
    x1 = getClosestWavelength(low, wvt)
    y2 = getBand(cube, wvt, high, kwidth=hw)
    x2 = getClosestWavelength(high, wvt)
    m = (y2-y1)/(x2-x1) #m is the slope at all pixels
    b = y2-m*x2         #b is the intercept at all pixels
    woi = np.linspace(wvt.index(x1),wvt.index(x2),wvt.index(x2)-wvt.index(x1)+1,dtype=int).tolist()
    h = np.zeros(y1.shape,dtype=np.float32)
    for i in woi:
        y = m*wvt[i]+b #continuum value
        h += y-getBand(cube,wvt,wvt[i],kwidth=1)
    # h = 1000*h/(x2-x1)
    
    return -h

def getBand(cube,wvt,wl,kwidth = 5):
    """retrieve band closest to target wavelength, wl

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        wl (float): target wavelength
        kwidth (int, optional): kernel width. Defaults to 5.

    Returns:
        (array): median filtered array closest to target wavelength
    """
    delta = [q-wl for q in wvt]
    bindex = delta.index(min(delta,key=abs))
    if kwidth == 1:
        r = cube[:,:,bindex]
    else:
        w = (kwidth-1)/2
        r = np.median(cube[:,:,int(bindex-w):int(bindex+w)],axis=2)
    return r

def getClosestWavelength(wl,band_list):
    """retrieve closest wavelength

    Args:
        wl (float): target wavelength
        band_list (list): list of wavelength values

    Returns:
        (float): band closest to target wavelength
    """
    delta = [q-wl for q in band_list]
    return band_list[delta.index(min(delta,key=abs))]

def buildSummary(p1,p2,p3):
    """builds  3-band summary product (browse product)

    Args:
        p1 (array): parameter 1
        p2 (array): parameter 2
        p3 (array): parameter 3

    Returns:
        (array): 3-band browse product
    """
    shp = np.shape(p1)
    shp = np.append(shp,3)
    a = np.empty(shp)
    a[:,:,0]=p1
    a[:,:,1]=p2
    a[:,:,2]=p3
    return a

def getBandDepth(cube, wvt,low, mid, hi, lw=5, mw=5, hw=5):
    """retrieves band depth

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        low (float): wavelength for lower anchor point
        mid (float): wavelength for center of band
        hi (float): wavelength for upper anchor point
        lw (int, optional): kernel width for median filter. Defaults to 5.
        mw (int, optional): kernel width for median filter. Defaults to 5.
        hw (int, optional): kernel width for median filter. Defaults to 5.

    Returns:
        img (array): band depth imag
    """
    # retrieve bands from cube
    Rlow = getBand(cube, wvt, low, kwidth=lw)
    Rmid = getBand(cube, wvt, mid, kwidth=mw)
    Rhi = getBand(cube, wvt, hi, kwidth=hw)
    
    # determine wavelengths for low, mid, hi
    WL = getClosestWavelength(low,wvt)
    WM = getClosestWavelength(mid,wvt)
    WH = getClosestWavelength(hi,wvt)
    
    a = (WM-WL)/(WH-WL)     #a gets multipled by the longer band
    b = 1.0-a               #b gets multiplied by the shorter band
    
    # compute the band depth using precomputed a and b
    img = 1.0 - (Rmid/(b*Rlow + a*Rhi))
    nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
    img = np.where(img>-np.inf,img,nmin)
    return img

def getBandDepthInvert(cube,wvt,low,mid,hi,lw=5,mw=5,hw=5):
    """retrieve inverted band depth

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        low (float): wavelength for lower anchor point
        mid (float): wavelength for center of band
        hi (float): wavelength for upper anchor point
        lw (int, optional): kernel width for median filter. Defaults to 5.
        mw (int, optional): kernel width for median filter. Defaults to 5.
        hw (int, optional): kernel width for median filter. Defaults to 5.

    Returns:
        img (array): inverted band depth image
    """
    # retrieve bands from cube
    Rlow = getBand(cube,wvt,low,kwidth=lw)
    Rmid = getBand(cube,wvt,mid,kwidth=mw)
    Rhi = getBand(cube,wvt,hi,kwidth=hw)

    # determine wavelength values for closest channels
    WL = getClosestWavelength(low,wvt)
    WM = getClosestWavelength(mid,wvt)
    WH = getClosestWavelength(hi,wvt)
    a = (WM-WL)/(WH-WL)     # a gets multipled by the longer band
    b = 1.0-a               # b gets multiplied by the shorter band

    # compute the band depth using precomputed a and b
    img = 1.0 - ((b*Rlow + a*Rhi)/Rmid)
    nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
    img = np.where(img>-np.inf,img,nmin)
    return img

def getBandRatio(cube,wvt,num_l,denom_l,num_w=5,denom_w=5):
    """retrieves band ratio

    Args:
        cube (array): multiband image array
        wvt (list): wave table
        num_l (float): numerator wavelength value
        denom_l (float): denomenator wavelength value
        num_w (int, optional): number of wavelengths from which to take a median value for numerator. Defaults to 5.
        denom_w (int, optional): number of wavelengths from which to take a median value for denom. Defaults to 5.

    Returns:
        img (array): band ratio image
    """
    num = getBand(cube, wvt, num_l,kwidth=num_w)
    denom = getBand(cube, wvt, denom_l,kwidth=denom_w)
    img = num/denom
    nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
    img = np.where(img>-np.inf,img,nmin)
    return img

def normalizeParameter(p):
    """normlizes an array

    Args:
        p (array): parameter array

    Returns:
        (array): normalized array with mean offset and std scaling
    """
    return (p-np.nanmean(p))/np.nanstd(p)

def browse2bit(B):
    """converts floating point arrays to uint8

    Args:
        B (array): multiband image array

    Returns:
        (array): converted uint8 array
    """
    A=B
    for i in range(np.shape(B)[2]):
        b = B[:,:,i]
        A[:,:,i] = np.array((255*((b-np.nanmin(b))/(np.nanmax(b)-np.nanmin(b)))),dtype='int')
    return A

def stretchBand(p, stype='linear', perc = 2, factor = 2.5):
    """function to stretch images

    Args:
        p (array): parameter band
        stype (str, optional): stretch type. Defaults to 'linear'.
        perc (int, optional): percentage for percent stretch. Defaults to 2.
        factor (float, optional): number of standard deviations in std stretch. Defaults to 2.5.
    Returns:
        sp (array): stretched array
    """
    # def linearStretch(image, low_percent,high_percent):
    if stype == 'linear':
        # Convert image to numpy array
        img_array = np.array(p)

        # Calculate lower and upper percentiles
        lower_percentile = np.percentile(img_array, perc)
        upper_percentile = np.percentile(img_array, 100-perc)

        # Linearly stretch the image
        sp = (img_array - lower_percentile) * 255 / (upper_percentile - lower_percentile)
        sp = np.clip(sp, 0, 255)
    elif stype == 'std':
        # Convert image to numpy array
        img_array = np.array(p)

        # Calculate mean and standard deviation of pixel values
        mean = np.mean(img_array)
        std = np.std(img_array)

        # Define lower and upper bounds based on standard deviation
        lower_bound = mean - (std * factor)
        upper_bound = mean + (std * factor)

        # Perform standard deviation stretch
        sp = (img_array - lower_bound) * (255 / (upper_bound - lower_bound))
        sp = np.clip(sp, 0, 255)

    return sp

def stretchNBands(img, stype = 'linear', perc=2, factor=2.5):
    """stretch multiple bands using stretchBand

    Args:
        img (multiband array): typically a 3-band browse product array
        stype (str, optional): stretch type. Defaults to 'linear'.
        perc (int, optional): percentage for percent stretch. Defaults to 2.
        factor (float, optional): number of standard deviations in std stretch. Defaults to 2.5.

    Returns:
        img2 (multiband array): stretch image array
    """
    img2=img
    n = np.shape(img)[2]
    for b in range(n):
        img2[:,:,b]=stretchBand(img[:,:,b], stype=stype, perc=perc, factor=factor)
    return img2

def cropZeros(p):
    """crops values below zero in an array

    Args:
        p (array): 2d image array

    Returns:
        (array): array with values below zero set to zero
    """
    return np.where(p<0,0.0,p)

def cropNZeros(img):
    """crops values below zero in a multiband array using cropZeros

    Args:
        img (multiband array): typically a 3-band browse product image

    Returns:
        img2: cropped multiband array
    """
    img2=img
    n=np.shape(img)[2]
    for b in range(n):
        img2[:,:,b] = cropZeros(img[:,:,b])
    return img2


# -------------------------------------------------------------------------
# point spectra utility functions
# -------------------------------------------------------------------------                
import pandas as pd
import numpy as np

def getRvalue(spectrum, wvt, wl, kwidth=5):
    """
    Returns the R value of a given spectrum at a specified wavelength.

    Parameters:
    spectrum (pandas.DataFrame): A pandas DataFrame containing the spectrum data.
    wvt (list): A list of wavelengths corresponding to the spectrum data.
    wl (float): The wavelength at which to calculate the R value.
    kwidth (int): The width of the kernel to use for calculating the R value. Default is 5.

    Returns:
    float: The R value of the spectrum at the specified wavelength.
    """
    delta = [q - wl for q in wvt]
    bindex = delta.index(min(delta, key=abs))
    if kwidth == 1:
        r = spectrum.iloc[bindex]
    else:
        w = (kwidth - 1) / 2
        min_index = bindex - w
        max_index = bindex + w
        if bindex - w < 0:
            min_index = 0
        if bindex + w > len(spectrum) - 1:
            max_index = len(spectrum) - 1
        r = np.median(spectrum.iloc[int(min_index):int(max_index)])
    return r

def getRvalueDepth(spectrum, wvt,low,mid,hi,lw=5,mw=5,hw=5):
    """
    Compute the band depth using precomputed a and b.

    Args:
    spectrum (numpy.ndarray): 1D array of spectral values.
    wvt (numpy.ndarray): 1D array of wavelength values.
    low (float): Lower wavelength value.
    mid (float): Middle wavelength value.
    hi (float): Higher wavelength value.
    lw (int): Width of the lower band.
    mw (int): Width of the middle band.
    hw (int): Width of the higher band.

    Returns:
    float: The computed band depth.
    """
    # retrieve bands from spectrum
    Rlow = getRvalue(spectrum, wvt, low, kwidth=lw)
    Rmid = getRvalue(spectrum, wvt, mid, kwidth=mw)
    Rhi  = getRvalue(spectrum, wvt, hi, kwidth=hw)
    
    # determine wavelengths for low, mid, hi
    WL = getClosestWavelength(low,wvt)
    WM = getClosestWavelength(mid,wvt)
    WH = getClosestWavelength(hi,wvt)
    
    a = (WM-WL)/(WH-WL)     #a gets multipled by the longer band
    b = 1.0-a               #b gets multiplied by the shorter band
    
    # compute the band depth using precomputed a and b
    paramValue = 1.0 - (Rmid/(b*Rlow + a*Rhi))
    return paramValue

def getRvalueDepthInvert(spectrum, wvt, low, mid, hi, lw=5, mw=5, hw=5):
    """
    Computes the band depth using the inverted method.

    Args:
        spectrum (numpy.ndarray): The spectrum to compute the band depth from.
        wvt (numpy.ndarray): The wavelengths of the spectrum.
        low (float): The lower wavelength of the band.
        mid (float): The middle wavelength of the band.
        hi (float): The higher wavelength of the band.
        lw (float, optional): The width of the lower band. Defaults to 5.
        mw (float, optional): The width of the middle band. Defaults to 5.
        hw (float, optional): The width of the higher band. Defaults to 5.

    Returns:
        float: The computed band depth.
    """
    # retrieve bands from spectrum
    Rlow = getRvalue(spectrum,wvt,low,kwidth=lw)
    Rmid = getRvalue(spectrum,wvt,mid,kwidth=mw)
    Rhi  = getRvalue(spectrum,wvt,hi,kwidth=hw)

    # determine wavelength values for closest channels
    WL = getClosestWavelength(low,wvt)
    WM = getClosestWavelength(mid,wvt)
    WH = getClosestWavelength(hi,wvt)
    a = (WM-WL)/(WH-WL)     # a gets multipled by the longer band
    b = 1.0-a               # b gets multiplied by the shorter band

    # compute the band depth using precomputed a and b
    paramValue = 1.0 - ((b*Rlow + a*Rhi)/Rmid)
    if paramValue is -np.inf:
        paramValue = np.nan
    return paramValue

def getRvalueRatio(spectrum, wvt, num_l, denom_l, num_w=5, denom_w=5):
    """
    Calculates the ratio of two R-values obtained from the given spectrum and wavelet transform.

    Args:
        spectrum (numpy.ndarray): The spectrum to calculate R-values from.
        wvt (numpy.ndarray): The wavelet transform to use for calculating R-values.
        num_l (int): The length of the numerator window.
        denom_l (int): The length of the denominator window.
        num_w (int, optional): The width of the numerator window. Defaults to 5.
        denom_w (int, optional): The width of the denominator window. Defaults to 5.

    Returns:
        float: The ratio of the two R-values. If the ratio is -inf, returns NaN instead.
    """
    num = getRvalue(spectrum, wvt, num_l, kwidth=num_w)
    denom = getRvalue(spectrum, wvt, denom_l, kwidth=denom_w)
    paramValue = num / denom
    if paramValue is -np.inf:
        paramValue = np.nan
    return paramValue

# ----------------------------------------------------------------
# read info from .sed files
# ----------------------------------------------------------------
def getReflectanceFromSed(sedFile):
    """
    Reads in a spectral energy distribution (SED) file and returns the wavelength and reflectance data.

    Args:
        sedFile (str): The path to the SED file.

    Returns:
        tuple: A tuple containing two lists - the wavelength data and the reflectance data.
    """
    with open(sedFile, 'r') as lf:
        sedInfo = np.array([line[:-1] for line in lf.readlines()])
    wvl=[]
    refl=[]
    i = 0
    # get index where data start
    for line in sedInfo:
        if line.__contains__('Wvl'):
            idx = i+1
        i = i+1
    # get data
    info = sedInfo[idx:]
    for line in info:
        b, r = line.split('\t')
        wvl.append(float(b))
        refl.append(float(r))
    return wvl,refl

def getSedFiles(sedPath):
    """
    Returns a pandas DataFrame containing reflectance values for all SED files in the given path.

    Args:
    sedPath (str): The path to the SED files.

    Returns:
    pandas.DataFrame: A DataFrame containing reflectance values for all SED files in the given path.
    """
    i = 0
    for file in glob.glob(sedPath):
        h = file.split('/')
        name = h[-1]
        wvl, r = getReflectanceFromSed(file)
        if i == 0:
            initialDict = {'Wavelength': wvl,
             name: r}
            df = pd.DataFrame(initialDict)
        else:
            df[name] = r
        i=i+1
    return df

# ----------------------------------------------------------------
# read files from USGS speclib07
# ----------------------------------------------------------------
# These functions are for reading files downloaded from the USGS splib07a library of reflectance spectra
# Input is a path to the reflectance .txt file. Output is a pandas.DataFrame of the wavelength values and 
# reflectance values.
def getReflectanceFromUSGS(usgsFile):
    """
    Reads reflectance data from a USGS file and returns it as a list.

    Parameters:
    usgsFile (str): The path to the USGS file.

    Returns:
    list: A list containing the reflectance values from the USGS file.
    """
    with open(usgsFile, 'r') as lf:
        fileInfo = np.array([line[:-1] for line in lf.readlines()])
    # get data
    refl = fileInfo[1:]
    refl = [float(i) for i in refl]
    for value in refl:
        if value<0:
            i = refl.index(value)
            refl[i] = np.nan
            refl[i] = np.nanmean(refl[(i-1):(i+1)])
    return refl

def getWavelengthFromUSGS(usgsFile):
    """
    Reads wavelength data from a USGS file and returns it as a list.

    Parameters:
    usgsFile (str): The path to the USGS file.

    Returns:
    list: A list containing the wavelength values from the USGS file converted to nm.
    """
    with open(glob.glob(usgsFile)[0], 'r') as lf:
        fileInfo = np.array([line[:-1] for line in lf.readlines()])
    # get data
    wvl = fileInfo[1:] #convert to nm
    wvl = [float(i)*1000 for i in wvl]
    for value in wvl:
        if value < 0:
            wvl[wvl.index(value)] = np.nan
    return wvl

def getSpecFiles(usgsPath):
    """
    Reads USGS files from a directory path, retrieves the wavelength and reflectance data,
    and returns a pandas DataFrame.

    Parameters:
    usgsPath (str): The path to the directory containing the USGS files.

    Returns:
    pandas.DataFrame: A DataFrame with wavelength and reflectance data from USGS files.
    """
    i = 0
    for file in glob.glob(usgsPath):
        h = file.split('/')
        name = h[-1]
        usgsWvlPath = '/'.join(h[:-1])+'/splib07a_Wavelengths*.txt'
        print(usgsWvlPath)
        wvl = getWavelengthFromUSGS(usgsWvlPath)
        r = getReflectanceFromUSGS(file)
        if i == 0:
            initialDict = {'Wavelength': wvl,
             name: r}
            df = pd.DataFrame(initialDict)
        else:
            df[name] = r
        i=i+1
    return df

