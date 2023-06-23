#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:34:25 2022
utility functions for parameter calculations

@author: phillms1
"""
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline as cs
import multiprocessing as mp

# function for parallel processing
def getCubicSplineIntegral(args):
    """return the integral of cubic spline fit

    Args:
        args (tuple): tuple of arguments for cubic spline fit

    Returns:
        integrand: integral of cubic spline of defined wavelength region
    """
    wv_um_,spec_vec = args
    if spec_vec.any() is np.nan:
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



