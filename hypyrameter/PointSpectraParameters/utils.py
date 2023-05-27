#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:37:22 2022

@author: phillms1
"""

import numpy as np

# -------------------------------------------------------------------------
# utility functions
# -------------------------------------------------------------------------                
def getRvalue(spectrum,wvt,wl,kwidth = 5):
    delta = [q-wl for q in wvt]
    bindex = delta.index(min(delta,key=abs))
    if kwidth == 1:
        r = spectrum.iloc[bindex]
    else:
        w = (kwidth-1)/2
        r = np.median(spectrum.iloc[int(bindex-w):int(bindex+w)])
    return r

def getClosestWavelength(wl,wvt):
    delta = [q-wl for q in wvt]
    return wvt[delta.index(min(delta,key=abs))]

# def buildSummary(p1,p2,p3):
#     shp = np.shape(p1)
#     shp = np.append(shp,3)
#     a = np.empty(shp)
#     a[:,:,0]=p1
#     a[:,:,1]=p2
#     a[:,:,2]=p3
#     return a

def getRvalueDepth(spectrum, wvt,low,mid,hi,lw=5,mw=5,hw=5):
    # retrieve bands from spectrum
    Rlow = getRvalue(spectrum,wvt,low,kwidth=lw)
    Rmid = getRvalue(spectrum, wvt,mid,kwidth=mw)
    Rhi  = getRvalue(spectrum,wvt,hi,kwidth=hw)
    
    # determine wavelengths for low, mid, hi
    WL = getClosestWavelength(low,wvt)
    WM = getClosestWavelength(mid,wvt)
    WH = getClosestWavelength(hi,wvt)
    
    a = (WM-WL)/(WH-WL)     #a gets multipled by the longer band
    b = 1.0-a               #b gets multiplied by the shorter band
    
    # compute the band depth using precomputed a and b
    paramValue = 1.0 - (Rmid/(b*Rlow + a*Rhi))
    return paramValue

def getRvalueDepthInvert(spectrum,wvt,low,mid,hi,lw=5,mw=5,hw=5):
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

def getRvalueRatio(spectrum,wvt,num_l,denom_l,num_w=5,denom_w=5):
    num = getRvalue(spectrum, wvt, num_l,kwidth=num_w)
    denom = getRvalue(spectrum, wvt, denom_l,kwidth=denom_w)
    paramValue = num/denom
    if paramValue is -np.inf:
        paramValue = np.nan
    return paramValue

def normalizeParameter(p):
    return (p-np.nanmean(p))/np.nanstd(p)