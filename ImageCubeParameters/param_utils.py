#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 15:34:25 2022
utility functions for parameter calculations

@author: phillms1
"""
import numpy as np

# function for parallel processing
def getPolyInt(args):
    wv_um_,spec_vec,d = args
    coefs = np.polyfit(wv_um_,spec_vec,d)
    poly=np.poly1d(coefs)
    pint = np.poly1d(poly.integ())
    integ = pint(wv_um_[-1])-pint(wv_um_[0])
    return integ
    
def getPoly(args):
    rp_w,rp_flat,d = args
    coefs = np.polyfit(rp_w,rp_flat,d)
    poly=np.poly1d(coefs)
    return poly

class utility_functions:
    def __init__(self):
        print('utilities on')
        
    def getBand(cube,wvt,wl,kwidth = 5):
        delta = [q-wl for q in wvt]
        bindex = delta.index(min(delta,key=abs))
        if kwidth == 1:
            r = cube[:,:,bindex]
        else:
            w = (kwidth-1)/2
            r = np.median(cube[:,:,int(bindex-w):int(bindex+w)],axis=2)
        return r
    
    def getClosestWavelength(wl,band_list):
        delta = [q-wl for q in band_list]
        return band_list[delta.index(min(delta,key=abs))]
    
    def buildSummary(p1,p2,p3):
        shp = np.shape(p1)
        shp = np.append(shp,3)
        a = np.empty(shp)
        a[:,:,0]=p1
        a[:,:,1]=p2
        a[:,:,2]=p3
        return a
    
    def getBandDepth(self,cube, wvt,low,mid,hi,lw=5,mw=5,hw=5):
        # retrieve bands from cube
        Rlow = self.getBand(cube,wvt,low,kwidth=lw)
        Rmid = self.getBand(cube, wvt,mid,kwidth=mw)
        Rhi = self.getBand(cube,wvt,hi,kwidth=hw)
        
        # determine wavelengths for low, mid, hi
        WL = self.getClosestWavelength(low,wvt)
        WM = self.getClosestWavelength(mid,wvt)
        WH = self.getClosestWavelength(hi,wvt)
        
        a = (WM-WL)/(WH-WL)     #a gets multipled by the longer band
        b = 1.0-a               #b gets multiplied by the shorter band
        
        # compute the band depth using precomputed a and b
        img = 1.0 - (Rmid/(b*Rlow + a*Rhi))
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    
    def getBandDepthInvert(self, cube,wvt,low,mid,hi,lw=5,mw=5,hw=5):
        # retrieve bands from cube
        Rlow = self.getBand(cube,wvt,low,kwidth=lw)
        Rmid = self.getBand(cube,wvt,mid,kwidth=mw)
        Rhi = self.getBand(cube,wvt,hi,kwidth=hw)
    
        # determine wavelength values for closest channels
        WL = self.getClosestWavelength(low,wvt)
        WM = self.getClosestWavelength(mid,wvt)
        WH = self.getClosestWavelength(hi,wvt)
        a = (WM-WL)/(WH-WL)     # a gets multipled by the longer band
        b = 1.0-a               # b gets multiplied by the shorter band
    
        # compute the band depth using precomputed a and b
        img = 1.0 - ((b*Rlow + a*Rhi)/Rmid)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    
    def getBandRatio(self, cube,wvt,num_l,denom_l,num_w=5,denom_w=5):
        num = self.getBand(cube, wvt, num_l,kwidth=num_w)
        denom = self.getBand(cube, wvt, denom_l,kwidth=denom_w)
        img = num/denom
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    
    def normalizeParameter(p):
        return (p-np.nanmean(p))/np.nanstd(p)
    
    def cropZeros(p):
        return np.where(p<0,0.0,p)
    
    def browse2bit(B):
        A=B
        for i in range(np.shape(B)[2]):
            b = B[:,:,i]
            A[:,:,i] = np.array((255*((b-np.nanmin(b))/(np.nanmax(b)-np.nanmin(b)))),dtype='int')
        return A
    
    def stretchBand(p,stype='linear',perc = 0.02):
        if stype == 'linear_':
            pn = np.where(p==0.0,np.nan,p)
            pn = np.where(p==2.0,np.nan,p)
            sp = (pn-(1-perc)*np.nanmin(pn))/((1-perc)*(np.nanmax(pn)-np.nanmin(pn)))
            sp = np.where(np.isnan(sp) is True,0.0,sp)
            sp = np.where(sp<0,0.0,sp)
            sp = np.where(sp>1,1.0,sp)
            
        # def linearStretch(image, low_percent,high_percent):
        if stype == 'linear':
            low_percent = perc
            high_percent = perc
            sp = np.empty(np.shape(p),dtype=np.float32)
            x = p[:,:].flatten()
            xx = np.sort(x)
            N = len(xx)
            cdf = np.arange(N)/N
            low_value_index = np.argmax(cdf>low_percent)
            high_value_index = np.argmin(cdf<(1-high_percent))
            low_value = xx[low_value_index]
            high_value = xx[high_value_index]
            x = np.where(x<low_value,low_value,x)
            x = np.where(x>high_value,high_value,x)
            sp[:,:] = np.reshape(x,np.shape(p))
            sp = [(r-np.min(r))/(np.max(r)-np.min(r)) for r in sp]
            # return low_value, high_value, stretchImage
        # if flag is '1std':
        #     sp = 
        return sp
    
    def cropNZeros(self,img):
        img2=img
        n=np.shape(img)[2]
        for b in range(n):
            img2[:,:,b] = self.cropZeros(img[:,:,b])
        return img2
    
    def stretchNBands(self,img,perc=0.02):
        img2=img
        n = np.shape(img)[2]
        for b in range(n):
            img2[:,:,b]=self.stretchBand(img[:,:,b],perc=perc)
        return img2
    


    