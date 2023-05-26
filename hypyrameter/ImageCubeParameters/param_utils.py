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
    wv_um_,spec_vec = args
    if spec_vec.any() is np.nan:
        integrand = np.nan
    else:
        splineFit = cs(wv_um_, spec_vec)
        integrand = splineFit.integrate(wv_um_[0], wv_um_[-1])
    return integrand
    
def getPolyInt(args):
    wv_um_,spec_vec,d = args
    coefs = np.polyfit(wv_um_,spec_vec,d)
    poly=np.poly1d(coefs)
    pint = np.poly1d(poly.integ())
    integrand = pint(wv_um_[-1])-pint(wv_um_[0])
    return integrand
    
def getPoly(args):
    rp_w,rp_flat,d = args
    coefs = np.polyfit(rp_w,rp_flat,d)
    poly=np.poly1d(coefs)
    return poly

class utility_functions:
    def __init__(self):
        print('utilities on')
    
    def getBandArea(self, cube, wvt, low, high, lw=5, hw=5):
        # this isn't really band area... it's average band depth 
        y1 = self.getBand(cube, wvt, low, kwidth=lw)
        x1 = self.getClosestWavelength(low, wvt)
        y2 = self.getBand(cube, wvt, high, kwidth=hw)
        x2 = self.getClosestWavelength(high, wvt)
        m = (y2-y1)/(x2-x1) #m is the slope at all pixels
        b = y2-m*x2         #b is the intercept at all pixels
        # breakpoint()
        woi = np.linspace(wvt.index(x1),wvt.index(x2),wvt.index(x2)-wvt.index(x1)+1,dtype=int).tolist()
        h = np.zeros(y1.shape,dtype=np.float32)
        for i in woi:
            y = m*wvt[i]+b #continuum value
            h += y-self.getBand(cube,wvt,wvt[i],kwidth=1)
        # h = 1000*h/(x2-x1)
        
        
        # fit a polynomial and integrate 
        # y1 = self.getBand(cube, wvt, low, kwidth=lw)
        # x1 = self.getClosestWavelength(low, wvt)
        # y2 = self.getBand(cube, wvt, high, kwidth=hw)
        # x2 = self.getClosestWavelength(high, wvt)
        # m = (y2-y1)/(x2-x1) #m is the slope at all pixels
        # b = y2-m*x2         #b is the intercept at all pixels
        # woi = np.linspace(wvt.index(x1),wvt.index(x2),wvt.index(x2)-wvt.index(x1)+1,dtype=int).tolist()
        # wl = [wvt[i] for i in woi]
        # h = np.zeros([y1.shape[0],y1.shape[1],len(woi)],dtype=np.float32)
        # j=0
        # ba = []
        # for i in woi:
        #     y = m*wvt[i]+b #continuum value
        #     if y.all() is False: 
        #         breakpoint()
        #     h[:,:,j] = 1 - self.getBand(cube,wvt,wvt[i],kwidth=1)/y
        #     j+=1
        # s1,s2,s3 = h.shape
        # h_flat = np.reshape(h,[s1*s2,s3])
        # bad_values = np.hstack((np.where(np.isinf(h_flat)),np.where(np.isnan(h_flat))))
        # for i in range(bad_values[0].shape[0]):
        #     try:
        #         h_flat[bad_values[0][i],bad_values[1][i]] = np.median((h_flat[bad_values[0][i],bad_values[1][i]-1],
        #                                                                h_flat[bad_values[0][i],bad_values[1][i]+1]))
        #     except IndexError as error:
        #         print(error)
        #         h_flat[bad_values[0][i],bad_values[1][i]] = 0
                
        # args = [(wl, h_flat[i,:]) for i in tqdm(range(s1*s2))]
        # print('\n\treturning integrated polynomial values')
        # # breakpoint()
        # with mp.Pool(6) as pool:
        #     for integrand in pool.imap(getCubicSplineIntegral,args):
        #         ba.append(integrand)
        # ba = np.reshape(ba,[s1,s2])
        
        # h = 1000*h/(x2-x1)
        
        return h
    
    def getBandAreaInvert(self, cube, wvt, low, high, lw=5, hw=5):
        # this isn't really band area... it's average height 
        y1 = self.getBand(cube, wvt, low, kwidth=lw)
        x1 = self.getClosestWavelength(low, wvt)
        y2 = self.getBand(cube, wvt, high, kwidth=hw)
        x2 = self.getClosestWavelength(high, wvt)
        m = (y2-y1)/(x2-x1) #m is the slope at all pixels
        b = y2-m*x2         #b is the intercept at all pixels
        woi = np.linspace(wvt.index(x1),wvt.index(x2),wvt.index(x2)-wvt.index(x1)+1,dtype=int).tolist()
        h = np.zeros(y1.shape,dtype=np.float32)
        for i in woi:
            y = m*wvt[i]+b #continuum value
            h += y-self.getBand(cube,wvt,wvt[i],kwidth=1)
        # h = 1000*h/(x2-x1)
        
        return -h
    
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
    
    def getBandDepth(self, cube, wvt,low, mid, hi, lw=5, mw=5, hw=5):
        # retrieve bands from cube
        Rlow = self.getBand(cube, wvt, low, kwidth=lw)
        Rmid = self.getBand(cube, wvt, mid, kwidth=mw)
        Rhi = self.getBand(cube, wvt, hi, kwidth=hw)
        
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
    


    