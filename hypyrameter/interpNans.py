#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 09:45:34 2023

interpolate NaN values

input: 
    data_cube:      image cube as numpy array (rows, columns, bands)
    band_centers:   center wavelengths of data cube
    method:         interpolation method: linear, cubic, polynomial

@author: phillms1
"""
import numpy as np
from tqdm import tqdm
import copy

class interpNaNs:
    
    def __init__(self, data_cube, band_centers, method='linear'):
        self.data_cube = copy.deepcopy(data_cube)
        self.band_centers = band_centers
        self.method = method
        
    def getNanBounds(self):
        # find the nan bands
        nanIndeces = np.where(np.isnan(self.data_cube))
        nanBands = np.unique(nanIndeces[2])
        nanBands.sort()
        nanBandDiffs = np.array([nanBands[i-1] - value for i, value in enumerate(nanBands) if i > 0])
        nanBreaks = np.where(nanBandDiffs != -1)[0]
        nanBounds = [nanBands[0], nanBands[-1]]
        for i in range(len(nanBreaks)):
            nanBounds.append(nanBands[nanBreaks[i]])
            nanBounds.append(nanBands[nanBreaks[i]+1])
        nanBounds.sort()
        nanChunkCount = int((len(nanBounds)/2))
        offset = [-1,1]*nanChunkCount
        nanBounds = [nanBounds[i]+value for i, value in enumerate(offset)]
        return nanBounds, nanChunkCount
    
    def linearInterp(self):
        nanBounds, nanChunkCount = self.getNanBounds()
        print(nanBounds)
        y = []
        x = []
        m = []
        b = []
        for i in range(nanChunkCount):
            ii = i*2
            bounds = (nanBounds[ii],nanBounds[ii+1])
            y.append(np.nanmedian(self.data_cube[:,:,(bounds[0]-3):bounds[0]],axis=2))
            y.append(np.nanmedian(self.data_cube[:,:,bounds[1]:(bounds[1]+3)],axis=2))
            x.append(self.band_centers[bounds[0]])
            x.append(self.band_centers[bounds[1]])
            dy = y[ii]-y[ii+1]
            dx = x[ii]-x[ii+1]
            m.append(dy/dx)
            b.append(y[ii]-m[i]*x[ii])
            bands_to_fill = np.arange(bounds[0]+1,bounds[1])
            # breakpoint()
            for z in tqdm(bands_to_fill):
                self.data_cube[:,:,z] = m[i]*self.band_centers[z] + b[i]
                
            
        
        
        
        
    