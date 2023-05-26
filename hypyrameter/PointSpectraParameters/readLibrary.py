#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:42:29 2022

@author: phillms1
"""
import numpy as np
import re
import glob
import pandas as pd

def getReflectanceFromUSGS(usgsFile):
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
        




