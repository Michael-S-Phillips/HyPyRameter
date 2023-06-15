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

def getReflectanceFromSed(sedFile):
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
        




