#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:10:55 2022

@author: phillms1
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paramCalculator import oreXpressParamCalculator
from readSed import *
from readLibrary import *

date_ = '220802'
sol_ = 'Sol106'
mission_ = 'LabSamples'
rootPath = f'/Volumes/HySpex_Back/RAVEN/FieldSeason_2022/oreXpress/{date_}/{mission_}/'

