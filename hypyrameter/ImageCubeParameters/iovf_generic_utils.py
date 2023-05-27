#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:18:52 2023

@author: phillms1
"""

import multiprocessing
from outliers import smirnov_grubbs as grubbs
import numpy as np
import timeit
import warnings

def get_vote_block(args):
    warnings.filterwarnings('error')
    # for parallelizing grubbiness
    grubbs_chunks,windows,identifier = args
    # print(type(grubbs_chunks))
    # print(np.shape(grubbs_chunks))
    volume = int(windows[0]*windows[1]*windows[2])
    if volume < 3*5*3:
        alpha = 0.05
    else:
        alpha = 0.125
    vb = np.zeros(volume, dtype=int)
    if True in np.isnan(grubbs_chunks):
        grubbs_chunks = np.where(np.isnan(grubbs_chunks), 0., grubbs_chunks)
    try:
        o = grubbs.two_sided_test_indices(grubbs_chunks, alpha=alpha)
    except RuntimeWarning:
        o = []
    except (IndexError, TypeError) as error:
        print(error)
        print(identifier)
        o = []
    if len(o) == volume:
        o = []
    for vote in o:
        vb[vote] = 1
    vb = np.reshape(vb, windows)
    return vb, identifier


def run_vote_block(args, ws, cpus, parallel = False, chunksize = None):
    if parallel:
        denom = int(np.ceil(240000000/ws))
        if chunksize is None:
            chunksize = int(np.ceil(np.asarray(args,dtype='object').shape[0]/denom))
        vbsi = []
        if __name__ == 'iovf_generic_utils':
            tic = timeit.default_timer()
            print('\n\tparallel processing grubbs test')
            with multiprocessing.Pool(cpus) as pool:
                for vb in pool.imap(get_vote_block, args,chunksize=chunksize):
                    vbsi.append(vb)
            toc = np.round((timeit.default_timer()-tic)/60,2)
            print(f'\n\tgrubbs testing took {toc} minutes')
    else:
        # this has not been tested
        vbsi = [get_vote_block(arg) for arg in args]
    return vbsi
