#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 09:50:57 2023

    input: 
        pre_file - HSP_..._PRE.IMG 
        out_dir - output directory
    output:
        flt_ - HSP_..._FLT.IMG saved to output directory.

@author: phillms1
"""
import sys
import os
import copy
import numpy as np
# import pandas as pd
import timeit
import itertools
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# import matlab.engine
from outliers import smirnov_grubbs as grubbs
from spectral import envi
from scipy import signal, interpolate
# from utils import img_cube
from iovf_generic_utils import run_vote_block
from tqdm import tqdm
from IPython.display import clear_output

class iovf:
    '''
    This class handles input image cubes (as np.arrays) and filters using an iterative outlier voting routine.
    inputs:
        pre_file - np.array of image cube (rows, columns, bands)
        out_dir - full path of output directory, ending with a '/'
        band_centers - optional input for center wavelength position of each band
        window - the size of the filtering window (aka kernel)
    outputs:
        a filtered cube saved as a .img in out_dir with .hdr file.
        
    TODO:
    '''
    
    def __init__(self, pre_image, out_dir, band_centers=False):
        
        self.pre_cube = pre_image
        self.out_dir = out_dir
        if band_centers:
            self.band_centers = band_centers

    def get_surrounding_indeces(self,update_indeces,window=None):
        if window is None: 
            print('must specifiy a window size')
            sys.exit(1)
        
        ii = int(np.floor(window[0]/2))
        jj = int(np.floor(window[1]/2))
        bb = int(np.floor(window[2]/2))
        a = np.linspace(-ii,window[0]-ii-1,num=window[0],dtype=int)
        b = np.linspace(-jj,window[1]-jj-1,num=window[1],dtype=int)
        c = np.linspace(-bb,window[2]-bb-1,num=window[2],dtype=int)
        a_,b_,c_ = np.meshgrid(a,b,c,indexing='ij')
        a_=a_.flatten()
        b_=b_.flatten()
        c_=c_.flatten()
        new_update_indeces = []
        for j in tqdm(range(np.shape(update_indeces)[1])):
            for i in range(len(a_)):
                new_update_indeces.append([update_indeces[0][j]+a_[i],update_indeces[1][j]+b_[i],update_indeces[2][j]+c_[i]])
        print('removing duplicate indeces')
        new_update_indeces.sort()
        new_update_indeces = list(new_update_indeces for new_update_indeces,_ in itertools.groupby(new_update_indeces))
        return new_update_indeces
    
    def get_grubbs_chunks(self, noise_cube, window=None,update_indeces=None):
        if window is None: 
            print('must specifiy a window size')
            sys.exit(1)
        
        s = noise_cube.shape
        if update_indeces is None:
            pixel_count = s[2]*s[1]*s[0]
            grubbs_chunks = np.empty(pixel_count,dtype='object')
            windows = []
            identifier = []
            counter = 0
            for b in tqdm(range(s[2])):
                for j in range(s[1]):
                     for i in range(s[0]):
                        # account for edge pixels
                        ii = int(np.floor(window[0]/2))
                        jj = int(np.floor(window[1]/2))
                        bb = int(np.floor(window[2]/2))
                        i0 = 0
                        i1 = 0
                        j0 = 0
                        j1 = 0
                        b0 = 0
                        b1 = 0
                        if i-ii < 0:
                            i0 = -(i-ii)
                        elif 1+i+ii > s[0]:
                            i1 = -(1+i+ii - s[0])
                        if j-jj < 0:
                            j0 = -(j-jj)
                        elif 1+j+jj > s[1]:
                            j1 = -(1+j+jj - s[1])
                        if b-bb < 0:
                            b0 = -(b-bb)
                        elif 1+b+bb > s[2]:
                            b1 = -(1+b+bb - s[2])
    
                        delta = [i0-i1,j0-j1,b0-b1]
                        adj_window = [window[g]-delta[g] for g in range(len(window))]
                        # volume = int(adj_window[0]*adj_window[1]*adj_window[2])
                        # vb = np.zeros(volume,dtype=int)
    
                        # grab the data from the window to run the grubbs test on
                        data = noise_cube[i-ii+i0:i+ii-i1+1,
                                          j-jj+j0:j+jj-j1+1,
                                          b-bb+b0:b+bb-b1+1].flatten()
                        
                        if type(data) is None:
                            print(f'pixel: {[i,j,b]} has a problem')
                            breakpoint() 
                        windows.append(adj_window)
                        grubbs_chunks[counter] = data
                        identifier.append(counter)
                        counter += 1
        
        elif update_indeces is not None:
            pixel_count = np.shape(update_indeces)[0]
            grubbs_chunks = np.empty(pixel_count,dtype='object')
            windows = []
            identifier = []
            # print('we have update_indeces')
            for i in tqdm(range(np.shape(update_indeces)[0])):
                # account for edge pixels
                ii = int(np.floor(window[0]/2))
                jj = int(np.floor(window[1]/2))
                bb = int(np.floor(window[2]/2))
                i0 = 0
                i1 = 0
                j0 = 0
                j1 = 0
                b0 = 0
                b1 = 0
                if update_indeces[i][0]-ii < 0:
                    i0 = -(update_indeces[i][0]-ii)
                elif 1+update_indeces[i][0]+ii > s[0]:
                    i1 = -(1+update_indeces[i][0]+ii - s[0])
                if update_indeces[i][1]-jj < 0:
                    j0 = -(update_indeces[i][1]-jj)
                elif 1+update_indeces[i][1]+jj > s[1]:
                    j1 = -(1+update_indeces[i][1]+jj - s[1])
                if update_indeces[i][2]-bb < 0:
                    b0 = -(update_indeces[i][2]-bb)
                elif 1+update_indeces[i][2]+bb > s[2]:
                    b1 = -(1+update_indeces[i][2]+bb - s[2])
                
                delta = [i0-i1,j0-j1,b0-b1]
                adj_window = [window[g]-delta[g] for g in range(len(window))]
                # volume = int(adj_window[0]*adj_window[1]*adj_window[2])
                # vb = np.zeros(volume,dtype=int)
                
                # grab the data from the window to run the grubbs test on
                data = noise_cube[update_indeces[i][0]-ii+i0:update_indeces[i][0]+ii-i1+1,
                                  update_indeces[i][1]-jj+j0:update_indeces[i][1]+jj-j1+1,
                                  update_indeces[i][2]-bb+b0:update_indeces[i][2]+bb-b1+1].flatten()
                
                pixel_index = update_indeces[i][0]+update_indeces[i][1]*s[0]+update_indeces[i][2]*s[0]*s[1]
                
                windows.append(adj_window)
                grubbs_chunks[i] = data
                identifier.append(pixel_index)
            
        return grubbs_chunks,windows,identifier
    
    def get_vote_block_serial(self,grubbs_chunks,windows):
        # not parallel
        vbs = []
        for i in tqdm(range(np.shape(grubbs_chunks)[0])):
            volume = int(windows[i][0]*windows[i][1]*windows[i][2])
            vb = np.zeros(volume,dtype=int)
            o = grubbs.two_sided_test_indices(grubbs_chunks[i],alpha=0.1)
            for vote in o:
                vb[vote]=1
            vb = np.reshape(vb,windows[i])
            vbs.append(vb)
        return vbs
    
    def build_vote_cube(self,vbsi,s,window=None,update_indeces=None):
        if window is None: 
            print('must specifiy a window size')
            sys.exit(1)
            
        vc = np.zeros(s,dtype=int)
        if update_indeces is None:
            pixel_index = 0
            for b in tqdm(range(s[2])):
                for j in range(s[1]):
                    for i in range(s[0]):
                        # account for edge pixels
                        ii = int(np.floor(window[0]/2))
                        jj = int(np.floor(window[1]/2))
                        bb = int(np.floor(window[2]/2))
                        i0 = 0
                        i1 = 0
                        j0 = 0
                        j1 = 0
                        b0 = 0
                        b1 = 0
                        if i-ii < 0:
                            i0 = -(i-ii)
                        elif 1+i+ii > s[0]:
                            i1 = -(1+i+ii - s[0])
                        if j-jj < 0:
                            j0 = -(j-jj)
                        elif 1+j+jj > s[1]:
                            j1 = -(1+j+jj - s[1])
                        if b-bb < 0:
                            b0 = -(b-bb)
                        elif 1+b+bb > s[2]:
                            b1 = -(1+b+bb - s[2])
                              
                        identifier = vbsi[pixel_index][1]
                        try:
                            vc[i-ii+i0:i+ii-i1+1,
                               j-jj+j0:j+jj-j1+1,
                               b-bb+b0:b+bb-b1+1] = vbsi[identifier][0] + vc[i-ii+i0:i+ii-i1+1,
                                                                             j-jj+j0:j+jj-j1+1,
                                                                             b-bb+b0:b+bb-b1+1]
                        except ValueError:
                            print(f'window size mismatch :< on pixel\n\t{i,j,b}')
                            breakpoint()
                        pixel_index += 1
                        
        elif update_indeces is not None:
            for i in tqdm(range(np.shape(update_indeces)[0])):
                # account for edge pixels
                ii = int(np.floor(window[0]/2))
                jj = int(np.floor(window[1]/2))
                bb = int(np.floor(window[2]/2))
                i0 = 0
                i1 = 0
                j0 = 0
                j1 = 0
                b0 = 0
                b1 = 0
                if update_indeces[i][0]-ii < 0:
                    i0 = -(update_indeces[i][0]-ii)
                elif 1+update_indeces[i][0]+ii > s[0]:
                    i1 = -(1+update_indeces[i][0]+ii - s[0])
                if update_indeces[i][1]-jj < 0:
                    j0 = -(update_indeces[i][1]-jj)
                elif 1+update_indeces[i][1]+jj > s[1]:
                    j1 = -(1+update_indeces[i][1]+jj - s[1])
                if update_indeces[i][2]-bb < 0:
                    b0 = -(update_indeces[i][2]-bb)
                elif 1+update_indeces[i][2]+bb > s[2]:
                    b1 = -(1+update_indeces[i][2]+bb - s[2])
                    
                pixel_index = update_indeces[i][0]+update_indeces[i][1]*s[0]+update_indeces[i][2]*s[0]*s[1] 
                if vbsi[i][1] != pixel_index:
                    print(f'not good. Your pixels in update_indeces may be out of order\n\tpixel is: {update_indeces[i]}')
                    breakpoint()
    
                vc[(update_indeces[i][0]-ii+i0):(update_indeces[i][0]+ii-i1+1),
                    (update_indeces[i][1]-jj+j0):(update_indeces[i][1]+jj-j1+1),
                    (update_indeces[i][2]-bb+b0):(update_indeces[i][2]+bb-b1+1)] = vbsi[i][0] + vc[(update_indeces[i][0]-ii+i0):(update_indeces[i][0]+ii-i1)+1,
                                                                                                   (update_indeces[i][1]-jj+j0):(update_indeces[i][1]+jj-j1)+1,
                                                                                                   (update_indeces[i][2]-bb+b0):(update_indeces[i][2]+bb-b1)+1]
             
        return vc
    
    def fit_spline(self):
        d1,d2,d3 = self.flt.shape
        self.flt_flat = self.flt.reshape((d1*d2,d3))
        std = np.nanstd(self.flt_flat,axis=0)
        x = copy.deepcopy(self.band_centers)
        sf = 8 #d1-np.sqrt(2*d1)
        # fit each spectrum with a spline
        self.spl = []
        ref_cube_flat = np.empty(self.flt_flat.shape)
        noise_cube_flat = np.empty(self.flt_flat.shape)
        for i in tqdm(range(self.flt_flat.shape[0])):
            w = 1/std
            y = self.flt_flat[i,:]
            nanas = np.isnan(y)
            y[nanas] = 0.
            w[nanas] = 0
            self.spl.append(interpolate.UnivariateSpline(x, self.flt_flat[i,:], w=w,s=sf))
            # self.spl[i].set_smoothing_factor(sf)
            ref_cube_flat[i,:] = self.spl[i](x)
            noise_cube_flat[i,:] = self.flt_flat[i,:]-ref_cube_flat[i,:]
        ref_cube = ref_cube_flat.reshape((d1,d2,d3))
        noise_cube = noise_cube_flat.reshape((d1,d2,d3))
        
        return ref_cube, noise_cube
    
    def center_data(self, in_cube, filter_size):
        # initiate ref and noise cubes.  
        ref_cube = np.empty(np.shape(in_cube))
        noise_cube = np.empty(np.shape(in_cube))
        rows = np.shape(in_cube)[0]
        cols = np.shape(in_cube)[1]
        nbands = np.shape(in_cube)[2]

        # pad the input cube
        pad_sz = int(np.ceil(filter_size/2))
        pad_cube = np.empty((rows,cols,nbands+pad_sz*2))
        pad_cube[:,:,pad_sz:(pad_sz+nbands)] = in_cube[:,:,:]


        for i in range(pad_sz):
            pad_cube[:,:,i] = in_cube[:,:,pad_sz-i]
            pad_cube[:,:,i+pad_sz+nbands] = in_cube[:,:,nbands-i-1]

        tic = timeit.default_timer()
            
        h = signal.medfilt(pad_cube,kernel_size=(1,1,filter_size))
        ref_cube = h[:,:,pad_sz:(pad_sz+nbands)]
        if pad_sz % 2 == 0: 
            window = pad_sz-1 
        else: 
            window = pad_sz

        # print(f'\nfilter window size: {window}')
        poly_order = 3
        if poly_order >= window: 
            poly_order = 1
        for i in tqdm(range(rows)):
            for j in range(cols):
                ref_cube[i,j,:] = signal.savgol_filter(ref_cube[i,j,:], window, 3, mode='mirror')
                noise_cube[i,j,:] = in_cube[i,j,:]-ref_cube[i,j,:]

        toc = timeit.default_timer()-tic
        print(f'\ntime to make reference cube \n\t{round(toc,2)} s')            
        return ref_cube, noise_cube
        
    def iterative_outlier_filter(self,update_indeces=None,window=None):
        if window is None: 
            print('must specifiy a window size')
            sys.exit(1)

        cpus = 6 #number of cpus to use in parallel process
        ws = window[0]*window[1]*window[2]
            
        # update indeces if None when running the whole image cube (first iteration)
        if update_indeces is None:
            s = self.noise_cube.shape
            # vc = np.zeros(s,dtype=int)
            
            # trying new stuff to parallelize
            # get grubbs
            print('\tgetting data chunks for grubbs test')
            grubbs_chunks,windows,identifier = self.get_grubbs_chunks(self.noise_cube, window)
            
            # get voting blocks
            print('\n\tpreparing data for parallel grubbs test')
            grubbs_shape = np.asarray(grubbs_chunks).shape
            args = [(grubbs_chunks[i],windows[i],identifier[i]) for i in tqdm(range(grubbs_shape[0]))]
            
            print('\n\trunning grubbs test')
            
            # run all in parallel *****************************
            vbsi = run_vote_block(args,ws,cpus,parallel=True)
            # **********************************************************
            
            # run all in serial *****************************
            # vbs = self.get_vote_block_serial(grubbs_chunks,windows)
            # **********************************************************
            
            # build vote cube
            print('\tbuilding vote cube')
            vc = self.build_vote_cube(vbsi, s, window)
            
        else:
            clear_output(wait=True)
            s = self.noise_cube.shape
            vc = np.zeros(s, dtype=int)
            # getting the all the indeces from surrounding pixels returns the indeces flipped from the input,
            # so the for loop indexing of update_indeces is flipped from the above case for round 1
            print('updating indeces to include surrounding pixels')
            update_indeces = self.get_surrounding_indeces(update_indeces, window)
            
            # parallel processing of update indeces
            # get grubbs
            print('\tgetting data chunks for grubbs test')
            grubbs_chunks,windows,identifier = self.get_grubbs_chunks(self.noise_cube, window, update_indeces)
            
            # get voting blocks
            print('\n\tpreparing data for parallel grubbs test')
            grubbs_shape = np.asarray(grubbs_chunks).shape
            args = [(grubbs_chunks[i],windows[i],identifier[i]) for i in tqdm(range(grubbs_shape[0]))]
            
            print('\n\trunning grubbs test')
            # run all in parallel *****************************
            vbsi = run_vote_block(args,ws,cpus,parallel=True)
            
            # build vote cube
            print('\tbuilding vote cube')
            vc = self.build_vote_cube(vbsi, s, window, update_indeces)
            # **************************************************************
             
        return vc
    
    def run_iovf(self, window = None, view_votes = False):
        if window is None: 
            print('must specifiy a window size')
            sys.exit(1)
        
        # get the noise cube from the input cube
        print('creating centered noise cube')
        self.ref_cube, self.noise_cube = self.center_data(self.flt[:,:,:], 19)
        # self.ref_cube,self.noise_cube = self.fit_spline()
            
        s = self.noise_cube.shape
        
        v = True
        r = 1
        pixel_count = s[0]*s[1]*s[2]
        
        if r < 2:
            vt = np.min(window)
        else:
            vt = np.max(window)
            
        self.voting_record = []
        threshold = 0
        regress_count = 0
        
        while v:
            
            print('\n********************************\n*                              *')
            print(f'* starting round {r} of voting   *')
            print('*                              *\n********************************')

            # iterate through the filter until there are no pixels with an outlier value of >5
            print('starting the election process')
            if r < 2:
                # on the first round, run the whole cube
                tic = timeit.default_timer()
                self.vc = self.iterative_outlier_filter(window=window)
                # self.noise_cube = np.where(self.vc>=vt,np.nan,self.noise_cube)
                update_indeces = np.where(self.vc>=vt)
                # ************************************
                toc = timeit.default_timer() - tic
                print(f'election took {np.round(toc/60,2)} minutes')
                self.voting_record.append(np.shape(update_indeces)[1])
                v = bool((np.shape(update_indeces)[1])>threshold)
                print(f'condition: {v}')
                
                if view_votes:
                    # plot the vote cube ***************
                    plt_vc = np.where(self.vc>=vt, self.vc, 0)
                    plt.imshow(np.sum(plt_vc,axis=2))
                    plt.show()
                    # **********************************
                    # plot the voting record
                    # plt.plot(self.voting_record)
                    # plt.show()
                    
            else:
                # on subsequent rounds, only go to pesky pixels
                tic = timeit.default_timer()
                if 'update_indeces' not in locals():
                    update_indeces = np.where(self.vc>=vt)
                previous_update_numbers = np.shape(update_indeces)[1]
                self.vc = self.iterative_outlier_filter(update_indeces,window)
                # self.noise_cube = np.where(self.vc>=vt,np.nan,self.noise_cube)
                update_indeces = np.where(self.vc>=vt)
                # ************************************
                toc = timeit.default_timer() - tic
                if toc > 60:
                    print(f'election took {np.round(toc/60,2)} minutes')
                else:
                    print(f'election took {np.round(toc,2)} seconds')
                
                self.voting_record.append(np.shape(update_indeces)[1])
                
                if view_votes:
                    # plot the vote cube ***************
                    plt_vc = np.where(self.vc>=vt, self.vc, 0)
                    plt.imshow(np.sum(plt_vc,axis=2))
                    plt.show()
                    # **********************************
                    # plot the voting record
                    # plt.plot(self.voting_record)
                    # plt.show()
                
                # ************* CONDITION BLOCK **************************
                # if previous_update_numbers<=np.shape(update_indeces)[1]:
                #     regress_count += 1
                # else:
                #     regress_count = 0
            
                # take into account if the program is "regressing", i.e., finding more bad values on subsequent iterations
                # v = bool(np.shape(update_indeces)[1]>threshold and regress_count <= 3)
                
                
                # sf = 4 #smoothing factor
                # if (r-1) < (sf+1):
                #     # stop when bad values are less than or equal to the threshold
                #     v = bool(np.shape(update_indeces)[1]>=threshold)
                # else:
                #     # calculate the derivative of the voting record, stop when it smooths out
                #     self.dVote = [d-self.voting_record[i-1] for i, d in enumerate(self.voting_record)if i>0]
                #     self.dSum = [np.sum(self.dVote[(i-sf):i]) for i,x in enumerate(self.dVote) if (i+1) >=sf]
                #     dCond = bool((self.dSum[-2] <=2 and self.dSum[-2] >=-2) and (self.dSum[-1] <=1 and self.dSum[-1] >=-1))
                #     v =  bool(np.shape(update_indeces)[1]>threshold and dCond is False)
                    
                # TODO:
                    # use update indeces instead of the voting record for the stopping condition
                # check for repeating patterns
                if r>=14:
                    self.dVote = [d-self.voting_record[i-1] for i, d in enumerate(self.voting_record)if i>0]
                    # four part pattern
                    pA = self.dVote[-8:-4]
                    pB = self.dVote[-4:]
                    pCheck4 = pA==pB
                    # five part pattern
                    pC = self.dVote[-10:-5]
                    pD = self.dVote[-5:]
                    pCheck5 = pC==pD
                    # six part pattern
                    pE = self.dVote[-12:-6]
                    pF = self.dVote[-6:]
                    pCheck6 = pE==pF
                    # seven part pattern
                    pG = self.dVote[-14:-7]
                    pH = self.dVote[-7:]
                    pCheck7 = pG==pH
                    
                    pCheck = bool(pCheck4 or pCheck5 or pCheck6 or pCheck7)
                    v = bool(np.shape(update_indeces)[1]>threshold and pCheck is False)
                else:
                    v = bool(np.shape(update_indeces)[1]>threshold)
                # ***********************************************************
                
                print(f'condition: {v}')

            print(f'************\n\t{np.shape(update_indeces)[1]} values to update after {r} rounds of voting\n************')
            print('updating bad values')
            if np.shape(update_indeces)[1]>threshold:
                for i in tqdm(range(np.shape(update_indeces)[1])):
                    
                    # account for edge pixels
                    ii = int(np.floor(window[0]/2))
                    jj = int(np.floor(window[1]/2))
                    bb = int(np.floor(window[2]/2))
                    i0 = 0
                    i1 = 0
                    j0 = 0
                    j1 = 0
                    b0 = 0
                    b1 = 0
                    if update_indeces[0][i]-ii < 0:
                        i0 = -(update_indeces[0][i]-ii)
                    elif 1+update_indeces[0][i]+ii > s[0]:
                        i1 = -(1+update_indeces[0][i]+ii - s[0])
                    if update_indeces[1][i]-jj < 0:
                        j0 = -(update_indeces[1][i]-jj)
                    elif 1+update_indeces[1][i]+jj > s[1]:
                        j1 = -(1+update_indeces[1][i]+jj - s[1])
                    if update_indeces[2][i]-bb < 0:
                        b0 = -(update_indeces[2][i]-bb)
                    elif 1+update_indeces[2][i]+bb > s[2]:
                        b1 = -(1+update_indeces[2][i]+bb - s[2])
                    
                    delta = [i0-i1,j0-j1,b0-b1]
                    adj_window = [window[g]-delta[g] for g in range(len(window))]
                    
                    
                    
                    # ******************************
                    # update with grid interpolation
                    try:
                        # ****************************
                        # 3d weighted average
                        og_value = self.noise_cube[update_indeces[0][i],
                                                    update_indeces[1][i],
                                                    update_indeces[2][i]]
                        df = copy.deepcopy(self.noise_cube[update_indeces[0][i]-ii+i0:update_indeces[0][i]+ii-i1+1,
                                                                        update_indeces[1][i]-jj+j0:update_indeces[1][i]+jj-j1+1,
                                                                        update_indeces[2][i]-bb+b0:update_indeces[2][i]+bb-b1+1])
                        df[ii-i0,jj-j0,bb-b0] = np.nan
                        x, y, z = np.meshgrid(np.linspace(-1,1,adj_window[1]), np.linspace(-1,1,adj_window[0]), np.linspace(-1,1,adj_window[2]))
                        d = np.sqrt(x*x+y*y+z*z)
                        sigma, mu = 1.0, 0.0
                        g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
                        g_ = g/np.sum(g)
                        weighted_avg = np.nansum(g_*np.array(df)) #/(adj_window[0]*adj_window[1]*adj_window[2])
                        if np.isnan(weighted_avg):
                            print('we got nansin here!')
                            breakpoint()
                        # ****************************
                        if weighted_avg == og_value:
                            
                            self.noise_cube[update_indeces[0][i],
                                            update_indeces[1][i],
                                            update_indeces[2][i]] = 0
                        else:   
                            self.noise_cube[update_indeces[0][i],
                                            update_indeces[1][i],
                                            update_indeces[2][i]] = weighted_avg
                    except:
                        print('\ncaught a naked except... not good')
                        breakpoint()
                    # 
                    # ******************************
            self.update_indeces = update_indeces    
            r += 1
        return
    
    def write_FLT(self,custom_appendix=''):
        '''
        pre_file: 
            full path to pre file
        out_dir:
            full path to output directory
            
        using the envi package available in spectral to write out the data
        img_data:
            image data
        img_file_name:
            file name and path with extension, e.g.,
                /Users/phillms1/Documents/Work/test_file.img
        lbs_data:
            lbs dictionary of metadata, takes in the original HSP LBS and augments as necessary
            See img_cube class.
            will be written to the same directory as the img file but with .LBS extension
        '''
        # get output file names
        p = self.pre_file.split('/')[-1]
        l = p.split('.')[0]
        lbs_name = f'{self.out_dir}{l}_FLT{custom_appendix}.LBS'
        im_name = f'{self.out_dir}{l}_FLT{custom_appendix}.img'
        IM_name = f'{self.out_dir}{l}_FLT{custom_appendix}.IMG'
        envi_hdr = f'{self.out_dir}{l}_FLT{custom_appendix}.HDR'

        if self.detector == 'L':
            # data will be given in increasing wavelength order, so we need to reverse it again
            img_data = self.flt[..., ::-1]
        elif self.detector == 'S':
            img_data = self.flt

        if hasattr(self, 'nanCs'):
            col_nans = np.ones([self.pre_row_len,self.pre_band_len])*65535.0  #re-insert nan columns
            for i in self.nanCs:                                              #re-insert nan columns
                img_data = np.insert(img_data,i,col_nans,axis=1)              #re-insert nan columns
        if hasattr(self, 'nanRs'):
            row_nans = np.ones([self.pre_col_len,self.pre_band_len])*65535.0  #re-insert nan rows
            for i in self.nanRs:                                              #re-insert nan rows
                img_data = np.insert(img_data,i,row_nans,axis=0)              #re-insert nan rows
        if self.detector == 'L':
            # re-insert nan band
            img_data = np.where(np.isnan(img_data),65535.0,img_data)
            nan_band = np.ones([np.shape(img_data)[0],np.shape(img_data)[1],1],dtype=np.float32)*65535.0
            img_data = np.concatenate((nan_band,img_data),axis=2)

        lbs_data_ = copy.deepcopy(self.lbs)
        if self.pre_.detector == 'L':
            lbs_data_.get('BAND_NAME').reverse()
        wavelength = copy.deepcopy(lbs_data_.get('BAND_NAME')) # for envi metadata
        wavelength = [np.float32(i.replace(" ","")) for i in wavelength]

        # format the band names correctly for saving
        nb=str()
        for i in lbs_data_.get('BAND_NAME'):
            if i is lbs_data_.get('BAND_NAME')[0]:
                nb = '('
            if i != lbs_data_.get('BAND_NAME')[-1]:
                nb = f'{nb}"{i}",'+'\n             '
            if i is lbs_data_.get('BAND_NAME')[-1]:
                nb = f'{nb}"{i}")'
        lbs_data_['BAND_NAME'] = nb

        if lbs_data_.get('ROWNUM'):
            if self.pre_.detector == 'L':
                lbs_data_.get('ROWNUM').reverse()
            # format the rownum correctly for saving
            nb=str()
            for i in lbs_data_.get('ROWNUM'):
                if i is lbs_data_.get('ROWNUM')[0]:
                    nb = '('
                if i != lbs_data_.get('ROWNUM')[-1]:
                    nb = f'{nb}"{i}",'+'\n             '
                if i is lbs_data_.get('ROWNUM')[-1]:
                    nb = f'{nb}"{i}")'
            lbs_data_['ROWNUM'] = nb

        # define save details 
        intlv = 'bsq'
        dtype = np.float32
        if self.detector == 'L':
            metadata = {'default bands': [26,123,152],
                        'default stretch': '1.0% linear',
                        'wavelength units': 'Nanometers',
                        'data ignore value': 65535.0,
                        'wavelength': wavelength}
        elif self.detector == 'S':
            metadata = {'default bands': [37,26,13],
                        'default stretch': '1.0% linear',
                        'wavelength units': 'Nanometers',
                        'data ignore value': 65535.0,
                        'wavelength': wavelength}

        # save
        print(f'writing out file:\n{IM_name}\n')
        tic = timeit.default_timer()
        # print(f'{envi_hdr}\n')
        envi.save_image(envi_hdr,img_data,
                        interleave = intlv, dtype=dtype, 
                        metadata = metadata, force = 'force') 
        toc = timeit.default_timer()-tic
        print(f'saving took {np.round(toc,2)} seconds')

        # remove the envi hdr file, we don't need it, rename img to IMG
        # os.system(f'rm {envi_hdr}')
        #os.remove(envi_hdr)
        os.rename(im_name, IM_name)
        #os.system(f'mv {im_name} {IM_name}')

        # save the LBS file
        quoted_keys = ['SPACECRAFT_CLOCK_START_COUNT','SPACECRAFT_CLOCK_STOP_COUNT','OBSERVATION_TYPE',
                       'OBSERVATION_ID','MRO:SENSOR_ID','MRO:WAVELENGTH_FILE_NAME']
        print(f'writing out file:\n{lbs_name}\n')
        with open(lbs_name,'w') as f:
            for key, value in lbs_data_.items():
                # if key is 'BAND_NAME':
                #     for i in range(len(value))
                if key in quoted_keys:
                    f.write('%s = "%s"\n' % (key,value))
                else:
                    f.write('%s = %s\n' % (key,value))

        return
    
    def run(self, save=False, window=[7,5,5]):
        t0 = timeit.default_timer()
        
        print('filtering the cube')
        time1 = timeit.default_timer()
        self.flt = copy.deepcopy(self.pre_cube)
        self.run_iovf(window=window)
        self.flt = self.noise_cube + self.ref_cube
        time2 = np.round((timeit.default_timer()-time1)/60,2)
        print(f'filtering took {time2} minutes')
            
        if save:
            self.write_FLT()
            
        t1 = np.round((timeit.default_timer()-t0)/60,2)
        print(f'\ntotal time for filtering:\n\t{t1} minutes')

