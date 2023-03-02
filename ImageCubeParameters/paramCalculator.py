#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 13:33:42 2022

@author: phillms1
"""
import numpy as np
import timeit
import matlab.engine
import os
from matplotlib import pyplot as plt 
import spectral.io.envi as envi
import multiprocessing as mp
from param_utils import utility_functions as u
from param_utils import getPoly,getPolyInt
from spectral import calc_stats,noise_from_diffs,mnf
from tqdm import tqdm


class paramCalculator:
    '''
    this class handles an input hyperspectral image (an envi .img) and returns 
    browse product summary images
    
    I/O
    vfile: /path/to/vis.hdr
        this is a .hdr file, not the .img
    sfile: /path/to/swir.hdr
        this is a .hdr file, not the .img
        
    '''
    
    def __init__(self, outdir, vfile=False, sfile=False,crop=False,bbl=[None],denoise=False):
        tic = timeit.default_timer()
        if vfile != False:
            self.vfile = vfile
            print('loading vnir object using spectral')
            self.v_ = envi.open(vfile) 
        if sfile !=False:
            self.sfile = sfile
            print('loading swir object using spectral')
            self.s_ = envi.open(sfile) 
        self.outdir = outdir
        print('\tobjects loaded\nloading data')
        
        # load wave tables
        if vfile != False:
            self.v_bands = [float(b) for b in self.v_.metadata['wavelength']]
            if 'default bands' in self.v_.metadata:
                self.v_preview_bands = [int(float(b)) for b in self.v_.metadata['default bands']]
            # loading vnir data
            self.v = np.flip(np.transpose(self.v_.load(),(1,0,2)),axis=0)
            print('\tdata loaded')
            if crop != False:
                r0,r1,c0,c1 = crop
                self.v = self.v[r0:r1,c0:c1,:]
            if denoise == True:
                print('denoising cube')
                self.denoiser(1)
        
        if sfile != False:
            self.s_bands = [float(b) for b in self.s_.metadata['wavelength']]
            if 'default bands' in self.s_.metadata:
                self.s_preview_bands = [int(float(b)) for b in self.s_.metadata['default bands']]
            # loading swir data
            self.s = np.array(self.s_.load())
            print('\tdata loaded')
            if crop!=False:
                r0,r1,c0,c1 = crop
                self.s = self.s[r0:r1,c0:c1,:]
            
            if bbl[0] is not None:
                abl = np.linspace(0, self.s_.nbands-1, self.s_.nbands)
                gbl = [i for i,b in enumerate(abl) if i not in bbl]
                self.s = self.s[:,:,gbl]
                self.s_bands = [b for i,b in enumerate(self.s_bands) if i in gbl]
            if denoise == True:
                print('denoising cube')
                self.denoiser(1)
        
        toc = timeit.default_timer()-tic
        print(f'{np.round(toc/60,2)} minutes to load data')
        
        # self.HCPINDEX2()
        # self.LCPINDEX2()
        # self.OLINDEX3()
    # -------------------------------------------------------------------------
    # utilities
    # -------------------------------------------------------------------------
    def denoiser(self,num_iter):
        eng = matlab.engine.start_matlab()
        eng.cd(os.getcwd(),nargout=0)
        print('matlab engine started')

        if hasattr(self, 'v'):
            time1 = timeit.default_timer()
            
            gNoiseIndeces = np.where(np.isnan(self.v))
            if np.shape(gNoiseIndeces[0]) > 0:
                # replace nan values with gaussian noise
                print('replacing nans with gaussian noise')
                gNormal = np.random.normal(size=np.shape(gNoiseIndeces)[1])
                for i in tqdm(range(np.shape(gNoiseIndeces)[1])):
                    self.v[gNoiseIndeces[0][i],
                           gNoiseIndeces[1][i],
                           gNoiseIndeces[2][i]] = gNormal[i]*np.nanstd(self.v[:, :, gNoiseIndeces[2][i]])
                    +np.nanmean(self.v[:, :, gNoiseIndeces[2][i]])
            if type(self.v_bands) is not list: self.v_bands = self.v_bands.tolist()
            wavelengths = matlab.double(self.v_bands)
            valid_cube = matlab.double(initializer=self.v.tolist(),size=self.v.shape,is_complex=False)
            print('\tentering matlab routine')
            self.v = np.array(eng.ngMeet_denoiser(valid_cube,wavelengths,num_iter))
            time2 = np.round((timeit.default_timer()-time1)/60,2)
            print(f'\tdenoise vnir cube took {time2} minutes')
        if hasattr(self, 's'):
            time1 = timeit.default_timer()
            
            gNoiseIndeces = np.where(np.isnan(self.s))
            if np.shape(gNoiseIndeces[0]) > 0:
                # replace nan values with gaussian noise
                print('replacing nans with gaussian noise')
                gNormal = np.random.normal(size=np.shape(gNoiseIndeces)[1])
                for i in tqdm(range(np.shape(gNoiseIndeces)[1])):
                    self.s[gNoiseIndeces[0][i],
                           gNoiseIndeces[1][i],
                           gNoiseIndeces[2][i]] = gNormal[i]*np.nanstd(self.s[:, :, gNoiseIndeces[2][i]])
                    +np.nanmean(self.s[:, :, gNoiseIndeces[2][i]])
                    
            if type(self.s_bands) is not list: self.s_bands = self.s_bands.tolist()
            wavelengths = matlab.double(self.s_bands)
            print('intializing matlab cube')
            valid_cube = matlab.double(initializer=self.s.tolist(),size=self.s.shape,is_complex=False)
            print('\tentering matlab routine')
            self.s_clean = np.array(eng.ngMeet_denoiser(valid_cube,wavelengths,num_iter))
            time2 = np.round((timeit.default_timer()-time1)/60,2)
            print(f'\tdenoise swir cube took {time2} minutes')
        
    def previewData(self):
        # preview data
        if hasattr(self, 'v'):
            plt.title('VIS Preview')
            if hasattr(self, 'v_preview_bands'):
                v_preview = self.v[:,:,self.v_preview_bands]
            else:
                choice = input('type 3 bands to display as a list\n\tlike this [55, 30, 10]')
                self.v_preview_bands = choice
            v_preview = [(r-np.min(r))/(np.max(r)-np.min(r)) for r in v_preview]
            plt.imshow(v_preview)
        if hasattr(self, 's'):
            plt.title('SWIR Preview')
            if hasattr(self, 's_preview_bands'):
                s_preview = self.v[:,:,self.s_preview_bands]
            else:
                choice = input('type 3 bands to display as a list\n\tlike this [55, 30, 10]')
                self.s_preview_bands = choice
            s_preview = self.s[:,:,self.s_preview_bands]
            s_preview = [(r-np.min(r))/(np.max(r)-np.min(r)) for r in s_preview]
            plt.imshow(s_preview)
        plt.show()

    # -------------------------------------------------------------------------
            
    # -------------------------------------------------------------------------
    # parameter library
    # -------------------------------------------------------------------------
    def HCPINDEX2(self):
        cube = self.s
        wvt = self.s_bands
        # extract data from image cube
        R2120 = u.getBand(cube, wvt,2120)
        R2140 = u.getBand(cube, wvt,2140)
        R2230 = u.getBand(cube, wvt,2230)
        R2250 = u.getBand(cube, wvt,2250)
        R2430 = u.getBand(cube, wvt,2430)
        R2460 = u.getBand(cube, wvt,2460)
        R2530 = u.getBand(cube, wvt,2530)
        R1690 = u.getBand(cube, wvt,1690)
    
        W2120 = u.getClosestWavelength(2120,wvt)
        W2140 = u.getClosestWavelength(2140,wvt)
        W2230 = u.getClosestWavelength(2230,wvt)
        W2250 = u.getClosestWavelength(2250,wvt)
        W2430 = u.getClosestWavelength(2430,wvt)
        W2460 = u.getClosestWavelength(2460,wvt)
        W2530 = u.getClosestWavelength(2530,wvt)
        W1690 = u.getClosestWavelength(1690,wvt)
    
    
        # compute the corrected reflectance interpolating 
        slope = (R2530 - R1690) / (W2530 - W1690)      
        intercept = R2530 - slope * W2530
    
        # weighted sum of relative differences
        Rc2120 = slope*W2120 + intercept
        Rc2140 = slope*W2140 + intercept
        Rc2230 = slope*W2230 + intercept
        Rc2250 = slope*W2250 + intercept
        Rc2430 = slope*W2430 + intercept
        Rc2460 = slope*W2460 + intercept
    
        img=((1-(R2120/Rc2120))*0.1) + ((1-(R2140/Rc2140))*0.1) + ((1-(R2230/Rc2230))*0.15) + ((1-(R2250/Rc2250))*0.3) + ((1-(R2430/Rc2430))*0.2) + ((1-(R2460/Rc2460))*0.15)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def LCPINDEX2(self):
        cube=self.s
        wvt=self.s_bands
        # extract data from image cube
        R1690 = u.getBand(cube, wvt, 1690)
        R1750 = u.getBand(cube, wvt, 1750)
        R1810 = u.getBand(cube, wvt, 1810)
        R1870 = u.getBand(cube, wvt, 1870)
        R1560 = u.getBand(cube, wvt, 1560)
        R2450 = u.getBand(cube, wvt, 2450)
    
        W1690 = u.getClosestWavelength(1690,wvt)
        W1750 = u.getClosestWavelength(1750,wvt)
        W1810 = u.getClosestWavelength(1810,wvt)
        W1870 = u.getClosestWavelength(1870,wvt)
        W1560 = u.getClosestWavelength(1560,wvt)
        W2450 = u.getClosestWavelength(2450,wvt)
    
        # compute the corrected reflectance interpolating 
        slope = (R2450 - R1560)/(W2450 - W1560)
        intercept = R2450 - slope * W2450
    
        # weighted sum of relative differences
        Rc1690 = slope*W1690 + intercept
        Rc1750 = slope*W1750 + intercept
        Rc1810 = slope*W1810 + intercept
        Rc1870 = slope*W1870 + intercept
    
        img=((1-(R1690/Rc1690))*0.2) + ((1-(R1750/Rc1750))*0.2) + ((1-(R1810/Rc1810))*0.3) + ((1-(R1870/Rc1870))*0.3)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def OLINDEX3(self):
        cube=self.s
        wvt=self.s_bands
        # extract data from image cube
        R1210 = u.getBand(cube, wvt,1210)
        R1250 = u.getBand(cube, wvt,1250)
        R1263 = u.getBand(cube, wvt,1263)
        R1276 = u.getBand(cube, wvt,1276)
        R1330 = u.getBand(cube, wvt,1330)
        R1750 = u.getBand(cube, wvt,1750)
        R1862 = u.getBand(cube, wvt,1862)
    
        # find closest Hyspex wavelength
        W1210 = u.getClosestWavelength(1210,wvt)
        W1250 = u.getClosestWavelength(1250,wvt)
        W1263 = u.getClosestWavelength(1263,wvt)
        W1276 = u.getClosestWavelength(1276,wvt)
        W1330 = u.getClosestWavelength(1330,wvt)
        W1750 = u.getClosestWavelength(1750,wvt)
        W1862 = u.getClosestWavelength(1862,wvt)
    
        # ; compute the corrected reflectance interpolating 
        slope = (R1862 - R1750)/(W1862 - W1750)   #;slope = ( R2120 - R1690 ) / ( W2120 - W1690 )
        intercept = R1862 - slope*W1862               #;intercept = R2120 - slope * W2120
    
        Rc1210 = slope * W1210 + intercept
        Rc1250 = slope * W1250 + intercept
        Rc1263 = slope * W1263 + intercept
        Rc1276 = slope * W1276 + intercept
        Rc1330 = slope * W1330 + intercept
    
        img = (((Rc1210-R1210)/(abs(Rc1210)))*0.1) + (((Rc1250-R1250)/(abs(Rc1250)))*0.1) + (((Rc1263-R1263)/(abs(Rc1263)))*0.2) + (((Rc1276-R1276)/(abs(Rc1276)))*0.2) + (((Rc1330-R1330)/(abs(Rc1330)))*0.4)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    # old, but functional, RPEAK1
    def RPEAK1(self):
        cube = self.v
        wvt = self.v_bands
        rp_wv = [442,533,600,710,740,775,800,833,860,892,925,963]
        rp_i = [wvt.index(u.getClosestWavelength(i,wvt)) for i in rp_wv]
        rp_w = [u.getClosestWavelength(i,wvt) for i in rp_wv]
        rp_ = cube[:,:,rp_i]
        x_ = np.linspace(rp_w[0],rp_w[-1],num=521)
        flatShape=(np.shape(rp_)[0]*np.shape(rp_)[1],np.shape(rp_)[2])       
        rp_l = np.zeros(flatShape[0])#[]#np.empty(flatShape[0])
        rp_r = np.zeros(flatShape[0])#[]#np.empty(flatShape[0])
        rp_flat = np.reshape(rp_,flatShape)
        # nanIndeces = np.where(rp_flat==0.0)
        goodIndeces = np.where(rp_flat!=0.0)
        goodIndx = np.unique(goodIndeces[0])
        # nanIndx = np.flipud(np.unique(nanIndeces[0]))
        # rp_flat_nan_index = np.where(rp_flat is np.nan)
        # coefs=[]
        poly=[]
        print('\tpreparing polynomial arguments')
        # could try to parallelize this
        # coefs=[np.polyfit(rp_w, rp_flat[i,:], 5) for i in tqdm(goodIndx)]
        # print('\tcreating polynomial functions')
        # poly=[np.poly1d(coefs[i]) for i in tqdm(list(range(len(coefs))))]
        
        # parallel attempt
        args = [(rp_w,rp_flat[i,:],5) for i in tqdm(goodIndx)]
        print('\n\tcalculating polynomials')
        with mp.Pool(6) as pool:
            for p in pool.imap(getPoly,args):
                poly.append(p)
            
        print('\treturning peak reflectance and wavelengths of peak reflectance')
        for j,i in tqdm(enumerate(goodIndx)):
            rp_l[i] = x_[list(poly[j](x_)).index(np.nanmax(poly[j](x_)))]/1000 
            rp_r[i] = np.nanmax(poly[j](x_)) 
        
        print('\tre-shaping arrays')
        shape2d = (np.shape(rp_)[0],np.shape(rp_)[1])
        rp_l=np.reshape(rp_l,shape2d)
        rp_r=np.reshape(rp_r,shape2d)
        return rp_l,rp_r
    
    def BDI1000VIS(self,rp_r=None):
        cube = self.v
        wvt = self.v_bands
        if rp_r is None:
            rp_l, rp_r = self.RPEAK1()
            
        # multispectral version
        # bdi_wv = [833,860,892,925,951,984,989] 
        # vi = [wvt.index(u.getClosestWavelength(i,wvt)) for i in bdi_wv]
        # wv_um = [u.getClosestWavelength(i,wvt)/1000 for i in bdi_wv]
        # wv_ = np.linspace(wv_um[0],wv_um[-1],num=201)
        
        vi0 = wvt.index(u.getClosestWavelength(833,wvt))
        vi1 = wvt.index(u.getClosestWavelength(989,wvt))
        n = vi1-vi0 + 1
        vi = np.linspace(vi0,vi1,n,dtype=int)
        wv_um = [wvt[i]/1000 for i in vi]
        bdi1000_cube = cube[:,:,vi]
        bdi_norm = np.empty(np.shape(bdi1000_cube))
        print('\tnormalizing input data')
        for b in tqdm(range(len(vi))):
            bdi_norm[:,:,b] = bdi1000_cube[:,:,b]/rp_r
        
        flatShape=(np.shape(bdi_norm)[0]*np.shape(bdi_norm)[1],np.shape(bdi_norm)[2])       
        bdi1000vis_value = np.zeros(flatShape[0])
        bdi_norm_flat = np.reshape(bdi_norm,flatShape)
        args = []
        bdi1000vis_value = []
        print('\tpreparing polynomial arguments')
        for i in tqdm(range(flatShape[0])):
            spec_vec = bdi_norm_flat[i,:]
            keepIndx = np.where(~np.isnan(spec_vec))[0]
            wv_um_ = [wv_um[q] for q in keepIndx]
            spec_vec_ = [spec_vec[q] for q in keepIndx]
            args.append((wv_um_,spec_vec_,4))
        print('\treturning integrated polynomial values')
        with mp.Pool(6) as pool:
            for integ in pool.imap(getPolyInt,args,chunksize=None):
                bdi1000vis_value.append(integ)
        
        print('\treshaping array')
        bdi1000vis_value = np.reshape(bdi1000vis_value,(np.shape(bdi_norm)[0],np.shape(bdi_norm)[1]))
        
        # old
        # bdi1000vis_value = np.empty((np.shape(cube)[0],np.shape(cube)[1]))
        # for i in tqdm(range(np.shape(cube)[0])):
        #     for j in range(np.shape(cube)[1]):
        #         spec_vec = bdi_norm[i,j,:]
        #         try:
        #             coefs = np.polyfit(wv_um,spec_vec,4)
        #             poly = np.poly1d(coefs)
        #             pint = np.poly1d(poly.integ())
        #             bdi1000vis_value[i,j] = pint(wv_um[-1])-pint(wv_um[0])#it.(wv_um,1.0-spec_vec)
        #         except:
        #             bdi1000vis_value[i,j] = np.nan
        return bdi1000vis_value
    def SH460(self):
        cube = self.v
        wvt = self.v_bands
        img = u.getBandDepthInvert(u, cube, wvt, 420, 460, 520)
        return img
    def BD530_2(self):
        cube = self.v
        wvt = self.v_bands
        img = u.getBandDepth(u, cube, wvt, 440, 530, 614)
        return img
    def BD670(self):
        # this is a custom parameter
        cube = self.v
        wvt = self.v_bands
        img = u.getBandDepth(u, cube, wvt, 620, 670, 745)
        return img
    def D700(self):
        # a custom parameter for chlorophyll
        cube = self.v
        wvt = self.v_bands
        
        # extract individual channels
        R630 = u.getBand(cube,wvt,630)
        R740 = u.getBand(cube,wvt,740)
        R760 = u.getBand(cube,wvt,760)
        R770 = u.getBand(cube,wvt,770)
        R690 = u.getBand(cube,wvt,690,kwidth=3)
        R710 = u.getBand(cube,wvt,710,kwidth=3)
        R720 = u.getBand(cube,wvt,720,kwidth=3)
        R830 = u.getBand(cube,wvt,830)
        
        # get closestgetClosestWavelengthngth
        W630 = u.getClosestWavelength(630,wvt)
        W740 = u.getClosestWavelength(740,wvt)
        W760 = u.getClosestWavelength(760,wvt)
        W770 = u.getClosestWavelength(770,wvt)
        W690 = u.getClosestWavelength(690,wvt)
        W710 = u.getClosestWavelength(710,wvt)
        W720 = u.getClosestWavelength(720,wvt)
        W830 = u.getClosestWavelength(830,wvt)
        
        # compute the interpolated continuum values at selected wavelengths between 630 and 830
        slope= (R830 - R630)/(W830 - W630)
        CR740 = R630 + slope*(W740 - W630)
        CR760 = R630 + slope*(W760 - W630)
        CR770 = R630 + slope*(W770 - W630)
        CR690 = R630 + slope*(W690 - W630)
        CR710 = R630 + slope*(W710 - W630)
        CR720 = R630 + slope*(W720 - W630)
    
        # compute d700 with IEEE NaN values in place of CRISM NaN
        img = 1 - (((R690/CR690) + (R710/CR710) + (R720/CR720))/((R740/CR740) + (R760/CR760) + (R770/CR770)))
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin) 
        return img
    def BD875(self):
        # this is a custom parameter
        cube = self.v
        wvt = self.v_bands
        img = u.getBandDepth(u, cube, wvt, 747, 875, 980)
        return img
    def BD920_2(self):
        cube = self.v
        wvt = self.v_bands
        img = u.getBandDepth(u, cube,wvt,807,920,984)
        return img
    def ELMSUL(self):
        # this is a custom parameter
        cube = self.v
        wvt = self.v_bands
        R400 = u.getBand(cube, wvt, 400,kwidth=1)
        R430 = u.getBand(cube, wvt, 430)
        R525 = u.getBand(cube, wvt, 525)
        R900 = u.getBand(cube, wvt, 900)
        W400 = u.getClosestWavelength(400, wvt)
        W430 = u.getClosestWavelength(430, wvt)
        W525 = u.getClosestWavelength(525, wvt)
        W900 = u.getClosestWavelength(900, wvt)
        # want in units of reflectance / um
        s1 = 1000*np.abs((R400-R430)/(W430-W400)) #should be small (close to zero)
        s2 = 1000*(R525-R430)/(W525-W430) #should be large
        s3 = 1000*np.abs((R900-R525)/(W900-W525)) #should be small (close to zero)
        img = s2-(s1+s3)
        return img, s3
    def BD2210_2(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2165,2210,2290) #(kaolinite group)
        img = u.getBandArea(u, cube, wvt, 2165, 2290)
        return img
    def BD2190(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2120,2185,2250,mw=3,hw=3) #(Beidellite, Allophane)
        img = u.getBandArea(u, cube, wvt, 2120, 2250)
        return img
    def BD2250(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2120, 2245, 2340,mw=7,hw=3) 
        img = u.getBandArea(u, cube, wvt, 2120, 2340)
        return img
    def BD2165(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2120,2165,2230,mw=3,hw=3) #(kaolinite group)
        img = u.getBandArea(u, cube, wvt, 2120, 2230)
        return img
    def BD2355(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2300, 2355, 2450) #(fe/mg phyllo group)
        img = u.getBandArea(u, cube, wvt, 2300, 2450)
        return img
    def BD2290(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,2250, 2290, 2350) #(fe/mg phyllo group)
        img = u.getBandArea(u, cube, wvt, 2210, 2350)
        return img
    def D2200(self):
        cube = self.s
        wvt = self.s_bands
        
        # extract individual channels
        R1815 = u.getBand(cube, wvt,1815, kwidth=5)
        R2165 = u.getBand(cube, wvt,2165)
        R2210 = u.getBand(cube, wvt,2210, kwidth=5)
        R2230 = u.getBand(cube, wvt,2230, kwidth=5)
        R2430 = u.getBand(cube, wvt,2430, kwidth=5)
    
    
        # retrieve wavelengths nearest the requested values
        W1815 = u.getClosestWavelength(1815, wvt)
        W2165 = u.getClosestWavelength(2165, wvt) 
        W2210 = u.getClosestWavelength(2210, wvt)
        W2230 = u.getClosestWavelength(2230, wvt)
        W2430 = u.getClosestWavelength(2430, wvt)
        
        # compute the interpolated continuum values at selected wavelengths between 1815 and 2430
        slope = (R2430 - R1815)/(W2430 - W1815)
        CR2165 = R1815 + slope*(W2165 - W1815)    
        CR2210 = R1815 + slope*(W2210 - W1815)
        CR2230 = R1815 + slope*(W2230 - W1815)
    
        # compute d2200 with IEEE NaN values
        img = 1 - (((R2210/CR2210) + (R2230/CR2230))/(2*(R2165/CR2165)))
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def D2300(self):
        cube = self.s
        wvt = self.s_bands
        # extract individual channels
        R1815 = u.getBand(cube,wvt,1815)
        R2120 = u.getBand(cube,wvt,2120)
        R2170 = u.getBand(cube,wvt,2170)
        R2210 = u.getBand(cube,wvt,2210)
        R2290 = u.getBand(cube,wvt,2290,kwidth=3)
        R2320 = u.getBand(cube,wvt,2320,kwidth=3)
        R2330 = u.getBand(cube,wvt,2330,kwidth=3)
        R2530 = u.getBand(cube,wvt,2530)
        
        # retrieve wavelengths nearest the requested values
        W1815 = u.getClosestWavelength(1815,wvt)
        W2120 = u.getClosestWavelength(2120,wvt)
        W2170 = u.getClosestWavelength(2170,wvt)
        W2210 = u.getClosestWavelength(2210,wvt)
        W2290 = u.getClosestWavelength(2290,wvt)
        W2320 = u.getClosestWavelength(2320,wvt)
        W2330 = u.getClosestWavelength(2330,wvt)
        W2530 = u.getClosestWavelength(2530,wvt)
        
        # compute the interpolated continuum values at selected wavelengths between 1815 and 2530
        slope = (R2530 - R1815)/(W2530 - W1815)
        CR2120 = R1815 + slope*(W2120 - W1815)
        CR2170 = R1815 + slope*(W2170 - W1815)
        CR2210 = R1815 + slope*(W2210 - W1815)
        CR2290 = R1815 + slope*(W2290 - W1815)
        CR2320 = R1815 + slope*(W2320 - W1815)
        CR2330 = R1815 + slope*(W2330 - W1815)
    
        # compute d2300 with IEEE NaN values
        img = 1 - (((R2290/CR2290) + (R2320/CR2320) + (R2330/CR2330))/((R2120/CR2120) + (R2170/CR2170) + (R2210/CR2210)))
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin) 
        return img
    def BD1900r2(self):
        cube = self.s
        wvt = self.s_bands
        img = u.getBandArea(u, cube, wvt, 1800, 2000)
        # extract individual channels, replacing CRISM_NANs with IEEE_NaNs
        # R1908 = u.getBand(cube,wvt,1908, kwidth = 1) 
        # R1914 = u.getBand(cube,wvt,1914, kwidth = 1) 
        # R1921 = u.getBand(cube,wvt,1921, kwidth = 1) 
        # R1928 = u.getBand(cube,wvt,1928, kwidth = 1) 
        # R1934 = u.getBand(cube,wvt,1934, kwidth = 1) 
        # R1941 = u.getBand(cube,wvt,1941, kwidth = 1) 
        # R1862 = u.getBand(cube,wvt,1862, kwidth = 1) 
        # R1869 = u.getBand(cube,wvt,1869, kwidth = 1) 
        # R1875 = u.getBand(cube,wvt,1875, kwidth = 1) 
        # R2112 = u.getBand(cube,wvt,2112, kwidth = 1) 
        # R2120 = u.getBand(cube,wvt,2120, kwidth = 1) 
        # R2126 = u.getBand(cube,wvt,2126, kwidth = 1) 
        
        # R1815 = u.getBand(cube, wvt, 1815);
        # R2132 = u.getBand(cube, wvt, 2132); 
        
        # # retrieve the CRISM wavelengths nearest the requested values
        # W1908 = u.getClosestWavelength(1908, wvt)
        # W1914 = u.getClosestWavelength(1914, wvt)
        # W1921 = u.getClosestWavelength(1921, wvt)
        # W1928 = u.getClosestWavelength(1928, wvt)
        # W1934 = u.getClosestWavelength(1934, wvt)
        # W1941 = u.getClosestWavelength(1941, wvt)
        # W1862 = u.getClosestWavelength(1862, wvt)#
        # W1869 = u.getClosestWavelength(1869, wvt)
        # W1875 = u.getClosestWavelength(1875, wvt)
        # W2112 = u.getClosestWavelength(2112, wvt)
        # W2120 = u.getClosestWavelength(2120, wvt)
        # W2126 = u.getClosestWavelength(2126, wvt)
        # W1815 = u.getClosestWavelength(1815, wvt)#  
        # W2132 = u.getClosestWavelength(2132, wvt) 
        
        # # compute the interpolated continuum values at selected wavelengths between 1815 and 2530
        # slope = (R2132 - R1815)/(W2132 - W1815)
        # CR1908 = R1815 + slope *(W1908 - W1815)
        # CR1914 = R1815 + slope *(W1914 - W1815)
        # CR1921 = R1815 + slope *(W1921 - W1815) 
        # CR1928 = R1815 + slope *(W1928 - W1815)
        # CR1934 = R1815 + slope *(W1934 - W1815)
        # CR1941 = R1815 + slope *(W1941 - W1815) 
           
        # CR1862 = R1815 + slope*(W1862 - W1815)
        # CR1869 = R1815 + slope*(W1869 - W1815)
        # CR1875 = R1815 + slope*(W1875 - W1815)    
        # CR2112 = R1815 + slope*(W2112 - W1815)
        # CR2120 = R1815 + slope*(W2120 - W1815)
        # CR2126 = R1815 + slope*(W2126 - W1815)
        # img= 1.0-((R1908/CR1908+R1914/CR1914+R1921/CR1921+R1928/CR1928+R1934/CR1934+R1941/CR1941)/(R1862/CR1862+R1869/CR1869+R1875/CR1875+R2112/CR2112+R2120/CR2120+R2126/CR2126))
        # nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        # img = np.where(img>-np.inf,img,nmin)
        return img
    def MIN2295_2480(self):
        cube = self.s
        wvt = self.s_bands
        img1 = u.getBandDepth(u,cube, wvt,2165,2295,2364)
        img2 = u.getBandDepth(u,cube,wvt,2364,2480,2570)
        img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
        img3[:,:,0] = img1
        img3[:,:,1] = img2
        img = np.min(img3,axis=2)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def MIN2250(self):
        cube = self.s
        wvt = self.s_bands
        img1 = u.getBandDepth(u,cube, wvt,2165, 2210, 2350)
        img2 = u.getBandDepth(u,cube,wvt,2165, 2265, 2350)
        img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
        img3[:,:,0] = img1
        img3[:,:,1] = img2
        img = np.min(img3,axis=2)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def MIN2345_2537(self):
        cube = self.s
        wvt = self.s_bands
        img1 = u.getBandDepth(u,cube, wvt,2250, 2345, 2430)
        img2 = u.getBandDepth(u,cube,wvt,2430, 2537, 2602)
        img3 = np.empty((np.shape(img1)[0],np.shape(img1)[1],2))
        img3[:,:,0] = img1
        img3[:,:,1] = img2
        img = np.min(img3,axis=2)
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img    
    def BDCARB(self):
        cube = self.s
        wvt = self.s_bands
        b1 = u.getBandArea(u,cube,wvt,2220,2375)
        b2 = u.getBandArea(u,cube,wvt,2400,2500)
        img = b1+b2
        # extract channels, replacing CRISM_NAN with IEEE NAN
        # R2230 = u.getBand(cube, wvt, 2230)
        # R2320 = u.getBand(cube, wvt, 2320)
        # R2330 = u.getBand(cube, wvt, 2330)
        # R2390 = u.getBand(cube, wvt, 2390)
        # R2520 = u.getBand(cube, wvt, 2520)
        # R2530 = u.getBand(cube, wvt, 2530)
        # R2600 = u.getBand(cube, wvt, 2600)
    
        # # identify nearest wavelengths
        # WL1 = u.getClosestWavelength(2230,wvt)
        # WC1 = (u.getClosestWavelength(2330,wvt)+u.getClosestWavelength(2320,wvt))*0.5
        # WH1 = u.getClosestWavelength(2390,wvt)
        # a =  (WC1 - WL1)/(WH1 - WL1)  # a gets multipled by the longer (higher wvln)  band
        # b = 1.0-a                     # b gets multiplied by the shorter (lower wvln) band
    
        # WL2 =  u.getClosestWavelength(2390,wvt)
        # WC2 = (u.getClosestWavelength(2530,wvt) + u.getClosestWavelength(2520,wvt))*0.5
        # WH2 =  u.getClosestWavelength(2600,wvt)
        # c = (WC2 - WL2)/(WH2 - WL2)   # c gets multipled by the longer (higher wvln)  band
        # d = 1.0-c                           # d gets multiplied by the shorter (lower wvln) band
    
        # # compute bdcarb
        # img = 1.0 - (np.sqrt((((R2320 + R2330)*0.5)/(b*R2230 + a*R2390))*(((R2520 + R2530)*0.5)/(d*R2390 + c*R2600))))
        # nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        # img = np.where(img>-np.inf,img,nmin)
        return img
    def SINDEX2(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepthInvert(u,cube,wvt,2120, 2290, 2400,mw=7,hw=3)
        img = u.getBandAreaInvert(u,cube,wvt, 2120, 2400, hw=3)
        return img
    def BD2100_2(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,1930, 2132, 2250,lw=3,hw=3)
        img = u.getBandArea(u, cube, wvt, 1930, 2250)
        return img
    def BD1900_2(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u,cube,wvt,1850, 1930, 2067)
        img = u.getBandArea(u, cube, wvt, 1850, 2067)
        return img
    def ISLOPE(self):
        cube = self.s
        wvt = self.s_bands
        # extract individual bands
        R1815 = u.getBand(cube, wvt, 1815)
        R2530 = u.getBand(cube, wvt, 2530)
    
        W1815 = u.getClosestWavelength(1815,wvt)
        W2530 = u.getClosestWavelength(2530,wvt)
    
        # want in units of reflectance / um
        img = 1000.0 * ( R1815 - R2530 )/ ( W2530 - W1815 )
        nmin = np.nanmin(np.where(img>-np.inf,img,np.nan))
        img = np.where(img>-np.inf,img,nmin)
        return img
    def BD1400(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u, cube, wvt, 1330, 1395, 1467, mw=3)
        img = u.getBandArea(u, cube, wvt, 1330, 1620)
        return img
    def BD1200(self):
        cube = self.s
        wvt = self.s_bands
        img = u.getBandArea(u, cube, wvt, 1115, 1260)
        return img
    def BD1450(self):
        cube = self.s
        wvt = self.s_bands
        # img = u.getBandDepth(u, cube, wvt, 1340, 1450, 1535, mw=3)
        img = u.getBandArea(u, cube, wvt, 1340, 1535)
        return img
    def BD2100_3(self):
        cube = self.s
        wvt = self.s_bands
        img = u.getBandArea(u, cube, wvt, 2016, 2220)
        return img
    def BD1750(self):
        cube = self.s
        wvt = self.s_bands
        img = u.getBandArea(u, cube, wvt, 1688, 1820)
        return img
    def GypTrip(self):
        cube = self.s
        wvt = self.s_bands
        t1 = u.getBandArea(u,cube,wvt,1420,1463)
        t2 = u.getBandArea(u,cube,wvt,1463,1515)
        t3 = u.getBandArea(u,cube,wvt,1515,1576)
        b1 = t1*t2*t3
        b2 = self.BD1750()
        img = b1+b2
        return img
    def ILL(self):
        cube = self.s
        wvt = self.s_bands
        a1 = u.getBandArea(u,cube,wvt,2160,2273)
        # a2 = u.getBandArea(u,cube,wvt,2280,2375)
        img = a1
        return img
    def IRR2(self):
        cube = self.s
        wvt = self.s_bands
        img = u.getBandRatio(u,cube,wvt,2530,2210)
        return img

    # -------------------------------------------------------------------------
    # Browse Products
    # -------------------------------------------------------------------------
    def MAF(self,norm=False):
        print('calculating MAF browse product')
        tic = timeit.default_timer()
        p1 = self.OLINDEX3()
        p2 = self.LCPINDEX2()
        p3 = self.HCPINDEX2()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'MAF finished in {round(toc,2)} seconds')
        return img
    def SUL(self,norm=False):
        print('calculating SUL browse product')
        tic = timeit.default_timer()
        p1 = self.SH460()
        p2,p3 = self.ELMSUL()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'SUL finished in {round(toc,2)} seconds')
        return img
    def HEM(self,norm=False):
        print('calculating HEM browse product')
        tic = timeit.default_timer()
        p1 = self.BD530_2()
        p2 = self.BD670()
        p3 = self.BD875()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'HEM finished in {round(toc,2)} seconds')
        return img
    def CPL(self,norm=False):
        print('calculating CPL browse product')
        tic = timeit.default_timer()
        wvt = self.v_bands
        p1 = u.getBandRatio(u,self.v, wvt, 680, 770)
        p2 = self.D700()
        p3 = u.getBandRatio(u,self.v, wvt, 470, 540)
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'CPL finished in {round(toc,2)} seconds')
        return img
        
    def FM2(self,norm=False):
        print('calculating FM2 browse product')
        tic = timeit.default_timer()
        p1 = self.BD530_2()
        p2 = self.BD920_2()
        p3 = self.BDI1000VIS()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'FM2 finished in {round(toc/60,2)} minutes')
        return img
    def FAL(self,norm=True):
        print('calculating FAL browse product')
        tic = timeit.default_timer()
        p1 = u.getBand(self.s,self.s_bands,2529)
        p2 = u.getBand(self.s,self.s_bands,1506)
        p3 = u.getBand(self.s,self.s_bands,1080)
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'FAL finished in {round(toc,2)} seconds')
        return img
    def TRU(self,norm=True):
        print('calculating TRU browse product')
        tic = timeit.default_timer()
        p1 = u.getBand(self.v,self.v_bands,637)
        p2 = u.getBand(self.v,self.v_bands,550)
        p3 = u.getBand(self.v,self.v_bands,463)
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'TRU finished in {round(toc,2)} seconds')
        return img
    def PAL(self, norm=False): 
        print('calculating PAL browse product')
        tic = timeit.default_timer()
        p1 = self.BD2210_2()
        p2 = self.BD2190()
        p3 = self.BD2165()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'PAL finished in {round(toc,2)} seconds')
        return img   
    def PHY(self, norm=False): 
        print('calculating PHY browse product')
        tic = timeit.default_timer()
        p1 = self.D2200()
        p2 = self.D2300()
        p3 = self.BD1900r2()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'PHY finished in {round(toc,2)} seconds')
        return img
    def PFM(self, norm=False): 
        print('calculating PFM browse product')
        tic = timeit.default_timer()
        p1 = self.BD2355()
        p2 = self.D2300()
        p3 = self.BD2290()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'PFM finished in {round(toc,2)} seconds')
        return img
    def CR2(self, norm=False): 
        print('calculating CR2 browse product')
        tic = timeit.default_timer()
        p1 = self.MIN2295_2480()
        p2 = self.MIN2345_2537()
        p3 = self.BDCARB()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'CR2 finished in {round(toc,2)} seconds')
        return img
    def HYD(self, norm=False): 
        print('calculating HYD browse product')
        tic = timeit.default_timer()
        p1 = self.SINDEX2()
        p2 = self.BD2100_2()
        p3 = self.BD1900_2()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'HYD finished in {round(toc,2)} seconds')
        return img
    def CHL(self, norm=False): 
        print('calculating CHL browse product')
        tic = timeit.default_timer()
        p1 = self.ISLOPE()
        p2 = self.BD1400()
        p3 = self.IRR2()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'CHL finished in {round(toc,2)} seconds')
        return img
    def HYS(self, norm=False): 
        print('calculating HYS browse product')
        tic = timeit.default_timer()
        p1 = self.MIN2250()
        p2 = self.BD2250()
        p3 = self.BD1900r2()
        if norm is True:
            p1 = u.normalizeParameter(p1)
            p2 = u.normalizeParameter(p2)
            p3 = u.normalizeParameter(p3)
        img = u.buildSummary(p1, p2, p3)
        toc = timeit.default_timer()-tic
        print(f'HYS finished in {round(toc,2)} seconds')
        return img
    # -------------------------------------------------------------------------
    # All Parameters 
    # -------------------------------------------------------------------------
    def calculateSwirParams(self):
        # SWIR Params
        tic = timeit.default_timer()
        print('\rcalculating OLINDEX3')
        p1=self.OLINDEX3()
        p1 = np.where(self.s[:,:,100]==0.0,np.nan,p1)
        print('\rcalculating LCPINDEX2')
        p2=self.LCPINDEX2()
        p2 = np.where(self.s[:,:,100]==0.0,np.nan,p2)
        print('\rcalculating HCPINDEX2')
        p3=self.HCPINDEX2()
        p3 = np.where(self.s[:,:,100]==0.0,np.nan,p3)
        print('\rcalculating BD1400')
        p4=self.BD1400()
        p4 = np.where(self.s[:,:,100]==0.0,np.nan,p4)
        print('\rcalculating BD1450')
        p5=self.BD1450()
        p5 = np.where(self.s[:,:,100]==0.0,np.nan,p5)
        print('\rcalculating BD1900_2')
        p6=self.BD1900_2()
        p6 = np.where(self.s[:,:,100]==0.0,np.nan,p6)
        print('\rcalculating BD1900r2')
        p7=self.BD1900r2()
        p7 = np.where(self.s[:,:,100]==0.0,np.nan,p7)
        print('\rcalculating BD2100_2')
        p8=self.BD2100_2()
        p8 = np.where(self.s[:,:,100]==0.0,np.nan,p8)
        print('\rcalculating BD2165')
        p9=self.BD2165()
        p9 = np.where(self.s[:,:,100]==0.0,np.nan,p9)
        print('\rcalculating BD2190')
        p10=self.BD2190()
        p10 = np.where(self.s[:,:,100]==0.0,np.nan,p10)
        print('\rcalculating BD2210_2')
        p11=self.BD2210_2()
        p11 = np.where(self.s[:,:,100]==0.0,np.nan,p11)
        print('\rcalculating BD2250')
        p12 =self.BD2250()
        p12 = np.where(self.s[:,:,100]==0.0,np.nan,p12)
        print('\rcalculating BD2290')
        p13=self.BD2290()
        p13 = np.where(self.s[:,:,100]==0.0,np.nan,p13)
        print('\rcalculating BD2355')
        p14=self.BD2355() 
        p14 = np.where(self.s[:,:,100]==0.0,np.nan,p14)
        print('\rcalculating BDCARB')
        p15=self.BDCARB()
        p15 = np.where(self.s[:,:,100]==0.0,np.nan,p15)
        print('\rcalculating D2200')
        p16=self.D2200()
        p16 = np.where(self.s[:,:,100]==0.0,np.nan,p16)
        print('\rcalculating D2300')
        p17=self.D2300()
        p17 = np.where(self.s[:,:,100]==0.0,np.nan,p17)
        print('\rcalculating IRR2')
        p18=self.IRR2()
        p18 = np.where(self.s[:,:,100]==0.0,np.nan,p18)
        print('\rcalculating ISLOPE')
        p19=self.ISLOPE()
        p19 = np.where(self.s[:,:,100]==0.0,np.nan,p19)
        print('\rcalculating MIN2250')
        p20=self.MIN2250()
        p20 = np.where(self.s[:,:,100]==0.0,np.nan,p20)
        print('\rcalculating MIN2295_2480')
        p21=self.MIN2295_2480()
        p21 = np.where(self.s[:,:,100]==0.0,np.nan,p21)
        print('\rcalculating MIN2345_2537')
        p22 = self.MIN2345_2537()
        p22 = np.where(self.s[:,:,100]==0.0,np.nan,p22)
        print('\rcalculating R2529')
        p23 = u.getBand(self.s,self.s_bands,2529)
        p23 = np.where(self.s[:,:,100]==0.0,np.nan,p23)
        print('\rcalculating R1506')
        p24 = u.getBand(self.s,self.s_bands,1506)
        p24 = np.where(self.s[:,:,100]==0.0,np.nan,p24)
        print('\rcalculating R1080')
        p25 = u.getBand(self.s,self.s_bands,1080)
        p25 = np.where(self.s[:,:,100]==0.0,np.nan,p25)
        print('\rcalculating SINDEX2')
        p26=self.SINDEX2()
        p26 = np.where(self.s[:,:,100]==0.0,np.nan,p26)
        print('\rcalculating BD1200')
        p27=self.BD1200()
        p27 = np.where(self.s[:,:,100]==0.0,np.nan,p27)
        print('\rcalculating BD2100_3')
        p28 = self.BD2100_3()
        print('\rcalculating GypTrip')
        p29 = self.GypTrip()
        print('\rcalculating ILL')
        p30 = self.ILL()
        print('\rSWIR parameters calculated!\r')
        toc = timeit.default_timer()-tic
        pTup = (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30)
        img = np.dstack(pTup)
        print(f'calculation took {round(toc/60,2)} minutes')
        return img
    
    def calculateVisParams(self):
        # VNIR PARAMETERS
        tic = timeit.default_timer()
        print('\rcalculating R637')
        p1 = u.getBand(self.v,self.v_bands,637)
        p1 = np.where(self.v[:,:,100]==0.0,np.nan,p1)
        print('\rcalculating R550')
        p2 = u.getBand(self.v,self.v_bands,550)
        p2 = np.where(self.v[:,:,100]==0.0,np.nan,p2)
        print('\rcalculating R463')
        p3 = u.getBand(self.v,self.v_bands,463)
        p3 = np.where(self.v[:,:,100]==0.0,np.nan,p3)
        print('\rcalculating SH460')
        p4=self.SH460()
        p4 = np.where(self.v[:,:,100]==0.0,np.nan,p4)
        print('\rcalculating BD530_2')
        p5=self.BD530_2()
        p5 = np.where(self.v[:,:,100]==0.0,np.nan,p5)
        print('\rcalculating BD670')
        p6=self.BD670()
        p6 = np.where(self.v[:,:,100]==0.0,np.nan,p6)
        print('\rcalculating D700')
        p7=self.D700()
        p7 = np.where(self.v[:,:,100]==0.0,np.nan,p7)
        print('\rcalculating BD875')
        p8=self.BD875()
        p8 = np.where(self.v[:,:,100]==0.0,np.nan,p8)
        print('\rcalculating BD920_2')
        p9=self.BD920_2()
        p9 = np.where(self.v[:,:,100]==0.0,np.nan,p9)
        print('\rcalculating RPEAK1')
        p10,rp_r=self.RPEAK1()
        p10 = np.where(self.v[:,:,100]==0.0,np.nan,p10)
        print('\rcalculating BDI1000VIS')
        p11=self.BDI1000VIS(rp_r=rp_r)
        p11 = np.where(self.v[:,:,100]==0.0,np.nan,p11)
        print('\rcalculating ELMSUL')
        p12,p_=self.ELMSUL()
        p12 = np.where(self.v[:,:,100]==0.0,np.nan,p12)
        
        toc = timeit.default_timer()-tic
        pTup = (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12)
        img = np.dstack(pTup)
        print('\rVis parameters calculated!\r')
        print(f'calculation took {round(toc/60,2)} minutes')
        return img
    # -------------------------------------------------------------------------
    # MNF Products
    # -------------------------------------------------------------------------
    def SWIR_MNF(self,mask=False):
        
        tic = timeit.default_timer()
        print('processing SWIR MNF')
        print('calculating signal')
        if mask is True:
            # these values (0.0 and 2.0) are specific NaN values for HySpex data
            # you can change this to mask NaN values in your data
            mask0 = np.where(self.s==0.0,0,1)
            mask2 = np.where(self.s==2.0,0,1)
            mask = mask0*mask2
            s_signal = calc_stats(self.s,mask=mask)
        elif mask is False:
            s_signal = calc_stats(self.s)
        print('calculating noise')
        rowCenter = int(np.ceil(np.shape(self.s)[0]/2))
        colCenter = int(np.ceil(np.shape(self.s)[1]/2))
        s_noise = noise_from_diffs(self.s[(rowCenter-100):rowCenter+100,(colCenter-100):(colCenter+100),:])
        print('calculating mnf')
        s_mnfr = mnf(s_signal, s_noise)
        print('reducing mnf')
        s_mnf10 = s_mnfr.reduce(self.s, num=10)
        toc = timeit.default_timer()-tic
        print('done!')
        print(f'{round(toc/60,2)} minutes to calculate SWIR MNF')
        return s_mnf10
    
    def VIS_MNF(self,mask=False):
        tic = timeit.default_timer()
        print('processing VIS MNF')
        print('calculating signal')
        if mask==True:
            mask0 = np.where(self.v==0.0,0,1)
            mask2 = np.where(self.s==2.0,0,1)
            mask = mask0*mask2
            v_signal = calc_stats(self.v,mask=mask)
        elif mask==False:
            v_signal = calc_stats(self.v)
            
        print('calculating noise')
        rowCenter = int(np.ceil(np.shape(self.v)[0]/2))
        colCenter = int(np.ceil(np.shape(self.v)[1]/2))
        v_noise = noise_from_diffs(self.v[(rowCenter-100):rowCenter+100,(colCenter-100):(colCenter+100),:])
        print('calculating mnf')
        v_mnfr = mnf(v_signal, v_noise)
        print('reducing mnf')
        v_mnf10 = v_mnfr.reduce(self.v, num=10)
        toc = timeit.default_timer()-tic
        print('done!')
        print(f'{round(toc/60,2)} minutes to calculate VIS MNF')
        return v_mnf10
    
    
    
    
    
    