''' --------------------------------------------------------------------
                PYSCO: PYthon Self Calibrating Observables
    --------------------------------------------------------------------
    ---
    pysco is a python module to create, and extract Kernel-phase data 
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    This file contains the definition of the kpo class:
    --------------------------------------------------

    an object that contains Ker-phase information (kpi), data (kpd) 
    and relevant additional information extracted from the fits header
    (hdr)
    -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits as pf
import copy
import pickle
import os
import sys
# import pdb
import glob
import gzip

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

from scipy.interpolate import griddata

from scipy.io.idl import readsav

from core import *
from kpi import *

class kpo():
    ''' Class used to manipulate multiple Ker-phase datasets

        -------------------------------------------------------------------
        The class is designed to handle a single or multiple frames
        that can be combined for statistics purpose into a single data
        set.
        ------------------------------------------------------------------- '''

    def __init__(self, kp_fname):
        # Default instantiation.
        self.kpi = kpi(kp_fname)
        try :                               
            self.uv = self.kpi.uv # for convenience!
            self.name = self.kpi.name
        except:
            print("UV info was not loaded from kpi")                                                
        # if the file is a complete (kpi + kpd) structure
        # additional data can be loaded.
        try:
            myf = gzip.GzipFile(kp_fname, "r")
            data = pickle.load(myf)
            myf.close()

            self.kpd = data['kpd']
            self.kpe = data['kpe']
            self.hdr = data['hdr']  
            self.wavel = data['wavel']  
            self.nsets = data['nsets']                                                  
            #self.nsets = np.size(self.hdr)     # [AL, 2014.03.19] Added nsets parameter                                                    
            #if nsets==0 : self.wavel = self.hdr['filter']
            #else : self.wavel = self.hdr[0]['filter']                                                                                          
        except:
            print("File %s contains kpi information only" % (kp_fname,))
                                                
        try:                                        
            self.bsp = data['bsp']
            self.bspe = data['bspe']                                                
        except:
            print("Bsp data was not loaded")    
                                                
        try:                                        
            self.vis2 = data['vis2']
            self.vis2e = data['vis2e']                                              
        except:
            print("Vis2 data was not loaded")                                                   

    # =========================================================================
    # =========================================================================
    # [AL, 2014.03.18] Added sg_ld and D parameters - window size in lambda/D
    #              if D<=0 then use the shortest baseline instead
    # [AL, 2014.03.20] Recentering and windowing parameters added
    # [AL, 2014.04.17] use_main_header option added to replace individual headers from the datacube by the main one
    # [AL, 2014.05.06] Extract bispectrum (bsp)
    # [AL, 2014.05.28] Adjust sampling points in pupil plane to make coordinates in uv-plane integer (increases quality)
    # [AL, 2014.10.07] unwrap_kp flag added. Kernel phases unwrapping is off by default
    def extract_kpd(self, path, plotim=False, ave="none",manual=0, re_center=True, window=True, sg_ld=1.0, D=0.0, 
        bsp=False, use_main_header=False, adjust_sampling=True, unwrap_kp=False):

        ''' extract kernel-phase data from one or more files (use regexp).

        If the path leads to a fits data cube, or to multiple single frame
        files, the extracted kernel-phases are consolidated into a
        unique kpd object.      

        Using the 'manual' flag invites you to click through files manually.
        
        '''
        fnames = glob.glob(path)
        nf = fnames.__len__()
        if nf != 0:
            fits_hdr = pf.getheader(fnames[0])
        elif nf == 0:
            fits_hdr = pf.getheader(fnames)
        else:
            print 'Frame number error'
        
        print "%d frames will be open" % (nf,)

        hdrs = [] # empty list of kp data headers

        # =========================
        if fits_hdr['NAXIS'] < 3:
            kpds = np.zeros((nf, self.kpi.nkphi)) # empty 2D array of kp
            vis2s = np.zeros((nf,self.kpi.nbuv))
            if bsp:
                bsps=[]                                                 
            for i, fname in enumerate(fnames):
                                                     # [AL, 2014.03.10] Added plotim parameter
                                                       # [AL, 2014.03.18] Added D and sg_ld parameters
                                                       # [AL, 2014.03.21] Added recenter and window parameters
                                                                
                res = \
                    extract_from_fits_frame(fname, self.kpi, save_im=True,plotim=plotim,
                    manual=manual,sg_ld=sg_ld,D=D,re_center=re_center,window=window,bsp=bsp,adjust_sampling=adjust_sampling, unwrap_kp=unwrap_kp)
                if bsp :
                    (hdr, sgnl, vis2, im, ac, bsp_res)=res
                else :
                    (hdr, sgnl, vis2, im, ac)=res                                   
                self.im, self.ac = im, ac
                kpds[i] = sgnl
                vis2s[i] = vis2
                if bsp :
                    bsps.append(bsp_res)                                                                    
                hdrs.append(hdr)
            if nf == 1:
                hdrs = hdr
                kpds = sgnl
                vis2s = vis2
                if bsp :                                                                
                    bsps=bsp_res                
        # =========================
        if fits_hdr['NAXIS'] == 3:
            kpds = np.zeros((fits_hdr['NAXIS3'], 
                             self.kpi.nkphi)) # empty kp array
            vis2s = np.zeros((fits_hdr['NAXIS3'],self.kpi.nbuv))    # [AL, 2014.04.16] Added                                                                                
            dcube = pf.getdata(fnames[0])
            nslices = fits_hdr['NAXIS3']
            #nslices=20 # hardcode                                                          
            if bsp:
                bsps=[] 
            for i in xrange(nslices):
                sys.stdout.write(
                    "\rextracting kp from img %3d/%3d" % (i+1,nslices))                                                                             
                sys.stdout.flush()  
                # [AL, 2014.12.09] correction to avoid uv-points readjustment                                                               
                if adjust_sampling and i==0 :
                    adj=True
                else : adj=False                                                                                                                        
                res = extract_from_array(dcube[i], fits_hdr, self.kpi, 
                                                 save_im=False, re_center=re_center,
                                                 wrad=50.0, plotim=plotim,sg_ld=sg_ld,D=D,bsp=bsp,adjust_sampling=adj,unwrap_kp=unwrap_kp)# [AL, 2014.04.16] Added plotim parameter
                                                                    #[AL, 2014.04.16] Added D and sg_ld parameters
                                                                    # [AL, 2014.03.21] changed re_center default value 
                if bsp :
                    (hdr, sgnl, vis2, bsp_res)=res
                else :
                    (hdr, sgnl, vis2)=res                                                                                                                                                                                                       
                kpds[i] = sgnl
                vis2s[i]= vis2
                hdrs.append(hdr)
                if bsp :
                    bsps.append(bsp_res)       
        # [Al, 2014.05.02] kpe and vis2e definition changed 
        # [Al, 2014.05.29] kpe is standard error now    
        # [AL, 2014.08.26] fixed kpe calculation (shift to mean instead of zero)                                                                        
        if len(kpds.shape)==2 :
            # [AL, 2015.01.21] Unbiased error                                   
            self.kpe = np.std(kpds-np.mean(kpds,axis=0), axis=0)/np.sqrt(kpds.shape[0]-1)
            self.vis2e = np.std(vis2s-np.mean(vis2s,axis=0), axis=0)/np.sqrt(kpds.shape[0]-1)
            if bsp : 
                self.bspe=np.std(bsps-np.mean(bsps,axis=0), axis=0)/np.sqrt(kpds.shape[0]-1)                                                
        else :
            self.kpe = np.zeros(self.kpi.nkphi)
            self.vis2e = np.zeros(np.shape(vis2s))  
            if bsp : 
                self.bspe=np.zeros(np.shape(np.asarray(bsps)))                                                  
                                                
        if ave == "median":
            print " median average"

            if nf>1 or fits_hdr['NAXIS'] == 3:
                self.hdr = hdrs[0]
                self.kpd = np.median(kpds, 0)
                self.vis2 = np.median(vis2s,0)                
                if bsp : 
                    self.bsp=np.median(bsps,0)                                                              
            else :
                self.hdr = hdrs 
                self.kpd = np.asarray(kpds, 0)
                self.vis2 = np.asarray(vis2s,0)
                if bsp : 
                    self.bsp=np.asarray(bsps)                                                           
            self.wavel = self.hdr['filter']                                                     

        elif ave == "mean":
            print " mean average"
            self.kpd = np.mean(kpds, 0)
            self.vis2 = np.mean(vis2s,0)
            if nf>1 or fits_hdr['NAXIS'] == 3:
                self.hdr = hdrs[0]
                self.kpd = np.mean(kpds, 0)
                self.vis2 = np.mean(vis2s,0)    
                if bsp : 
                    self.bsp=np.mean(bsps,0)                                                                    
            else :
                self.hdr = hdrs
                self.kpd = np.asarray(kpds, 0)
                self.vis2 = np.asarray(vis2s,0) 
                if bsp : 
                    self.bsp=np.asarray(bsps)
            self.wavel = self.hdr['filter']
                                                
        elif ave == "none":
            print " no average"
            self.kpd = np.asarray(kpds)
            self.vis2 = np.asarray(vis2s)
            self.hdr = hdrs 
            if fits_hdr['NAXIS']>3 :                                                
                self.nsets = nslices # =np.size(hdrs)   # [AL, 2014.04.22] changed to nslices
            else :                                              
                self.nsets = np.size(hdrs)  # [AL, 2014.04.22] changed to nslices                                                               
            #self.nsets = np.size(hdrs)                                             
            if self.nsets == 1:
                self.wavel = self.hdr['filter']
                self.nsets = 1
            elif not use_main_header:
                self.wavel = [] # create a list of wavelengths
                for hd in hdrs:
                    self.wavel.append(hd['filter'])
            else :
                self.wavel=self.hdr[0]['filter'] # [AL, 2014.04.22] changed
                self.nsets=nslices      
            if nf==1 and bsp:
                self.bsp=np.asarray(bsps)                                                                   



    # =========================================================================
    # =========================================================================
    def extract_idl(self, path, mfpath, plotim=False, ave="none"):
        '''Extracts kernel phase data from an idlvar file.        
        '''

        data = readsav(path)

        self.hdr = get_idl_keywords(mfpath+data.mf_file)
        self.wavel = self.hdr['filter']
        self.bispectrum = data.bs

        self.kpd = data.cp
        self.vis2 = data.v2

        self.kpe = data.cp_sig
        self.vis2e = data.v2_sig

        # self.kpe = np.std(kpds, 0)
        # self.vis2e = np.std(vis2s,0)

        # if ave == "median":
        #     print "median average"
        #     self.kpd = np.median(kpds, 0)
        #     self.vis2 = np.median(vis2s,0)
        #     self.hdr = hdrs[0]

        # if ave == "mean":
        #     print "mean average"
        #     self.kpd = np.mean(self.kpds, 0)
        #     self.vis2 = np.mean(self.vis2s,0)
        #     self.hdr = hdrs[0]

        # if ave == "none":
        #     print "no average"
        #     self.kpd = self.kpds
        #     self.vis2 = self.vis2s
        #     self.hdr = self.hdrs

    # =========================================================================
    # =========================================================================
    def copy(self):
        ''' Returns a deep copy of the Multi_kpd object.
        '''
        res = copy.deepcopy(self)
        return res

    # =========================================================================
    # =========================================================================
    def calibrate(self, calib, regul="None"):
        ''' Returns a new instance of Multi_kpd object.

        Kernel-phases are calibrated by the calibrator passed as parameter.
        Assumes for now that the original object and the calibrator are
        collapsed into one single kp data set. '''

        res = copy.deepcopy(self)

        if np.size(calib.kpd.shape) == 1:
            res.kpd -= calib.kpd
            return res
        else:
            coeffs = super_cal_coeffs(self, calib, regul)
            
        return res

    # =========================================================================
    # =========================================================================
    def average_kpd(self, algo="median"):
        ''' Averages the multiple KP data into a single series.

        Default is "median". Other option is "mean".
        '''
        if algo == "median":
            aver = np.median(self.kpd, 0)
        else:
            aver = np.mean(self.kpd, 0)


        # ----------------------
        # update data structures
        # ----------------------
        self.kpd = aver

        # -------------------------------------
        # update data header (mean orientation)
        # -------------------------------------
        nh = self.hdr.__len__()
        ori = np.zeros(nh)

        for i in xrange(nh):
            ori[i] = self.hdr[i]['orient']

        self.hdr = self.hdr[0] # only keep one header
        self.hdr['orient'] = np.mean(ori)

        return self.kpd

    # =========================================================================
    # =========================================================================
    def save_to_file(self, fname):
        '''Saves the kpi and kpd data structures in a pickle

        --------------------------------------------------------------
        The data can then later be reloaded for additional analysis,
        without having to go through the sometimes time-consuming
        extraction from the original fits files.

        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpd.gz", so as to make
        it obvious that the file is a gzipped kpd data structure.
        --------------------------------------------------------------
        '''

        try:
            data = {'name'   : self.kpi.name,
                    'mask'   : self.kpi.mask,
                    'uv'     : self.kpi.uv,
                    'TFM'    : self.kpi.TFM,
                    'KerPhi' : self.kpi.KerPhi,
                    'RED'    : self.kpi.RED}                                                                            
        except:
            print("kpi data structure is incomplete")
            print("File %s was not saved to disk" % (fname,))
            return(None)

        try:
            data['hdr'] = self.hdr
            data['kpd'] = self.kpd
            data['kpe'] = self.kpe # [AL : 26.02.2014] Corrected due to misprint

        except:
            print("kpd data structure is incomplete")
            print("File %s was nevertheless saved to disk" % (fname,))
        
        # [AL, 2014.05.08] Bispectral data saving                                       
        try:
            data['bsp'] = self.bsp
            data['bspe'] = self.bspe
        except:
            print("Bsp data is missing")
                                                
        try:
            data['wavel'] = self.wavel
            data['nsets'] = self.nsets  
        except:
            print("Wavelength or Nsets data is missing")                                                
                                                
        # [AL, 2014.05.08] Vis2 data saving                                     
        try:
            data['vis2'] = self.vis2
            data['vis2e'] = self.vis2e
        except:
            print("Vis2 data is missing")

        # [AL, 2014.05.23]  uvrelations data      
        try:
            data['uvrel'] = self.kpi.uvrel
        except:
            print("Uvrel data is missing")                                              
                                                
        try:
            myf = gzip.GzipFile(fname, "wb")
        except:
            print("File %s cannot be created" % (fname,))
            print("Data was not saved to disk")
            return(None)

        pickle.dump(data, myf, -1)
        myf.close()
        print("File %s was successfully written to disk" % (fname,))
        return(0)

    # =========================================================================
    # =========================================================================
    def plot_uv_phase_map(self, data=None, reso=400):

        uv = self.kpi.uv
        
        Kinv = np.linalg.pinv(self.kpi.KerPhi)

        dxy = np.max(np.abs(uv))
        xi = np.linspace(-dxy, dxy, reso)
        yi = np.linspace(-dxy, dxy, reso)

        if data == None:
            data = np.dot(Kinv, self.kpd)
        z1 = griddata((np.array([uv[:,0], -uv[:,0]]).flatten(),
                       np.array([uv[:,1], -uv[:,1]]).flatten()),
                      np.array([data, -data]).flatten(),
                      (xi[None,:], yi[:,None]), method='linear')
        
        z2 = griddata((np.array([uv[:,0], -uv[:,0]]).flatten(), 
                       np.array([uv[:,1], -uv[:,1]]).flatten()), 
                      np.array([self.kpi.RED, self.kpi.RED]).flatten(),
                      (xi[None,:], yi[:,None]), method='linear')

        plt.imshow(z1)
        return (z1, z2)
