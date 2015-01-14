''' --------------------------------------------------------------------
                PYSCO: PYthon Self Calibrating Observables
    --------------------------------------------------------------------
    ---
    pysco is a python module to create, and extract Kernel-phase data 
    structures, using the theory of Martinache, 2010, ApJ, 724, 464.
    ----

    This file contains the definition of the kpi class:
    --------------------------------------------------

    an object that contains the linear model for the optical system
      of interest. Properties of this model are:
      --> name   : name of the model (HST, Keck, Annulus_19, ...)
      --> mask   : array of coordinates for pupil sample points
      --> uv     : matching array of coordinates in uv plane (baselines)
      --> RED    : vector coding the redundancy of these baselines
      --> TFM    : transfer matrix, linking pupil-phase to uv-phase
      --> KerPhi : array storing the kernel-phase relations
      --> uvrel :  matrix storing the relations between sampling points and uv-points
      -------------------------------------------------------------------- '''

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import pickle
import os
import sys
import gzip

from core import *

from scipy.io.idl import readsav

class kpi(object):
    ''' Fundamental kernel-phase relations

    -----------------------------------------------------------------------
    This object condenses all the knowledge about a given instrument pupil 
    geometry into a series of arrays useful for kernel-phase analysis as 
    well as for other purposes, such as wavefront sensing.
    ----------------------------------------------------------------------- '''

    name = "" # default array name. Should be descriptive of the array geometry

    # =========================================================================
    # =========================================================================

    def __init__(self, file=None, maskname=None):
        ''' Default instantiation of a KerPhase_Relation object:

        -------------------------------------------------------------------
        Default instantiation of this KerPhase_Relation class is achieved
        by loading a pre-made file, containing all the relevant information
        -------------------------------------------------------------------'''
        try:
            # -------------------------------
            # load the pickled data structure
            # -------------------------------
            myf = gzip.GzipFile(file, "r")
            data = pickle.load(myf)
            myf.close()

            # -------------------------------
            # restore the variables for this 
            # session of Ker-phase use!
            # -------------------------------
            #print "TEST:", data['toto']

            try:    self.name = data['name']
            except: self.name = "UNKNOWN"

            self.uv     = data['uv']
            self.mask   = data['mask']
            self.RED    = data['RED']
            self.KerPhi = data['KerPhi']
            self.TFM    = data['TFM']
        
            self.nbh   = self.mask.shape[0]
            self.nbuv  = self.uv.shape[0]
            self.nkphi = self.KerPhi.shape[0]
        												
            try : self.uvrel = data['uvrel']	
            except: self.uvrel = np.array([])
        
        except: 
            print("File %s isn't a valid Ker-phase data structure" % (file))
            try: 
                if maskname == None:
                    print 'Creating from coordinate file'
                    self.from_coord_file(file)
                else:
                    print 'Creating from mfdata file'
                    self.from_mf(file,maskname)
            except:
                print("Failed.")
                return None

    # =========================================================================
    # =========================================================================

    def from_mf(self, file, maskname, array_name=""):
        ''' Creation of the KerPhase_Relation object from a matched filter file.

        ----------------------------------------------------------------
        This duplicates the functionality of from_coord_file for masking data.

        Input is a matched filter idlvar file. 
        ---------------------------------------------------------------- '''

        mfdata = readsav(file)

        maskdata = readsav(maskname)

        self.mask = maskdata['xy_coords']
        self.nbh  = mfdata.n_holes   # number of sub-Ap
        print 'nbuv = ', mfdata.n_baselines
        self.nbuv = mfdata.n_baselines

        ndgt = 6 # number of digits of precision for rounding
        prec = 10**(-ndgt)

        # ================================================
        # Create a kpi representation of the closure phase
        # operator
        # ================================================
        
        self.uv = np.zeros((self.nbuv,2))

        self.uv[:,0] = mfdata.u
        self.uv[:,1] = mfdata.v

        print self.uv.shape

        # 2. Calculate the transfer matrix and the redundancy vector
        # --------------------------------------------------------------
        self.RED = np.ones(self.nbuv, dtype=float)             # Redundancy

        self.nkphi  = mfdata.n_bispect # number of Ker-phases

        self.KerPhi = np.zeros((self.nkphi, self.nbuv)) # allocate the array
        self.TFM = self.KerPhi # assuming a non-redundant array!

        for k in range(0,self.nkphi):
            yes = mfdata.bs2bl_ix[k,:]
            self.KerPhi[k,yes] = 1

    # =========================================================================
    # =========================================================================

    def from_coord_file(self, file, array_name="", Ns=3):
        ''' Creation of the KerPhase_Relation object from a pupil mask file:

        ----------------------------------------------------------------
        This is the core function of this class, really...

        Input is a pupil coordinates file, containing one set of (x,y) 
        coordinates per line. Coordinates are in meters. From this, all 
        the intermediate products that lead to the kernel-phase matrix 
        KerPhi are calculated.
								
	  Set Ns < 2 for undersampled data [AL, 20.02.2014]
        ---------------------------------------------------------------- '''
        self.mask = 1.0 * np.loadtxt(file) # sub-Ap. coordinate files 
        self.nbh  = self.mask.shape[0]   # number of sub-Ap

        ndgt = 6 # number of digits of precision for rounding
        prec = 10**(-ndgt)

        # ================================================
        # Determine all the baselines in the array.
        # ================================================

        # 1. Start by doing all the possible combinations of coordinates 
        # --------------------------------------------------------------
        # in the array to calculate the baselines. The intent here, is 
        # to work with redundant arrays of course, so there will be plenty 
        # of duplicates.

        nbh = self.nbh # local representation of the class variable

        try:
            from numba import jit
            print 'Using numba to compile'

            @jit
            def do_combinations(nbh,mask):
                tic = time.time()
                uvx = np.zeros(nbh * (nbh-1)) # prepare empty arrays to store
                uvy = np.zeros(nbh * (nbh-1)) # the baselines

                k = 0 # index for possible combinations (k = f(i,j))
                
                uvi = np.zeros(nbh * (nbh-1), dtype=int) # arrays to store the possible
                uvj = np.zeros(nbh * (nbh-1), dtype=int) # combinations k=f(i,j) !!

                for i in range(nbh):     # do all the possible combinations of
                    for j in range(nbh): # sub-apertures
                        if i != j:
                            uvx[k] = mask[i,0] - mask[j,0]
                            uvy[k] = mask[i,1] - mask[j,1]
                            # ---
                            uvi[k], uvj[k] = i, j
                            k+=1
                return uvi, uvj, uvx, uvy
            toc = time.time()
            print "%.3f seconds elapsed" % (toc-tic)

            uvi, uvj, uvx, uvy = do_combinations(nbh,self.mask)

        except:
            uvx = np.zeros(nbh * (nbh-1)) # prepare empty arrays to store
            uvy = np.zeros(nbh * (nbh-1)) # the baselines

            k = 0 # index for possible combinations (k = f(i,j))
            
            uvi = np.zeros(nbh * (nbh-1), dtype=int) # arrays to store the possible
            uvj = np.zeros(nbh * (nbh-1), dtype=int) # combinations k=f(i,j) !!

            for i in range(nbh):     # do all the possible combinations of
                for j in range(nbh): # sub-apertures
                    if i != j:
                        uvx[k] = self.mask[i,0] - self.mask[j,0]
                        uvy[k] = self.mask[i,1] - self.mask[j,1]
                        # ---
                        uvi[k], uvj[k] = i, j
                        k+=1

        try:
            a = np.unique(np.round(uvx, ndgt)) # distinct u-component of baselines
            nbx    = a.shape[0]                # number of distinct u-components

            @jit
            def fill_uv_sel(nbx,uvx,uvy,a,prec,ndgt):
                uv_sel = np.zeros((0,2)) # array for "selected" baselines
                for i in range(nbx):     # identify distinct v-coords and fill uv_sel
                    b = np.where(np.abs(uvx - a[i]) <= prec)
                    c = np.unique(np.round(uvy[b], ndgt))
                    nby = np.shape(c)[0] # number of distinct v-compoments
                    for j in range(nby):
                        uv_sel = np.append(uv_sel, [[a[i],c[j]]], axis=0)
                return uv_sel
            uv_sel = fill_uv_sel(nbx,uvx,uvy,a,prec,ndgt)


        except:
            a = np.unique(np.round(uvx, ndgt)) # distinct u-component of baselines
            nbx    = a.shape[0]                # number of distinct u-components
            uv_sel = np.zeros((0,2))           # array for "selected" baselines

            for i in range(nbx):     # identify distinct v-coords and fill uv_sel
                b = np.where(np.abs(uvx - a[i]) <= prec)
                c = np.unique(np.round(uvy[b], ndgt))
                nby = np.shape(c)[0] # number of distinct v-compoments
                for j in range(nby):
                    uv_sel = np.append(uv_sel, [[a[i],c[j]]], axis=0)

        self.nbuv = np.shape(uv_sel)[0]/2 # actual number of distinct uv points
        self.uv   = uv_sel[:self.nbuv,:]  # discard second half (symmetric)
        print "%d distinct baselines were identified" % (self.nbuv,)

        # 1.5. Special case for undersampled data
        # ---------------------------------------
        if (Ns < 2):
            uv_sampl = self.uv.copy()   # copy previously identified baselines
            uvm = np.abs(self.uv).max() # max baseline length
            keep = (np.abs(uv_sampl[:,0]) < (uvm*Ns/2.)) * \
                (np.abs(uv_sampl[:,1]) < (uvm*Ns/2.))
            self.uv = uv_sampl[keep]
            self.nbuv = (self.uv.shape)[0]

            print "%d baselines were kept (undersampled data)" % (self.nbuv,)

        # 2. Calculate the transfer matrix and the redundancy vector
        # [AL, 2014.05.22] keeping relations between uv points and sampling points
        # --------------------------------------------------------------
        self.TFM = np.zeros((self.nbuv, self.nbh), dtype=float) # matrix
        self.RED = np.zeros(self.nbuv, dtype=float)             # Redundancy
        # relations matrix (-1 = not connected. NB: only positive baselines are saved)
        self.uvrel=-np.ones((nbh,nbh),dtype='int') 									
        for i in range(self.nbuv):
            a=np.where((np.abs(self.uv[i,0]-uvx) <= prec) *
                       (np.abs(self.uv[i,1]-uvy) <= prec))
            for k in range(len(a[0])) :
                 self.uvrel[uvi[a][k],uvj[a][k]]=i	
                 #self.uvrel[uvj[a][k],uvi[a][k]]=i
																						
            self.TFM[i, uvi[a]] +=  1.0
            self.TFM[i, uvj[a]] -=  1.0
            self.RED[i]         = np.size(a)
        # converting to relations matrix
        											

        # 3. Determine the kernel-phase relations
        # ----------------------------------------

        # One sub-aperture is taken as reference: the corresponding
        # column of the transfer matrix is discarded. TFM is now a
        # (nbuv) x (nbh - 1) array.
        
        # The choice is up to the user... but the simplest is to
        # discard the first column, that is, use the first aperture
        # as a reference?

        self.TFM = self.TFM[:,1:] # cf. explanation
        self.TFM = np.dot(np.diag(1./self.RED), self.TFM) # experiment #[Al, 2014.05.12] Frantz's version									
        U, S, Vh = np.linalg.svd(self.TFM.T, full_matrices=1) 

        S1 = np.zeros(self.nbuv)
        S1[0:nbh-1] = S

        self.nkphi  = np.size(np.where(abs(S1) < 1e-3)) # number of Ker-phases
        KPhiCol     = np.where(abs(S1) < 1e-3)[0]
        self.KerPhi = np.zeros((self.nkphi, self.nbuv)) # allocate the array

        for i in range(self.nkphi):
            self.KerPhi[i,:] = (Vh)[KPhiCol[i],:]

        print '-------------------------------'
        print 'Singular values for this array:\n', np.round(S, ndgt)
        print '\nRedundancy Vector:\n', self.RED
        self.name = array_name

    # =========================================================================
    # =========================================================================

    def plot_pupil_and_uv(self, xymax = 8.0):
        ''' Nice plot of the pupil sampling and matching uv plane.

        --------------------------------------------------------------------
        xymax just specifies the size of the region represented in the plot,
        expressed in meters. Should typically be slightly larger than the 
        largest baseline in the array.
        --------------------------------------------------------------------'''

        plt.clf()
        f0 = plt.subplot(121)
        f0.plot(self.mask[:,0], self.mask[:,1], 'bo')
        f0.axis([-xymax, xymax, -xymax, xymax], aspect='equal')
        plt.title(self.name+' pupil')
        f1 = plt.subplot(122)

        f1.plot(self.uv[:,0],   self.uv[:,1], 'bo') # plot baselines + symetric
        f1.plot(-self.uv[:,0], -self.uv[:,1], 'ro') # for a "complete" feel
        plt.title(self.name+' uv coverage')
        f1.axis([-2*xymax, 2*xymax, -2*xymax, 2*xymax], aspect='equal')


        # complete previous plot with redundancy of the baseline
        # -------------------------------------------------------
        dy = 0.1*abs(self.uv[0,1]-self.uv[1,1]) # to offset text in the plot.
        for i in range(self.nbuv):
            f1.text(self.uv[i,0]+dy, self.uv[i,1]+dy, 
                    int(self.RED[i]), ha='center')
        
        f0.axis('equal')
        f1.axis('equal')
        #plt.draw()


    # =========================================================================
    # =========================================================================

    def save_to_file(self, file):
        ''' Export the KerPhase_Relation data structure into a pickle
        
        ----------------------------------------------------------------
        To save on disk space, this procedure uses the gzip module.
        While there is no requirement for a specific extension for the
        file, I would recommend that one uses ".kpi.gz", so as to make
        it obvious that the file is a gzipped kpi data structure.
        ----------------------------------------------------------------  '''
        try: 
            data = {'name'   : self.name,
                    'mask'   : self.mask,
                    'uv'     : self.uv,
                    'TFM'    : self.TFM,
                    'KerPhi' : self.KerPhi,
                    'RED'    : self.RED,
                    'uvrel'    : self.uvrel}																				
        except:
            print("KerPhase_Relation data structure is incomplete")
            print("File %s wasn't saved!" % (file,))
            return None
        # -------------
        try: myf = gzip.GzipFile(file, "wb")
        except:
            print("File %s cannot be created."+
                  " KerPhase_Relation data structure wasn't saved." % (file,))
            return None
        # -------------
        pickle.dump(data, myf, -1)
        myf.close()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
