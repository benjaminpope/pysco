# original test file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits as pf
import pickle
import gzip
#import pymultinest ###### Removed temporary ! #######
import os, threading, subprocess
import matplotlib.pyplot as plt
import json
import time

import whisky as pysco # for compatibility and ease of use

'''---------------------------------------------------------
test.py - a script for analysing masking/kernel phase data 
using either the MCMC Hammer ensemble affine invariant MCMC 
algorithm, or the MultiNest multimoded nested sampling 
algorithm. 

Comment out whichever you don't want!

This depends significantly on emcee, the Python MCMC Hammer,
and PyMultiNest, a wrapper for the Fortran-based MultiNest.
---------------------------------------------------------'''


ddir = '/suphys/latyshev/Data/KerPhases/'

txtdir='sampling_points/'
kpidir='kpi/'
fitsdir='single_star_images/'

kpifile='ann.kpi.gz'
kpifile_txt='sampling_ann.txt'
fitsfile = 'ann.fits'

#kpiname = ddir+'full_pupil.kpi.gz'

#------------------------
# first, load your data!
#------------------------

# creating kpi from txt file
#full_p_kpi = pysco.kpi(ddir+txtdir+kpifile_txt)
#full_p_kpi.save_to_file(ddir+kpidir+kpifile)

a = pysco.kpo(ddir+kpidir+kpifile,'ann')

a.extract_kpd(ddir+fitsdir+fitsfile,manual=64)

a.kpe *= 0.1

#----------------------------------------------
# Use this for MCMC Hammer
# Warning - get ivar right!
# Estimate visually or with multinest first.
#----------------------------------------------

ivar = [124, 265, 17.3] # this has to be pretty much near the peak or MCMC Hammer can get lost

chains = pysco.hammer(a,ndim=3,ivar=ivar,plot=True)

params = np.mean(chains,axis=0)

#--------------------------------------------------
# Use this for MultiNest
# - make sure you cover the whole parameter domain!
#--------------------------------------------------

#paramlimits = [25.,400.,0.,360.,1.0001,60] #sepmin,sepmax,anglemin,anglemax,cmin,cmax
#model = pysco.nest(a,paramlimits=paramlimits,multi=False)
#params = [entry['median'] for entry in model]

#------------------------------------
# Use this for detec limits
# Check your errors!
#------------------------------------

# limits = pysco.detec_limits(a,threads=8,smin=60,smax=250,cmax=10000,nsep=64,ncon=64)

#------------------------------------
# Get correlation diagram
#------------------------------------

pysco.correlation_plot(a,params)
plt.show()
