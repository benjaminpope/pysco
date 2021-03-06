''' --------------------------------------------------------------------
              PYMASK: Python aperture masking analysis pipeline
    --------------------------------------------------------------------
    ---
    pymask is a python module for fitting models to aperture masking
    data reduced to oifits format by the IDL masking pipeline.

    It consists of a class, cpo, which stores all the relevant information
    from the oifits file, and a set of functions, cp_tools, for manipulating
    these data and fitting models.

    Fitting is based on the MCMC Hammer algorithm (aka ensemble affine 
    invariant MCMC) or the MultiNest algorithm (aka multimodal nested
    sampling). Both of these must be installed correctly or else 
    pymask won't work!
    See readme.txt for more details.

    - Ben
    ---
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
import pdb
import oifits

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi/180.0

import cp_tools
from cp_tools import *

import cpo
from cpo import *

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
#plt.set_cmap(cm.gray)
(plt.rcParams)['image.origin']        = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

plt.close()
