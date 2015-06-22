''' --------------------------------------------------------------------
    PYSCOSOUR: Python Self Calibrating Observables with a Shot of Pymask
    --------------------------------------------------------------------
    ---
    pyscosour is a python module for fitting models to aperture masking
    data reduced to oifits format by the IDL masking pipeline, or 
    for reducing and fitting to kernel phase data from filled pupils.

    It consists of classes, cpo, kpo and kpi, which store all the relevant 
    information from the dataset, and a set of functions, namely fitting 
    and core, for manipulating these data and fitting models.

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
# import pdb
# import oifits

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi/180.0

import kpo
from kpo import *

import kpi
from kpi import *

import core
from core import *

import fitting
from fitting import *

import limits
from limits import *

import grid
from grid import *

import calibration
from calibration import *

# import diffract_tools 
# from diffract_tools import *

# -------------------------------------------------
# set some defaults to display images that will
# look more like the DS9 display....
# -------------------------------------------------
#plt.set_cmap(cm.gray)
(plt.rcParams)['image.origin']        = 'lower'
(plt.rcParams)['image.interpolation'] = 'nearest'
# -------------------------------------------------

plt.close()
