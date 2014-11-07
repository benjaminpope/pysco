import numpy as np
from scipy.optimize import leastsq
from core import *
from fitting import *
from grid import *
from limits import *
from kpo import *
from kpi import *
import sys
from numpy.random import rand, randn
from random import choice, shuffle
    
# =========================================================================
# =========================================================================
def super_cal_coeffs(src, cal, model=None, regul="None"):
    ''' Determine the best combination of vectors in the "cal" array of
    calibrators for the source "src". 

    Regularisation is an option:
    - "None"         ->  no regularisation
    - anything else  ->  Tikhonov regularisation  '''

    A      = np.matrix(cal.kpd).T # vector base matrix
    ns     = A.shape[1]           # size of the vector base
    b      = src.kpd              # column vector
    if model != None: b -= model  # optional model subtraction

    if regul == "None":
        coeffs = np.dot(np.linalg.pinv(np.dot(A.T,A)),
                        np.dot(A.T,b).T)
    else:
        coeffs = np.dot(np.linalg.pinv(np.dot(A.T,A)+np.identity(ns)),
                        np.dot(A.T,b).T)
    return coeffs.T

# =========================================================================
# =========================================================================
def cal_search(src, cal, regul="None"):
    ''' Proceed to an exhaustive search of the parameter space.

    In each point of the space, the best combination of calibrator frames
    is found, and saved. Not entirely sure what this is going to be useful
    for...
    '''
    
    ns, s0, s1 = 50, 20.0, 200.0
    nc, c0, c1 = 50,  2.0, 100.0
    na, a0, a1 = 60,  0.0, 360.0
    
    seps = s0 + np.arange(ns) * (s1-s0) / ns
    angs = a0 + np.arange(na) * (a1-a0) / na
    cons = c0 + np.arange(nc) * (c1-c0) / nc

    coeffs = super_cal_coeffs(src, cal, None, "tik")
    cals = np.zeros((ns, na, nc, coeffs.size))

    for i,sep in enumerate(seps):
        for j, ang in enumerate(angs):
            for k, con in enumerate(cons):
                model  = binary_model([sep, ang, con], src.kpi, src.hdr)
                coeffs = super_cal_coeffs(src, cal, model, "tik")
                cals[i,j,k] = coeffs

    return cals