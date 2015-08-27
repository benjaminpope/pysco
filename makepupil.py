import numpy as np 
import matplotlib.pyplot as plt 
import pysco
import time

'''----------------------------------------------------
makepupil.py - generate a pupil model
----------------------------------------------------'''

ddir = './geometry/'
pupil = 'medcrossmeas'
# pupil = 'wfc3_uvis2'
kpdir = './kerphi/'

'''----------------------------------------------------
Load the data as a row-phase object
----------------------------------------------------'''
tic = time.time()
a = pysco.kpi(ddir+pupil+'.txt',bsp_mat=None,Ns=3.0)
toc = time.time()

print 'Took', toc-tic,'seconds'

a.save_to_file(kpdir+pupil+'model.pick')

print 'Kernel phase structure saved to %s' % (kpdir+pupil+'model.pick')