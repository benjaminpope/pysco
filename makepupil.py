import numpy as np 
import matplotlib.pyplot as plt 
import whisky as pysco
import time

'''----------------------------------------------------
makepupil.py - generate a pupil model
----------------------------------------------------'''

ddir = './geometry/'
pupil = 'medcross'
kpdir = './kerphi/'

'''----------------------------------------------------
Load the data as a row-phase object
----------------------------------------------------'''
tic = time.time()
a = pysco.kpi(ddir+pupil+'.txt')
toc = time.time()

print 'Took', toc-tic,'seconds'

a.save_to_file(kpdir+pupil+'model.pick')