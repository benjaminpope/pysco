# ---------------------------------------------------- task_binaries.py -----------------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# This file contains code to generate binaries from existing single stars
# ====================================================================================================================
import numpy as np
import pyfits as pf

from display import displayImage
from sim import simBinary
from sim import imageToFits

import datetime

ddir='/import/pendragon1/latyshev/Data/KerPhases/'
in_dir='outputs/frozen_noiseless500/PSF/'
out_dir='outputs/frozen_noiseless500/PSF/binaries/'
in_file='golay9_scale50_0.6.fits'
out_file='golay9_0.6_scale50_frozen_s135_c5_a0_500.fits'


pupilSize=8.0		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
plateScale=27.0		# plate scale (mas/pixel)
scale=100.0			# scale factor (pixels/m)
wl=2.6e-6			#base wavelength

#chip_px=1038		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)

exp_time=0.0		# exposure time in seconds
	
strehl=0.6
contrast=5.
sepPx=5 # approx 2 lambda/D
angle=0.


'''
im=pf.getdata(ddir+in_dir+in_file)
im1=np.zeros(np.shape(im))
for i in range(0,im.shape[0]) :
	im1[i]=simBinary(im[i], c=contrast, sep=sepPx, ang=angle, lambdaD=0., pscale=0., forceInt=True)

dt=datetime.datetime.now()
imageToFits(im1,path=ddir+out_dir,filename=out_file,
           tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
		tint=exp_time,filter=wl)
'''

#for s in ['no_ao',0.01'','0.02','0.05','0.1','0.2','0.4','0.6'] :
for s in ['0.02','0.05','0.1','0.2','0.4','0.6','0.8','0.9'] :
	in_file='full_hex15_'+s+'.fits'
	out_file='full_hex15_frozen_'+s+'_s135_c5_a0_7000.fits'	
	im=pf.getdata(ddir+in_dir+in_file)
	im1=np.zeros(np.shape(im))
	for i in range(0,im.shape[0]) :
		im1[i]=simBinary(im[i], c=contrast, sep=sepPx, ang=angle, lambdaD=0., pscale=0., forceInt=True)
	dt=datetime.datetime.now()
	imageToFits(im1,path=ddir+out_dir,filename=out_file,
           tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
		tint=exp_time,filter=wl)
		
data=[]
data.append(('full_hex15','full_hex15','full_hex15_scale50','full_hex15_scale50'))
data.append(('ann_hex15','ann_hex15','ann_hex15_scale50','ann_hex15_scale50'))
data.append(('ann_hex15_w05','ann_hex15_w05','ann_hex15_w05_scale50','ann_hex15_w05_scale50'))
data.append(('golay9','golay9','golay9_scale50','golay9_scale50'))
# lines in data array to analyse
active = range(0,len(data))	

for s in ['no_ao','0.05','0.1','0.2','0.4','0.6','0.8','0.9'] :
	for num in active :
		in_file=data[num][3]+'_'+s+'.fits'
		out_file=data[num][3]+'_'+s+'_s135_c5_a0_500.fits'
		print(in_file)
		im=pf.getdata(ddir+in_dir+in_file)
		im1=np.zeros(np.shape(im))
		for i in range(0,im.shape[0]) :
			im1[i]=simBinary(im[i], c=contrast, sep=sepPx, ang=angle, lambdaD=0., pscale=0., forceInt=True)
		dt=datetime.datetime.now()
		imageToFits(im1,path=ddir+out_dir,filename=out_file,
				tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
				tint=exp_time,filter=wl)
