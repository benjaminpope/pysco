# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 11:43:28 2014

@author: latyshev
"""
import numpy as np

from common_tasks import mas2rad
from common_tasks import rad2mas

from phasescreen import phaseScreen
from phasescreen import getPhasesEvolution
from display import displayImage
from sim import getFocalImages
from sim import imageToFits

from sim import simBinary

import datetime
import sys



scale=100.
plateScale=25.0
pupilSize=5.093
innSize=1.829


atmosphereSize=15.
v=10.0
#chip_px=1038		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)
chip_px=512		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)

exp_time=1.416		# exposure time in seconds
exp_dens=1		# number of exposure screens per second	
sfrq=0.5			# number of samples per second
stime=0.1			# desired sampling time (0 = maximum)
wl_ch=1.57e-6 #CH4_s
wl_ks=2.15e-6 #ks band 

ddir = '/import/pendragon1/latyshev/Data/KerPhases/'
mask_dir='masks/'
fits_dir='outputs/temp/PSF/'
name='med_cross_hex'
mask_ext='.npy'
fits_ext='.fits'

mask=np.load(ddir+mask_dir+name+mask_ext)

wl=wl_ch
scale_f=wl/(pupilSize*mas2rad(plateScale))
# input Image
pupil = np.ones((int(scale*pupilSize),int(scale*pupilSize)),dtype='complex') + 0j # plane wave

act_num=483	#0.8 strehl
# act_num=128	#0.5 strehl
# act_num=92	#0.4 strehl
ao=np.sqrt(act_num)/pupilSize # actuators density
nFrames=1 #number of phasescreens

s=[]
screens=[]
for i in range(0,nFrames) :
	print("---Processing atmosphere %d (%d total)---" % (i+1,nFrames))
	phases=phaseScreen(atmosphereSize,scale,r0=0.2,seed=i,ao=ao) 
	s.append(np.exp(-(phases-phases.mean()).std()**2))# current strehl
	#generating phasescreen over exposure time for current atmosphere
	pupilScreens=getPhasesEvolution(phases,pupilSize,scale,v,sfrq,stime,expTime=exp_time,expNum=int(exp_dens*exp_time))
	screens.append(pupilScreens[0])
	
s=np.asarray(s)
screens=np.asarray(screens)	
print ("Mean strehl=%f SD=%f" % (s.mean(),s.std()))



res=getFocalImages(pupil,mask,screens,scale=scale_f,cropPix=chip_px,photonNoise=False,processes=-1)
dt=datetime.datetime.now()


#Saving single image or datacube
imageToFits(res,path=ddir+fits_dir,filename=name+fits_ext,
	tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
	tint=exp_time,filter=wl)
												
#binary
contrast=5.
sepPx=5 # approx 2 lambda/D
angle=0.


im1=np.zeros(np.shape(res))

# please see simBinary for different options including sepatation in lambda/D
im1=simBinary(res, c=contrast, sep=sepPx, ang=angle, lambdaD=0., pscale=0., forceInt=True)									
# Saving is similar to the previos case