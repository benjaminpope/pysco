# ---------------------------------------------------- ex_atmosphere2.py ---------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# -------------- This file contains an example of atmospheric seeing modelling using common_tasks lib-----------------
# ====================================================================================================================
from sim import getDistFocalImage
from sim import getDistFocalImages
from display import displayImage

from pupil import getNHolesMask
from pupil import getAnnulusMask
from pupil import getFullMask

from phasescreen import phaseScreen

from common_tasks import mas2rad

import numpy as np


pupilSize=5.0		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
scale=100.0 		# scale factor (pixels/m)
v=1.0			# wind velocity (m/s)
sfrq=2			# number of samples per second
stime=2.0		# desired sampling time (0 = maximum)
atmosphereSize=10.0	# atmosphere patch size in meters
plateScale=10.0		# plate scale (mas/pixel)
wl=1e-6			#base wavelength
chip_px=512			# number of elements per chip (1dim)
exp_time=0.1			# exposure time in seconds

# atmosphere
#wl_delay=phaseScreen(atmosphereSize,scale,r0=0.4,seed=0,ao=0)
# atmosphere
phases=phaseScreen(atmosphereSize,scale,r0=0.2,seed=0,ao=0)*10

# masks
#input N holes Mask 
mask_n=getNHolesMask(diam=pupilSize,holeDiam=0.1*pupilSize,holeCoords=[0.,0.,0.2*pupilSize,0.2*pupilSize],border=0.0,scale=scale)
#input annulus Mask 
mask_ann=getAnnulusMask(diam=pupilSize,innDiam=0.9*pupilSize,border=0.0,scale=scale)
#input full pupil Mask 
mask_full=getFullMask(diam=pupilSize,border=0.0,scale=scale)
#load jwst mask
mask_jwst=np.load('jwst_1000.npy')

#activeMask
mask=mask_n

# image
image = np.ones((scale*pupilSize,scale*pupilSize),dtype='complex') + 0j

# scale factor for FFT
#scale_f=1.22*wl/(np.power(pupilSize,2)*mas2rad(plateScale)*scale)
scale_f=wl/(pupilSize*mas2rad(plateScale))

# p=getDistFocalImage(image,mask,wl_delay,pupilSize,scale,v,exp=exp_time)

p=getDistFocalImage(image,mask,
	phases,pupilSize,scale,plateScale,detectorSize=chip_px,v=1.0,t=0.0,
	wl=wl,expTime=exp_time)

displayImage(p**0.1,
	axisSize=[-plateScale*len(p)/2,plateScale*len(p)/2,-plateScale*len(p)/2,plateScale*len(p)/2],
	xlabel='mas', ylabel='mas', title='Power Spectrum **0.1',showColorbar=True,flipY=True,cmap='gray') 


ps=getDistFocalImages(image,mask,
	phases,pupilSize,scale,plateScale,detectorSize=chip_px,
	v=v,sfrq=10,stime=2.0,wl=wl,expTime=exp_time,
	display=True,delay=0.4,displayFactor=0.2,returnData=True)

