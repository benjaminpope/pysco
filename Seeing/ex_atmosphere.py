# ---------------------------------------------------- ex_atmosphere.py ----------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains an example of atmospheric seeing modelling ----------------------------------
# ====================================================================================================================
from phasescreen import phaseScreen
from phasescreen import getPhasesEvolution

from display import displayImages
from display import displayImage

from pupil import getNHolesMask
from pupil import getAnnulusMask
from pupil import getFullMask

from sim import getFocalImage
from sim  import getFocalImages

from common_tasks import mas2rad

import numpy as np

pupilSize=5.0		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
scale=100.0 		# scale factor (pixels/m)
v=10.0			# wind velocity (m/s)
sfrq=2			# number of samples per second
stime=2.0		# desired sampling time (0 = maximum)
atmosphereSize=10.0	# atmosphere patch size in meters
plateScale=11.5		# plate scale (mas/pixel)
wl=1e-6			#base wavelength
chip_px=512			# number of elements per chip (1dim)
exp_time=0.1			# exposure time in seconds


#----------old--------
# generating phaseScreen
#wl_delay=phaseScreen(atmosphereSize,scale,r0=0.2,seed=0,ao=0,maxVar=1e-5)
#wl_delay=phaseScreen(atmosphereSize,scale,r0=0.2,seed=0,ao=0,maxVar=0)
# converting delay to phases from 0 to 2*pi
#phases=delayToPhase(wl_delay,wl=wl)
#---------------------
# --- simulating athmosphere
# actuators number is for scale=100.0	
act_num=394 # number of actuators per aperture (strehl=0.6)
# act_num=193 # number of actuators per aperture (strehl=0.4)
# act_num=99 # number of actuators per aperture (strehl=0.2)
# act_num=65 # number of actuators per aperture (strehl=0.1)
# act_num=47 # number of actuators per aperture (strehl=0.05)
# act_num=34 # number of actuators per aperture (strehl=0.02)
# act_num=28 # number of actuators per aperture (strehl=0.01)
ao=np.sqrt(act_num)/pupilSize # actuators density
# generating phaseScreen
phases=phaseScreen(atmosphereSize,scale,r0=0.2,seed=0,ao=0,maxVar=0)
#generating evolution
pupilScreens=getPhasesEvolution(phases,pupilSize,scale,v,sfrq,stime,expTime=exp_time)
# input Image
pupil = np.ones((scale*pupilSize,scale*pupilSize),dtype='complex') + 0j # plane wave
#input N holes Mask 
mask_n=getNHolesMask(diam=pupilSize,holeDiam=0.5*pupilSize,holeCoords=[0.,0.,0.2*pupilSize,0.2*pupilSize],border=0.0,scale=scale)
#input annulus Mask 
mask_ann=getAnnulusMask(diam=pupilSize,innDiam=0.9*pupilSize,border=0.0,scale=scale)
#input full pupil Mask 
mask_full=getFullMask(diam=pupilSize,border=0.0,scale=scale)
#load jwst mask
mask_jwst=np.load('jwst_1000.npy')
# golay 9 mask
mask_golay9=getNHolesMask(diam=pupilSize*2,holeDiam=0.01*pupilSize,
	holeCoords=[-2.7,-1.56,-2.7,1.56,-1.35,0.78,1.35,-3.9,1.35,-2.34,1.35,3.9,2.7,1.56,4.05,-2.34,4.05,2.34],border=0.0,scale=scale)

#activeMask
mask=mask_n

# scale factor for FFT
#scale_f=1.22*wl/(np.power(pupilSize,2)*mas2rad(plateScale)*scale)
scale_f=wl/(pupilSize*mas2rad(plateScale))

#propagating (1 image)

p=getFocalImage(pupil,mask,pupilScreens[0],scale=scale_f,cropPix=chip_px)
displayImage(p**0.1,
	axisSize=[-plateScale*len(p)/2,plateScale*len(p)/2,-plateScale*len(p)/2,plateScale*len(p)/2],
	xlabel='mas', ylabel='mas', title='Power Spectrum **0.1',showColorbar=True,flipY=True,cmap='gray') 

# display atmosphere
displayImage(phases,axisSize=[-pupilSize/2,pupilSize/2,-pupilSize/2,pupilSize/2],xlabel='m', ylabel='m', title='Pupil Phase Screen',showColorbar=True,flipY=True) 

#display mask
displayImage(mask, axisSize=[-pupilSize/2,pupilSize/2,-pupilSize/2,pupilSize/2], 
	xlabel='m', ylabel='m', title='Pupil Mask',showColorbar=False,flipY=True, cmap='binary_r') 



# propagating (all images)
ps=getFocalImages(pupil,mask,pupilScreens,scale=scale_f,cropPix=chip_px)
displayImages(ps**0.1,delay=0.3,
	axisSize=[-plateScale*len(p[0])/2,plateScale*len(p[0])/2,-plateScale*len(p[0])/2,plateScale*len(p[0])/2],
	xlabel='mas', ylabel='mas', title='Power Spectrum **0.1',figureNum=1,showColorbar=True,flipY=True,cmap='gray') 
	
	
