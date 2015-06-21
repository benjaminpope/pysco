from phasescreen import phaseScreen
from phasescreen import getPhasesEvolution

from display import displayImage

from sim import getFocalImage
from sim import getFocalImages
from sim import imageToFits
from sim import normalizeImage
from sim import normalizeImages

from common_tasks import mas2rad
from common_tasks import rad2mas

from display import displayImage

import datetime

import numpy as np
import sys



pupilSize=7.77		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
v=10.0			# wind velocity (m/s)
atmosphereSize=15.0	# atmosphere patch size in meters
plateScale=14.1		# plate scale (mas/pixel)
scale=65.9		# scale factor (pixels/m)
wl=2.1e-6			#base wavelength

chip_px=256		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)

exp_time=0.0		# exposure time in seconds
expCoeff=0.01
exp_dens=200		# number of exposure screens per second	
sfrq=1.			# number of samples per second
stime=0.1			# desired sampling time (0 = maximum)
nFrames=1		# 10 frames in a datacube (from 10 different atmospheres)


ddir = '/import/pendragon1/latyshev/Data/KerPhases/'
s_dir='sampling_points/'
mask_dir='masks/'
fits_dir='outputs/temp/PSF/'
s_ext='.txt'
mask_ext='.npy'
fits_ext='.fits'

names=[]
names.append(('gpi12_10','gpi12_10'))
active=range(0,len(names))



# --- Loading data -----------
# data structure: (mask, sampling points ,mask name, sampling points name)
data=[]
for i in active :
	data.append((np.load(ddir+mask_dir+names[i][0]+mask_ext),np.loadtxt(ddir+s_dir+names[i][1]+s_ext),names[i][0],names[i][1]))
		
scale_f=wl/(pupilSize*mas2rad(plateScale))
# input Image
pupil = np.ones((int(scale*pupilSize),int(scale*pupilSize)),dtype='complex') + 0j # plane wave
	
# --- simulating athmosphere
act_num=498 # number of actuators per aperture (strehl=0.6, D=7.77)
act_num=1318 # number of actuators per aperture (strehl=0.8, D=7.77)

ao=np.sqrt(act_num)/pupilSize # actuators density
#aopower=0.083
# we generate nFrames different atmospheres to get errors for kernel phases later
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

i=0
res=getFocalImages(pupil,data[i][0],screens,scale=scale_f,cropPix=chip_px,photonNoise=False,expCoeff=expCoeff,processes=-1)
dt=datetime.datetime.now()
imageToFits(res,path=ddir+fits_dir,filename=data[i][2]+fits_ext,
	tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
	tint=exp_time,filter=wl)
												

	