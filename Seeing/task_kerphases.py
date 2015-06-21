# ---------------------------------------------------- model1.py -----------------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# This file contains code to generate pupils, sampling points and atmosphere modelling for kernel phases analysis task
# ====================================================================================================================

from phasescreen import phaseScreen
from phasescreen import getPhasesEvolution

from display import displayImage

from pupil import getNHolesMask
from pupil import getAnnulusMask
from pupil import getFullMask
from pupil import getSamplingPoints
from pupil import getCircularSamplingPoints
from pupil import getSquareSamplingPoints
from pupil import placeSamplingPoints
from pupil import getAnnulusHexMask
from pupil import getFullHexMask

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


# Here we take NACO telescope parameters

pupilSize=8.0		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
v=10.0			# wind velocity (m/s)
atmosphereSize=15.0	# atmosphere patch size in meters
plateScale=27.0		# plate scale (mas/pixel)
scale=100.0			# scale factor (pixels/m)
wl=2.6e-6			#base wavelength

#chip_px=1038		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)
chip_px=256		# number of elements per chip (1dim) (scaled to work correctly with kPhases code)

exp_time=0.5		# exposure time in seconds
exp_dens=200		# number of exposure screens per second	
sfrq=1.			# number of samples per second
stime=0.1			# desired sampling time (0 = maximum)
nFrames=10			# 10 frames in a datacube (from 10 different atmospheres)
	
strehl=0.6
holeD=0.92	 		# diameter of the hole
obscD=1.116		 	# central obscuration area
#n_dens=pupilSize/holeD/np.cos(np.pi/6)		# number of sampling points across the diameter
								# here we assume two points can be as close as centers
								# of hexagon of outer circle diameter = holeD
n_dens=pupilSize/holeD					# number of sampling points across the diameter
s_dens=(n_dens/pupilSize)	# sampling density (number of points per length)

ddir = '/import/pendragon1/latyshev/Data/KerPhases/'
s_dir='sampling_points/'
mask_dir='masks/'
fits_dir='outputs/temp/PSF/'
s_ext='.txt'
mask_ext='.npy'
fits_ext='.fits'

																																			
#holeCoords=holeCoords/np.abs(holeCoords).max()*(pupilSize/2*(n_dens-1)/n_dens)
'''	
# -----------  Generating masks and sampling points	------------------	
# golay 9 mask with holes diameter = pupilSize/n_dens
s_golay9=np.array([[-2.07,-2.78860],
			    [2.07,-2.78860],
                     [-1.38000,-1.59349],
                     [-3.45000,-0.398372],
                     [2.07000,-0.398372],
                     [3.45000,-0.398372],
                     [-0.690000,1.99186],
                     [-1.38000,3.18697],
                     [1.38000,3.18697]])
s_golay9=np.asarray(s_golay9)
																																					
mask_golay9=getNHolesMask(diam=pupilSize,holeDiam=holeD,
	holeCoords=s_golay9,border=0.0,scale=scale)	
	
mask_ann=getAnnulusMask(diam=pupilSize,innDiam=pupilSize-holeD*2,border=0.0,scale=scale)
mask_ann_w05=getAnnulusMask(diam=pupilSize,innDiam=pupilSize-holeD,border=0.0,scale=scale)

s_ann=getCircularSamplingPoints(mask_ann,dens=s_dens,scale=scale,prec=2)
s_ann15=getCircularSamplingPoints(mask_ann,dens=s_dens*1.5,scale=scale,prec=2)
s_ann15_w05=getCircularSamplingPoints(mask_ann_w05,dens=s_dens*1.5,scale=scale,prec=2)
s_ann_lin=getSamplingPoints(mask_ann,dens=s_dens,scale=scale,prec=2)
s_ann_lin15=getSamplingPoints(mask_ann,dens=s_dens*1.5,scale=scale,prec=2)

mask_full=getFullMask(diam=pupilSize,border=0.0,scale=scale)
s_full=getCircularSamplingPoints(mask_full,dens=s_dens,scale=scale,prec=2)
s_full15=getCircularSamplingPoints(mask_full,dens=s_dens*1.5,scale=scale,prec=2)
s_full_lin=getSamplingPoints(mask_full,dens=s_dens,scale=scale,prec=2)
s_full_lin15=getSamplingPoints(mask_full,dens=s_dens*1.5,scale=scale,prec=2)

mask_ann_hex,s_ann_hex=getAnnulusHexMask(diam=pupilSize,innDiam=pupilSize-holeD*2,hexDiam=holeD,border=0.0,scale=scale,returnSampling=True)
mask_full_hex,s_full_hex=getFullHexMask(diam=pupilSize,hexDiam=holeD,border=0.0,scale=scale,returnSampling=True)

mask_ann_hex15,s_ann_hex15=getAnnulusHexMask(diam=pupilSize,innDiam=pupilSize-holeD*2,hexDiam=holeD/1.5,border=0.0,scale=scale,returnSampling=True)
mask_ann_hex15_w05,s_ann_hex15_w05=getAnnulusHexMask(diam=pupilSize,innDiam=pupilSize-holeD*1.15,hexDiam=holeD/1.5,border=0.0,scale=scale,returnSampling=True)
mask_ann_hex20_w05,s_ann_hex20_w05=getAnnulusHexMask(diam=pupilSize,innDiam=pupilSize-holeD*1.18,hexDiam=holeD/2.,border=0.0,scale=scale,returnSampling=True)

mask_full_hex15,s_full_hex15=getFullHexMask(diam=pupilSize,hexDiam=holeD/1.5,border=0.0,scale=scale,returnSampling=True)

# --------- saving masks --------------
np.save(ddir+mask_dir+'golay9',mask_golay9)

np.save(ddir+mask_dir+'ann',mask_ann)
np.save(ddir+mask_dir+'ann_w05',mask_ann_w05)
np.save(ddir+mask_dir+'full',mask_full)

np.save(ddir+mask_dir+'ann_hex',mask_ann_hex)
np.save(ddir+mask_dir+'full_hex',mask_full_hex)

np.save(ddir+mask_dir+'ann_hex15',mask_ann_hex15)
np.save(ddir+mask_dir+'ann_hex15_w05',mask_ann_hex15_w05)
np.save(ddir+mask_dir+'full_hex15',mask_full_hex15)
# ---------------------------------------

# ------- Saving sampling points --------

np.savetxt(ddir+s_dir+'golay9'+s_ext,s_golay9)

#np.savetxt(ddir+s_dir+'ann2'+s_ext,s_ann2)

np.savetxt(ddir+s_dir+'full'+s_ext,s_full)
np.savetxt(ddir+s_dir+'full15'+s_ext,s_full15)
np.savetxt(ddir+s_dir+'ann'+s_ext,s_ann)
np.savetxt(ddir+s_dir+'ann15'+s_ext,s_ann15)
np.savetxt(ddir+s_dir+'ann15_w05'+s_ext,s_ann15_w05)

np.savetxt(ddir+s_dir+'full_lin'+s_ext,s_full_lin)
np.savetxt(ddir+s_dir+'full_lin15'+s_ext,s_full_lin15)
np.savetxt(ddir+s_dir+'ann_lin'+s_ext,s_ann_lin)
np.savetxt(ddir+s_dir+'ann_lin15'+s_ext,s_ann_lin15)

np.savetxt(ddir+s_dir+'full_hex'+s_ext,s_full_hex)
np.savetxt(ddir+s_dir+'full_hex15'+s_ext,s_full_hex15)
np.savetxt(ddir+s_dir+'ann_hex'+s_ext,s_ann_hex)
np.savetxt(ddir+s_dir+'ann_hex15'+s_ext,s_ann_hex15)
np.savetxt(ddir+s_dir+'ann_hex15_w05'+s_ext,s_ann_hex15_w05)
'''

# -------- data structure is the following: -----------
# -------- (maskname,sampling_txt) -----------
names=[]

names.append(('golay9','golay9'))
#names.append(('ann','ann2'))

names.append(('ann','ann'))
#names.append(('full','full'))
#names.append(('ann','ann15'))
#names.append(('full','full15'))


names.append(('full','full_lin'))
names.append(('full','full_lin15'))
#names.append(('ann','ann_lin'))
#names.append(('ann','ann_lin15'))

names.append(('ann_hex','ann_hex'))
names.append(('full_hex','full_hex'))
names.append(('ann_hex15','ann_hex15'))
names.append(('ann_hex20','ann_hex20'))
#names.append(('ann_hex30','ann_hex30'))

names.append(('full_hex15','full_hex15'))
names.append(('full_hex20','full_hex20'))
#names.append(('full_hex30','full_hex30'))

names.append(('ann_w05','ann'))
names.append(('ann_w05','ann15_w05'))
names.append(('ann_hex15_w05','ann_hex15_w05'))
names.append(('ann_hex20_w05','ann_hex20_w05'))

names=[]
names.append(('golay9','golay9'))
names.append(('ann_hex15','ann_hex15'))
names.append(('full_hex15','full_hex15'))
names.append(('ann_hex15_w05','ann_hex15_w05'))

names=[]
names.append(('golay9_scale50','golay9'))
names.append(('ann_hex15_scale50','ann_hex15'))
names.append(('full_hex15_scale50','full_hex15'))
names.append(('ann_hex15_w05_scale50','ann_hex15_w05'))

active=range(0,len(names))


data=[]
# --- Loading data -----------
# data structure: (mask, sampling points ,mask name, sampling points name)
for i in active :
	data.append((np.load(ddir+mask_dir+names[i][0]+mask_ext),np.loadtxt(ddir+s_dir+names[i][1]+s_ext),names[i][0],names[i][1]))
	
'''	
# displaying data
num=0
for i in active :
	print("%s mask + %s sampling points" % (names[i][0],names[i][1]))
	displayImage(placeSamplingPoints(data[num][0],data[num][1],scale))
	num+=1
'''	
# scale factor for FFT
scale_f=wl/(pupilSize*mas2rad(plateScale))
# input Image
pupil = np.ones((int(scale*pupilSize),int(scale*pupilSize)),dtype='complex') + 0j # plane wave
	
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

# ---- propagating ----
mask_names,idx=np.unique(np.asarray(data)[:,2],return_index=True)
num=0
for i in idx :
	print("Processing %s mask (%d of %d)" % (data[i][2],num+1,len(idx)))
	res=getFocalImages(pupil,data[i][0],screens,scale=scale_f,cropPix=chip_px,photonNoise=False)
	dt=datetime.datetime.now()
	imageToFits(normalizeImages(res),path=ddir+fits_dir,filename=data[i][2]+fits_ext,
		tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
		tint=exp_time,filter=wl)
	num+=1
	
#----------- Special experiment ------------
nFrames=500			# 10 frames		
coeff=1.0
atmosphereSize=15.
scale=50.
coeff=1.47

# scale factor for FFT
scale_f=wl/(pupilSize*mas2rad(plateScale))
# input Image
pupil = np.ones((int(scale*pupilSize),int(scale*pupilSize)),dtype='complex') + 0j # plane wave

for atm in ['frozen'] :
	if atm=='frozen' :
		exp_time=0.0
		expCoeff=0.01
		exp_dens=200
	elif	atm=='3t0' :
		exp_time=0.06
		expCoeff=0.12
		exp_dens=200
	elif	atm=='25t0' :
		exp_time=0.5
		expCoeff=1.0
		exp_dens=80
	elif	atm=='100t0' :
		exp_time=2.0
		expCoeff=4.0
		exp_dens=200	
	screens=np.zeros((nFrames,max(int(exp_dens*exp_time),1),int(pupilSize*scale),int(pupilSize*scale)),dtype='float')
	#for s in [0,0.1,0.2,0.4,0.6] :
	for s in [0,0.05,0.1,0.2,0.4,0.6] :
		fits_dir='outputs/'+atm +'_noiseless'+str(nFrames)+'_cut/PSF/'
		if s==0 :
			suff='no_ao'
			ao=0.0
		else :
			suff=str(s)
			if s==0.01 :			
				act_num=28				
			if s==0.02 :			
				act_num=34				
			elif s==0.05 :			
				act_num=47			
			elif s==0.1 :			
				act_num=65
			elif s==0.2 :			
				act_num=99
			elif s==0.4 :			
				act_num=193
			elif s==0.6 :			
				act_num=394	
			elif s==0.8 :			
				act_num=1030	
			elif s==0.9 :			
				act_num=2480					
			ao=np.sqrt(act_num*coeff)/pupilSize
		noise=0.
		# we generate nFrames different atmospheres to get errors for kernel phases later
		s=[]
		#screens=[]
		print("\n")
		for i in range(0,nFrames) :
			sys.stdout.write("\r---Processing atmosphere %d (%d total)---" % (i+1,nFrames))
			#print("---Processing atmosphere %d (%d total)---" % (i+1,nFrames))
			phases=phaseScreen(atmosphereSize,scale,r0=0.2,seed=i,ao=ao,showStrehl=False,telSize=8.) 
			s.append(np.exp(-(phases-phases.mean()).std()**2))# current strehl
			#generating phasescreen over exposure time for current atmosphere
			#pupilScreens=getPhasesEvolution(phases,pupilSize,scale,v,sfrq,stime,expTime=exp_time,expNum=int(exp_dens*exp_time))
			#screens.append(pupilScreens[0])
			screens[i]=(getPhasesEvolution(phases,pupilSize,scale,v,sfrq,stime,expTime=exp_time,expNum=int(exp_dens*exp_time),showProgress=False))[0]
		s=np.asarray(s)
		#screens=np.asarray(screens)	
		print ("Mean strehl=%f SD=%f" % (s.mean(),s.std()))
		# ---- propagating ----
		mask_names,idx=np.unique(np.asarray(data)[:,2],return_index=True)
		num=0
		for i in idx :
			print("Processing %s mask (%d of %d)" % (data[i][2],num+1,len(idx)))
			res=getFocalImages(pupil,data[i][0],screens,scale=scale_f,cropPix=chip_px,photonNoise=False,expCoeff=expCoeff,processes=-1)
			dt=datetime.datetime.now()
			imageToFits(res,path=ddir+fits_dir,filename=data[i][2]+'_'+suff+fits_ext,
				tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),
				tint=exp_time,filter=wl)
			num+=1	
