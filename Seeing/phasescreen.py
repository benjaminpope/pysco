# ------------------------------------------------------------ phasescreen.py ----------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains functions for generating atmosphere phasescreen and getting its evolution----
# ------------------- above a pupil---------------------------------------------- ------------------------------------
# ====================================================================================================================
import numpy as np
import sys
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------phaseScreen-------------------------------------------------------------
# generating phasescreen using Kolmogorov/von Karmen model 
# --------------------------------------------------------------------------------------------------------------------
# for reference see 
# Welsh, Byron M. 
# "Fourier-series-based atmospheric phase screen generator for simulating anisoplanatic geometries and temporal evolution."  
# Optical Science, Engineering and Instrumentation'97. International Society for Optics and Photonics, 1997. 
### input parameters
# size=1.0 		# size of the screen (m)
# scale=1024.0 	# scale factor (pixels/m)
# r0=0.3		# Fried parameter (m)
# seed = 0		# seed for random (-1 = no seed)
# L0=10.0		# outer scale in meters (-1 = infinity)
# ao = 10.0		# actuators density (number of actuators per metre) for AO correction. 0 if no correction
#strehl		# desired strehl ratio (inflate the result atmosphere)
'''
# The following parameters are not used anymore
# aopower=1.0    # power of ao amplitutes correction 
#fc=10.          # cutoff frequency (in lambda/D)
#lambdaD		#lambda/D value in pixels (=int(rad2mas(lambda/D)*pscale)+1)
#telSize= telescope diameter in meters to avoid tilt

#NB: if any of   aopower, fc, lambdaD parameters = 0 then no AO applied
'''
### output parameter
# phases  - array with phase shifts
#def phaseScreen(size,scale,r0=0.2,ao=0.0,L0=-1.0,seed=-1,aopower=50.0,fc=10.,lambdaD=0.,strehl=0.0) :
def phaseScreen(size,scale,r0=0.2,ao=0.0,L0=-1.0,seed=-1,strehl=0.0,showStrehl=True,telSize=0.) :
	''' generating phasescreen using Kolmogorov/von Karmen model  '''
	# size of screen in pixels
	Nxy=int(np.round(size*scale)) 
	#
	# Calculating center of the image
	Nc = Nxy //2
	#
	# initializing random generator 
	if seed != -1 :		
		np.random.seed(seed)
	#
	# generating random complex array size x size with normal distribution
	gaussc=np.random.randn(Nxy,Nxy)+1j*np.random.randn(Nxy,Nxy)
	#
	#generating spatial freqs (normalized)
	#freqs=(np.arange(Nxy,dtype=float)-Nc)/(2*size)
	#generating spatial freqs (normalized)
	freqs=(np.arange(Nxy,dtype=float)-Nc)/(2*scale*size)
	#
	#creating array (rho) with distances (spatial freqs)
	xx,yy  = np.meshgrid(freqs, freqs)
	rho = np.hypot(yy,xx)
	# avoiding division by zero
	if rho.min()==0:
		rho[rho.argmin()/len(rho),rho.argmin()-int(rho.argmin()/len(rho))*len(rho)]=1e-9		
	# Obtaining amplitutes from power spectrum density
	# correcting r0 to match the units
	r0_fixed=r0/float(size)
	if L0==-1 :
		asd=np.sqrt(0.023)*(np.power(r0_fixed*rho,-5.0/6)/rho)
	else : asd=np.sqrt(0.023)*(np.power(r0_fixed,-5.0/6))*pow((np.pow(rho,2)+pow(L0,2)),-11.0/12)
	# Removing infinity from psd
	asd[Nc,Nc]=0
	#
	# applying AO (supergaussian filter) if required
	# Section is replaced by Frantz's code
	'''	
	# Alexey's version
	if ao != 0 :
		filter = 1 - np.exp(-1.0*pow((rho*2*scale/ao),10.0))
		aopow=aopower
	else : 
		filter = np.ones(np.shape(asd))
		aopow=1.0		
	#
	#generating phasescreen 
	phases=np.fft.ifft2(np.fft.fftshift(asd*(filter*(aopow-1)/aopow+aopow)*gaussc)).real	
	# end
	'''
	'''
	# Frantz's version
	if (np.abs(aopower)+np.abs(lambdaD)+np.abs(fc)) >0 :
		in_fc = (rho <= (fc*lambdaD/(2*scale*size)))
		asd[in_fc]/=aopower #
	phases=np.fft.ifft2(np.fft.fftshift(asd*gaussc)).real	
	# end	
	'''
	# Peter's version 
	if ao>0 :
		in_fc= (rho<= (ao/(2*scale)))
		#flat spectrum inside
		asd[in_fc]=asd[True-in_fc].max()
		if telSize>0 :
			sz=int(np.ceil(size/telSize*2))
			asd[Nc-sz:Nc+sz+1,Nc-sz:Nc+sz+1]=0	
	phases=np.fft.ifft2(np.fft.fftshift(asd*gaussc)).real
	# end
	
	s = np.exp(-(phases-phases.mean()).std()**2)# current strehl
	if showStrehl : print("Current strehl: %f" % (s))	
	if strehl > 0 :
		phases *= np.sqrt(np.log(strehl)/np.log(s)) 	
	return phases
# --!EOFunc-----------------------------------------------------------------------------------------------------------
# Code for test and draw
# phases=phaseScreen(1,2048,r0=0.3,seed=0,ao=0)
#result=plt.matshow(phases)
#plt.show(result)
# checks
#print ("Max phase delay = %f nm" % (np.max(phases)-np.min(phases)/1e-9))
#print ("Max phase difference at 1000nm (rad) = %f" % (np.max(phases)-np.min(phases)*2*3.14/1e-6))
#print ("Test rms = %f" % phases.std())

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------delayToPhase------------------------------------------------------------
# changing wave delay to phase delay for a particular wavelength 
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# phasescreen - input phasescreen (for original or close wavelength) with delay in meters
# wl - wavelength
# norm - normalize to [0:2*pi)
# output:
#	- phases - delay in radians
def delayToPhase(phasescreen,wl=1e-6, norm=False) :
	''' changing wave delay to phase delay for a particular wavelength  '''
	if norm==True :
		phases=((phasescreen-phasescreen.min()) % wl)/wl*2*np.pi
	else :
		phases=(phasescreen-phasescreen.min())/wl*2*np.pi
	return phases	
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------phaseeToDelay------------------------------------------------------------
# changing phase delay to wave delay for a particular wavelength 
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
# phasescreen - input phasescreen (for original or close wavelength) with delay in radians
# wl - wavelength
# output:
#	- phases - delay in meters
def phaseToDelay(phasescreen,wl=1e-6) :
	''' changing phase delay to wave delay for a particular wavelength  '''
	wl_delay = phasescreen*(wl/(2*np.pi))
	return wl_delay	
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# ----------------------------------------extractRegion---------------------------------------------------------------
# the function takes matrix data as a periodic one and extracts data of sizex x sizey beginning in (x,y) cooordinate
# NB: Please keep in mind that 1st index of a matrix is related to Y axis and 2nd index is to X axis
# --------------------------------------------------------------------------------------------------------------------
def extractRegion(data,x,y,sizex,sizey) :
	''' the function takes matrix data as a periodic 
	one and extracts data of sizex x sizey beginning in (x,y) cooordinate '''
	region=np.empty((sizey,sizex),dtype=data.dtype)
	# placing the starting point inside the matrix
	new_y=int(y-np.floor(y/data.shape[0])*data.shape[0])
	new_x=int(x-np.floor(x/data.shape[1])*data.shape[1])
	# calculating the point inside data matrix where region ends
	x_end = new_x+sizex-int(np.floor((new_x+sizex-1)/data.shape[1])*data.shape[1])
	y_end = new_y+sizey-int(np.floor((new_y+sizey-1)/data.shape[0])*data.shape[0])
	#extract region
	# data border in general may split the region into following rectangles
	# 1 2
	# 3 4	
	if x_end>new_x and y_end>new_y : # case 1: all the rectangles are within the data border
		region[0:sizey,0:sizex]=data[new_y:y_end,new_x:x_end]		
	elif x_end>new_x and y_end<=new_y : # case 2: rectangles 3 and 4 are out of the data border
		region[0:(data.shape[1]-new_y),0:sizex]=data[new_y:data.shape[1],new_x:x_end]
		region[(data.shape[1]-new_y):sizey,0:sizex]=data[0:y_end,new_x:x_end]	
	elif x_end<=new_x and y_end>new_y : # case 3: rectangles 2 and 4 are out of the data border
		region[0:sizey,0:data.shape[0]-new_x]=data[new_y:y_end,new_x:data.shape[0]]
		region[0:sizey,(data.shape[0]-new_x):sizex]=data[new_y:y_end,0:x_end]	
	elif x_end<=new_x and y_end<=new_y : # case 4: rectangles 2, 3 and 4 are out of the data border
		# rect 1
		region[0:(data.shape[1]-new_y),0:(data.shape[0]-new_x)]=data[new_y:(data.shape[1]),new_x:(data.shape[0])]
		# rect 4
		region[(data.shape[1]-new_y):sizey,(data.shape[0]-new_x):sizex]=data[0:y_end,0:x_end]
		# rect 2
		region[0:(data.shape[1]-new_y),(data.shape[0]-new_x):sizex]=data[new_y:data.shape[1],0:x_end]
		# rect 3
		region[(data.shape[1]-new_y):sizey,0:(data.shape[0]-new_x)]=data[0:y_end,new_x:data.shape[0]]
	return region
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------getPupilScreen----------------------------------------------------
# A function for calculating phasescreen over the pupil at an arbitary moment of time t
# --------------------------------------------------------------------------------------------------------------------
# Here we assume that wind is blowing with constant speed v and (0,0) coordinate of pupil is located 
# at (x0,y0) point of phasescreen. The direction of wind is detected automatically to get the maximum coverage of phasescreen
# phases			# pre-generated phasescreen
# pupilSize=0.5		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
# scale=2048.0 		# scale factor (pixels/m)
# v=1.0			# wind velocity (m/s)
# t				# point of time to get phasescreen from 
# x0=0, y0=0 		# initial coordinates (assuming the screen is periodic)
def getPupilScreen(phases,pupilSize,scale,v,t,x0=0,y0=0) :
	''' A function for calculating phasescreen over the pupil at an arbitary moment of time t '''
	#
	# calculating pupil diameter in pixels
	pupilpix=int(np.floor(scale*pupilSize))
	#
	# getting screen size in pixels
	screenpix=len(phases)
	#
	# movement path angle (tg)
	#mvtangle = 1.0*pupilpix/screenpix
	mvlength=np.hypot(screenpix,pupilpix)
	#
	# converting velocity and velocity projections to pixels/s
	vpix=np.floor(scale*v)
	vpix_x=int(np.round(vpix*screenpix/mvlength))
	vpix_y=int(np.round(vpix*pupilpix/mvlength))
	#
	# a number of checks
	if screenpix< pupilpix :
		print("Phase screen is too small. The minimum size is %f m" % (screenpix/scale))
		return
	if t*vpix_x+x0 > np.floor(screenpix/pupilpix)*screenpix :
		print("Time value is too big. The maximum time is %f s" % (np.floor(screenpix/pupilpix)*screenpix/vpix_x))
		return
	# current shift of a screen 
	mv_x=int(vpix_x*t)
	mv_y=int(vpix_y*t)
	#
	return extractRegion(phases,x0+mv_x,y0+mv_y,pupilpix,pupilpix)
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------getPhasesEvolution------------------------------------------
# show phasescreen movement above the pupil
# --------------------------------------------------------------------------------------------------------------------
# phases - pre-generated phasescreen/visibility screens
# pupilSize=0.5		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
# scale=2048.0 		# scale factor (pixels/m)
# v=1.0			# wind velocity (m/s)
# sfrq=10			# number of samples per second
# stime=2.0			# desired sampling time (0 = maximum)
# expTime=0.0		# exposure time for each shot
# expNum=5			# number of frames taken for averaging
# showProgress=True    # show processing progress
# returns 
#	- evolving phasescreens array (pupilScreens). 
# 	 If exposure time > 0 then each element of pupilScreens arraycontains
#	 additional array of arrays with phasescreens over exposure time
#	 check len(shape(pupilScreens)) before using the output
def getPhasesEvolution(phases,pupilSize,scale,v,sfrq,stime=0.0,expTime=0.0,expNum=5, showProgress=True) :
	''' show phasescreen movement above the pupil '''
	# duration of sample
	sdur=1.0/sfrq	
	# getting screen size by X axis in pixels
	screenpix = phases.shape[1]
	# calculating pupil diameter in pixels
	pupilpix=int(np.floor(scale*pupilSize))
	# calculating Vpix_x - projected speed of wind in pixels/s
	vpix_x=int(np.round(np.floor(scale*v)*screenpix/np.hypot(screenpix,pupilpix)))
	if stime==0 :	# calculating maximum sampling time
		stime = (np.floor(screenpix/pupilpix)*screenpix/vpix_x)-expTime
	elif (stime+expTime)*vpix_x > np.floor(screenpix/pupilpix)*screenpix :
		print("Sampling or exposure time is too long. The maximum time period is %f s" % (1.0*np.floor(screenpix/pupilpix)*screenpix/vpix_x))
		return
	#
	# initializing time counter
	time = 0.0
	#pupilScreens=[]
	if expTime>0.0 and expNum>1 :
		pupilScreens=np.zeros((int(stime/sdur)+1,expNum,int(pupilSize*scale),int(pupilSize*scale)),dtype='float')
	else :
		pupilScreens=np.zeros((int(stime/sdur)+1,int(pupilSize*scale),int(pupilSize*scale)),dtype='float')
	# Sampling
	time_n=0		
	while (time<stime) :
		if showProgress : print("\nTime=%fs. End time: %fs" % (time,stime))			
		# exposure time != 0 - smoothing screen
		if expTime>0.0 and expNum>1 :
			#expScreen=[]
			for expFrame in range(0,expNum) :
				if showProgress : sys.stdout.write("\r	- Exposure screen %d of %d" % (expFrame+1,expNum))
				#print("	- Exposure screen %d of %d" % (expFrame+1,expNum))
				#expScreen.append(getPupilScreen(phases,pupilSize,scale,v,time+expTime*expFrame/expNum))
				pupilScreens[time_n][expFrame]=getPupilScreen(phases,pupilSize,scale,v,time+expTime*expFrame/expNum)
			#pupilScreens.append(expScreen)
		else :
			#pupilScreens.append(getPupilScreen(phases,pupilSize,scale,v,time))
			pupilScreens[time_n]=getPupilScreen(phases,pupilSize,scale,v,time)
		time+=sdur
		time_n+=1
	#return np.asarray(pupilScreens)
	return pupilScreens		
# --!EOFunc-----------------------------------------------------------------------------------------------------------
