# ---------------------------------------------------- sim.py --------------------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains functions for image simulations ---------------------------------------------
# ====================================================================================================================

import numpy as np
import scipy as sp
import scipy.misc

from phasescreen import getPupilScreen
from phasescreen import getPhasesEvolution

from display import displayImages
from display import displayImage

from pupil import getScaledMask

from common_tasks import shift_image
from common_tasks import rad2mas
from common_tasks import mas2rad

from astropy.io import fits

import sys
import multiprocessing as mp



shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq

# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getFocalImage---------------------------------------------
# generates an image through a mask
# --------------------------------------------------------------------------------------------------------------------
# mask - square array of zeroes and ones where 0=light is blocked
# image - complex array representing input image in pupil plane
# seeing - complex seeing in pupil plane. If not complex (phases only), then seeing=cos(phases)+1j*sin(phases)
# 		seeing can be complex seeing in pupil plane over exposure time if len(shape(seeing))==3
#		Then each element is treated as a seeing at different exposure time
# scale - scale to zoom out input image in order to establish correct correspondense between 
#		FFT and image
#		usually scale~lambda[m]/(plateScale(rad/pix)*pupil_size[m])
# cropPix - number of pixels in output image. It will be either cropped or enlarged
#		0 = no change
# photonNoise=False - introduce photon noise modelled by poisson distribution
# expCoeff=0.0 - a coefficient to reduce the overall flux to model comparable long exposures
#		0 = no coefficient
# showProgress=True - show processing progress
# output:
#	- res - array of powers
def getFocalImage(image,mask,seeing=[],scale=1.0,cropPix=0, photonNoise=False, expCoeff=0.0,showProgress=True) :
	''' generates an image through a mask '''
	if len(mask) != len(image) :
		print("Scaling mask")
		mask_scaled=getScaledMask(mask,diam=1.0,scale=len(mask))
	else : 
		mask_scaled=mask	
	img=np.array(image)
	img*=mask_scaled
	if np.shape(seeing)[1]>0 :
		if np.shape(seeing)[1] != len(image) :
			print("Image and seeing array must be of the same size")
			return
	# preparing for scaling
	f_scale=scale
	if scale<=0.0 :
		print("Scale should be >= 0. Using 1.0 instead")
		f_scale=1.0
	# constructing padded image
	pad_size = int(round(np.shape(img)[0]*f_scale))
	img_size = np.shape(img)[0]
	if f_scale >= 1.0 :
		padded_img = np.zeros((pad_size,pad_size),dtype='complex')	
		pad_start=int(pad_size//2-img_size//2)
		pad_end=pad_start+img_size
	else :
		padded_img = np.zeros(np.shape(img),dtype='complex')
		pad_start=int(img_size//2-pad_size//2)
		pad_end=pad_start+pad_size
	# calculating number of seeings to apply
	nSeeings = 1
	if (len(np.shape(seeing))==3) :
		nSeeings = np.shape(seeing)[0]
	# constructing a result (non-cropped) image
	padded_res = np.zeros(np.shape(padded_img),dtype='float')
	# looping over a number of seeings
	currentSeeing=0
	while (currentSeeing<nSeeings) :
		# extracting a current seeing
		if (len(np.shape(seeing))==2) :
			cur_seeing = seeing
		else :
			cur_seeing = seeing[currentSeeing]
			if showProgress :
				print("	- Processing exposure image %d of %d" % (currentSeeing+1,nSeeings))
		#  applying the first seeing		
		if (seeing.dtype!='complex') :
			# converting phases to complex seeing instead
			cur_seeing=np.cos(cur_seeing)+1j*np.sin(cur_seeing)
		cur_img=img*cur_seeing
		if f_scale>=1.0 :
			#padded_img[pad_start:pad_end,pad_start:pad_end]=img
			padded_img[0:img_size,0:img_size]=cur_img
		else :
			padded_img=cur_img[pad_start:pad_end,pad_start:pad_end]
		#padded_img[pad_start:pad_end,pad_start:pad_end]=img[pad_start:pad_end,pad_start:pad_end]		
		# FFT
		#padded_res = shift(fft(shift(padded_img)))
		if expCoeff>0 :
			padded_img*=expCoeff
		if nSeeings>0 :
			padded_img/=nSeeings
		temp=np.abs(shift(fft(padded_img)))**2
		if photonNoise :
			temp=np.random.poisson(temp,temp.shape).astype(float)
		padded_res += temp		
		currentSeeing+=1					
	# extracting the result
	'''	
	if nSeeings > 1:
		padded_res/=nSeeings
	'''
	if cropPix<= pad_size and cropPix>0 :
		res = padded_res[pad_size//2-cropPix//2:pad_size//2-cropPix//2+cropPix,
				pad_size//2-cropPix//2:pad_size//2-cropPix//2+cropPix]
	elif cropPix==0 :
		if f_scale>=1.0 :
			res=padded_res[pad_start:pad_end,pad_start:pad_end]
		else :
			res=padded_res
	elif cropPix<0 :
		print("Croppix must be >=0")
		return
	else :
		#res=sp.misc.imresize(padded_res.real,(cropPix,cropPix))
		res=np.zeros((cropPix,cropPix))
		res[cropPix//2-pad_size//2:cropPix//2-pad_size//2+pad_size,
			cropPix//2-pad_size//2:cropPix//2-pad_size//2+pad_size]=padded_res	
	return res	
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getFocalImageMp---------------------------------------------
# getFocalImage for multiprocessing
# --------------------------------------------------------------------------------------------------------------------
def getFocalImageMp(params) :
	image,mask,seeing,scale,cropPix, photonNoise, expCoeff, i, nRes = params
	sys.stdout.write("\r---Processing image %d of %d---" % (i+1,nRes))
	return getFocalImage(image,mask,seeing,scale,cropPix, photonNoise, expCoeff,showProgress=False)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getFocalImages---------------------------------------------
# generates images through a mask
# --------------------------------------------------------------------------------------------------------------------
# mask - square array of zeroes and ones where 0=light is blocked
# image - complex image representing input image in pupil plane. 
# seeings - complex seeings in pupil plane. If not complex (phases only), then seeing=cos(phases)+1j*sin(phases)
# 		each seeing can be complex seeing in pupil plane over exposure time if len(shape(seeings))==4
#		Then each element is treated as an array of seeings over a different exposure time
#scale - scale to zoom out input image in order to establish correct correspondense between 
#		FFT and image
#		usually scale~lambda[m]/(plateScale(rad/pix)*pupil_size[m])
# cropPix - number of pixels in output image. It will be either cropped or enlarged
#		0 = no change
# photonNoise=False - introduce photon noise modelled by poisson distribution
# expCoeff=0.0 - a coefficient to reduce the overall flux to model comparable long exposures
#		0 = no coefficient 
# processes - numper of processes to use. -1 - do not use multiprocessing. 0 - cpu count
# output:
#	- res - complex array powers in focal plane
def getFocalImages(image,mask,seeings=[],scale=1.0,cropPix=0,photonNoise=False, expCoeff=0.0, processes=-1) :
	''' generates images through a mask '''
	if len(mask) != len(image) :
		print("Scaling mask")
		mask_scaled=getScaledMask(mask,diam=1.0,scale=len(mask))
	else : 
		mask_scaled=mask	
	if len(seeings)==0 :
		nRes = 1		
	else :
		nRes = len(seeings)
	f_scale=scale		
	if scale<=0.0 :
		print("Scale should be >= 0. Using 1.0 instead")
		f_scale=1.0	
	print("\n")
	if processes==-1 or nRes==1:		
		res = []
		for i in range(nRes) :
			sys.stdout.write("\r---Processing image %d of %d---" % (i+1,nRes))
			res.append(getFocalImage(image,mask_scaled,seeing=seeings[i],
				scale=f_scale,cropPix=cropPix,photonNoise=photonNoise,expCoeff=expCoeff))		
	else:
		if processes==0 : pool = mp.Pool(processes=min(mp.cpu_count(),nRes))
		else : pool = mp.Pool(processes=min(processes,nRes))
		res=pool.map(getFocalImageMp, ((image,mask_scaled,seeings[i],f_scale,cropPix,photonNoise,expCoeff,i,nRes) for i in range(nRes)))				
		pool.close()
		pool.join()
		pool=None			
	return np.asarray(res)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------imageToFits---------------------------------------------
# saving generated image to a fits file
# --------------------------------------------------------------------------------------------------------------------
# image - input image
# path - path to file
# filename- output filename (.fits or .fits.gz)
# Fits headers:	
# tel  - telescope
#        pscale                 		    # simulation plate scale (mas)
#        'odate'  : "Jan 1, 2000",          # UTC date of observation
#        'otime'  : "0:00:00.00",           # UTC time of observation
#        'tint'   : 1.0,                    # integration time (sec)
#        'coadds' : 1,                      # number of coadds
#        'RA'     : 0.000,                  # right ascension (deg)
#        'DEC'    : 0.000,                  # declination (deg)
#        'filter' : 1e-6,              	    # central wavelength (meters)
#        'orient' : 0.0                     # P.A. of the frame (deg)

def imageToFits(image,path='./',filename='image.fits',
	tel='simu',pscale=10.0,odate='Jan 1, 2000', otime= "0:00:00.00",
	tint=1.0,coadds=1,RA=0.0,DEC=0.0,filter=1.6e-6,orient=0.0) :
	''' saving generated image to a fits file '''
	prihdr = fits.Header()
	prihdr.append(('TELESCOP',tel))
	prihdr.append(('PSCALE',pscale))
	prihdr.append(('ODATE',odate))
	prihdr.append(('OTIME',otime))
	prihdr.append(('TINT',tint))
	prihdr.append(('FNAME',filename))
	prihdr.append(('COADDS',coadds))
	prihdr.append(('RA',RA))
	prihdr.append(('DEC',DEC))
	prihdr.append(('FILTER',filter))
	prihdr.append(('ORIENT',orient))
	hdu = fits.PrimaryHDU(image,prihdr)
	hdu.writeto(path+filename)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------normalizeImage--------------------------------------------
# normalizing image to [0,1] range
# --------------------------------------------------------------------------------------------------------------------
# image - input image
# res - output image	

def normalizeImage(image) :
	''' normalizing image to [0,1] range '''
	max = image.max()
	min = image.min()
	if max==min :
		return np.zeros(np.shape(image))
	return (image-min)/(max-min)
# --!EOFunc-----------------------------------------------------------------------------------------------------------	
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------normalizeImages--------------------------------------------
# normalizing an array of images to [0,1] range
# --------------------------------------------------------------------------------------------------------------------
# images - input images array[NxMxK] where N is number of images
# res - output image	

def normalizeImages(images) :
	# normalizing an array of images to [0,1] range
	if len(np.shape(images))==2 :
		return normalizeImage(images)
	elif len(np.shape(images))!=3 :
		print("Incorrect input array")
		return images
	else :
		res = []
		max_res=images[0].max()
		min_res=images[0].min()
		for i in range(1,np.shape(images)[0]) :
			max = images[i].max()
			min= images[i].min()
			if min<min_res :
				min_res=min
			if max>max_res :
				max_res=max
		for i in range(0,np.shape(images)[0]):
			res.append((images[i]-min_res)/(max_res-min_res))
	return np.asarray(res)
# --!EOFunc-----------------------------------------------------------------------------------------------------------		

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------getDistFocalImage----------------------------------------------------
# A function for generating image through a mask with atmosphere phasescreen over the pupil at an arbitary moment of time t
# --------------------------------------------------------------------------------------------------------------------
# Here we assume that wind is blowing with constant speed v and (0,0) coordinate of pupil is located 
# at (x0,y0) point of phasescreen. The direction of wind is detected automatically to get the maximum coverage of phasescreen
# mask - square array of zeroes and ones where 0=light is blocked
# image - complex image representing input image in pupil plane. 
# phaseScreen		# pre-generated phasescreen (delay in meters)
# pupilSize=0.5		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
# scale=2048.0 		# scale factor (pixels/m)
# plateScale=5.0		# plate scale (mas/pixel)
# detectorSize = 0	# size of the detector. If 0 then use the same size as pupil
# v=1.0			# wind velocity (m/s)
# t				# point of time to get phasescreen from 
# x0=0, y0=0 		# initial coordinates (assuming the screen is periodic)
# wl = 1e-6			# wavelength. 0.0 - do not take wavelength into consideration
# expTime=0.0		# exposure time for each shot
# expNum=5			# number of frames taken for averaging
# compl=False - if False - return complex array, otherwise return amplitudes and phases separately
# output:
#	- res -  array of Power Spectrum in Focal Plane
def getDistFocalImage(image,mask,
	phaseScreen,pupilSize,scale,plateScale,detectorSize=0,
	v=1.0,t=0.0,x0=0,y0=0,wl=0.0,expTime=0.0,expNum=5,
	compl=False) :
	'''A function for generating image through a mask with 
	atmospheric phasescreen over the pupil at an arbitary moment of time t'''
	#----------old--------
	# phase screen patch extraction 
	#wl_delay=getPupilScreen(phaseScreen,pupilSize,scale,v,t,x0,y0) 
	# converting to phase delay
	#phases=delayToPhase(wl_delay,wl=1e-6)
	#---------------------
	# phase screen patch extraction 
	if expTime>0 :
		phases=getPhasesEvolution(phases=phaseScreen,pupilSize=pupilSize,scale=scale,
			v=v,sfrq=1/expTime,stime=expTime,expTime=expTime,expNum=expNum)
	else :
		phases=getPupilScreen(phaseScreen,pupilSize,scale,v,t,x0,y0) 
	# scale factor for FFT
	scale_f=1.0
	if wl > 0.0 :
		scale_f=wl/(pupilSize*mas2rad(plateScale))		
		#scale_f=1.22*wl/(np.power(pupilSize,2)*mas2rad(plateScale)*scale)
	# crop
	cropPix=detectorSize
	if detectorSize ==0 :
		cropPix = int(scale*pupilSize)
	# returning image
	if expTime>0 :
		return getFocalImage(image,mask,phases[0],scale=scale_f,cropPix=cropPix)
	else :
		return getFocalImage(image,mask,phases,scale=scale_f,cropPix=cropPix)
# ------------------------------------------------EOFunc---------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------getDistFocalImages----------------------------------------------------
# A function for generating evolving images through a mask with atmosphere phasescreen over the pupil
# --------------------------------------------------------------------------------------------------------------------
# Here we assume that wind is blowing with constant speed v and (0,0) coordinate of pupil is located 
# at (0,0) point of phasescreen. The direction of wind is detected automatically to get the maximum coverage of phasescreen
# mask - square array of zeroes and ones where 0=light is blocked
# image - complex image representing input image in pupil plane. 
# phaseScreen		# pre-generated phasescreen (delay in meters)
# pupilSize=0.5		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
# scale=2048.0 		# scale factor (pixels/m)
# plateScale=5.0		# plate scale (mas/pixel)
# detectorSize = 0	# size of the detector. If 0 then use the same size as pupil
# v=1.0			# wind velocity (m/s)
# sfrq=10			# number of samples per second
# stime=2.0			# desired sampling time (0 = maximum)
# wl = 1e-6			# wavelength. 0.0 - do not take wavelength into consideration
# expTime=0.0		# exposure time for each shot
# expNum=5			# number of frames taken for averaging
# display - do we need to display each frame (powers, log(x+10) scale),
#		NB: if we don't return the data, display will be in frame-by-frame mode only
# delay=0.0 - delay between frames
# displayFactor=1.0 - power for result display
# returnData - do we need to return an array(s) of images
# output:
#	- res - array of images of power spectrum
def getDistFocalImages(image,mask,
	phaseScreen,pupilSize,scale,
	plateScale=0.1,
	detectorSize=0,
	v=1.0,sfrq=10,stime=0.0,
	wl=0.0,
	expTime=0.0,
	expNum=5,
	display=True,delay=0.0,displayFactor=1.0,
	returnData=True) :
	'''A function for generating evolving images through a mask with 
	atmospheric phasescreen over the pupil'''
      #
	# duration of sample
	sdur=1.0/sfrq	
	# getting screen size by X axis in pixels
	screenpix = phaseScreen.shape[1]
	# calculating pupil diameter in pixels
	pupilpix=int(np.floor(scale*pupilSize))
	# plate size - plate field of view in mas
	plateSize = plateScale * pupilpix
	# calculating Vpix_x - projected speed of wind in pixels/s
	vpix_x=int(np.round(np.floor(scale*v)*screenpix/np.hypot(screenpix,pupilpix)))
	s_time = stime
	if stime==0 :	# calculating maximum sampling time
		s_time = (np.floor(screenpix/pupilpix*scale)*screenpix/vpix_x)
	elif stime*vpix_x > np.floor(screenpix/pupilpix)*screenpix :
		print("Sampling time is too long. The maximum time is %f s" % (np.floor(screenpix/pupilpix)*screenpix/vpix_x))
		return
	#
	if returnData :
	      nSamples = int(s_time*sfrq)
	else :
		nSamples=1
      #
	# initializing time counter
	time = 0.0
	i = 0
	if returnData :
		pows = np.empty((nSamples),dtype='object')
	# Sampling
	while (time<s_time and i<nSamples) :
		print("Time=%fs. End time: %fs" % (time,s_time))	
		pows[i]=getDistFocalImage(image,mask,phaseScreen,
			pupilSize,scale,plateScale,detectorSize,
			v,time,x0=0.0,y0=0.0,wl=wl,expTime=expTime,expNum=expNum)
		if display and not returnData:
			displayImage(np.power(np.abs(pows),displayFactor),axisSize=[-plateSize/2,plateSize/2,-plateSize/2,plateSize/2],
				xlabel='mas', ylabel='mas', 
				title=("Focal Plane Power Spectrum ** %3.1f" % displayFactor),
				showColorbar=True,flipY=True,cmap='gray') 
		time+=sdur
		if returnData:
			i+=1
	#
	if returnData :
		if display :
			displayImages(np.power(np.abs(pows),displayFactor),delay,
				axisSize=[-plateSize/2,plateSize/2,-plateSize/2,plateSize/2],
				xlabel='mas', ylabel='mas', 
				title=("Focal Plane Power Spectrum ** %3.1f" % displayFactor),
				figureNum=1,showColorbar=True,flipY=True,cmap='gray') 			
		return pows
# ------------------------------------------------EOFunc---------------------------------------------------------------
		
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------simBinary-------------------------------------------------------------
# simulates a binary star for a given contrast ratio, separation, and angle from single star
# assumes that a sigle star is in the very center of a given image		
# --------------------------------------------------------------------------------------------------------------------						
# img - input image with single star centered
# c - contrast ratio
# sep - binary stars separation. in pixels if both lambdaD and pscale==0.
#	  in mas if pscale > 0
#	  in lambda/D if lambda/D > 0 and pscale>0		
# ang - separation angle
# lambdaD - lambda/D value
# pscale - plate scale
# forceInt - force integer value for separation in terms of pixels
# output:
#	- res - an image with a binary star for a given set of parameters
def simBinary(img, c, sep, ang=0., lambdaD=0., pscale=0., forceInt=True) :
	if pscale>0 :
		sep_px=sep/pscale
		if lambdaD>0 :
			sep_px*=rad2mas(lambdaD)
	else : sep_px=sep		
	x_sep=sep_px*np.cos(ang)
	y_sep=sep_px*np.sin(ang)
	if forceInt :
		x_sep=int(np.round(x_sep))
		y_sep=int(np.round(y_sep))
	return img+shift_image(img,x=x_sep,y=y_sep,doRoll=False)/c
	
	