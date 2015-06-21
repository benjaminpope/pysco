# ------------------------------------------------------------ pupil.py ----------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains functions for generation, saving and loading different pupil configurations--
# ====================================================================================================================
import numpy as np
import scipy as sp
import scipy.misc
shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getNHolesMask----------------------------------------
# generates curcular pupil with arbitary number of holes in it
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# holeDiam - diameter of a single hole (m)
# holeCoords - Nx2 array with hole coordinates [[x0,y0],[x1,y1],[x2,y2],...] in meters from the center of the pupil
# border - optional frame border 
# scale - # scale factor (pixels/m)
# output:
#	- int array of (diam+2*border)*scale x (diam+2*border)*scale size where 
#	  1 shows area inside the pupil, 0 - outside
def getNHolesMask(diam=1.0,holeDiam=0.1,holeCoords=np.array([[0.,0.],[0.2,0.2]]),border=0.0,scale=1024) :
	''' generates curcular pupil with arbitary number of holes in it '''
	nHoles = np.shape(holeCoords)[0]
	rad=diam/2
	holeRad=holeDiam/2
	#checks
	if holeDiam > diam :
		print("Diameter of pupul smaller than diameter of holes in mask")
		return
	if nHoles<1 :
		print("Number of coordinate smust be grater than zero")
		return
	#
	for i in range(nHoles) :
		if (holeRad+abs(holeCoords[i,0]))<border or (abs(holeCoords[i,0])+holeRad)>(border+diam) or (holeRad+abs(holeCoords[i,1]))<border or (abs(holeCoords[i,1])+holeRad)>(border+diam) :
			print("Hole position (x%d,y%d) is out of range. " % (i))
			return
	# resolution of the output image and other quantities
	size=int((2*border+diam)*scale)
	center=int((border+rad)*scale)
	holeRadPix=int(holeRad*scale)
	#calculating distances from the very centre
	xx,yy  = np.meshgrid(np.arange(size)-center, np.arange(size)-center)
	dist = np.hypot(yy,xx)
	#calculating holes Mask
	holesDist=dist[center-holeRadPix:center+holeRadPix+1,center-holeRadPix:center+holeRadPix+1]
	holesMask=np.zeros(np.shape(holesDist),dtype=int)
	for i in range(holesMask.shape[0]) :
		for j in range(holesMask.shape[1]) :
			if holesDist[i,j] <= holeRadPix :
				holesMask[i,j]=1
	# generating result mask
	mask=np.zeros(np.shape(dist),dtype=int)
	# opening holes
	for i in range(nHoles) :
		min_x=int(holeCoords[i,0]*scale)+center-holeRadPix
		max_x=int(holeCoords[i,0]*scale)+center+holeRadPix+1
		min_y=int(holeCoords[i,1]*scale)+center-holeRadPix
		max_y=int(holeCoords[i,1]*scale)+center+holeRadPix+1		
		mask[min_x:max_x,min_y:max_y]=holesMask	
	return np.transpose(mask)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
# test
#mask=getNHolesMask(diam=1.0,holeDiam=0.1,holeCoords=[0.,0.,0.2,0.2],border=0.0,scale=1024)
#displayImage(mask,title='Pupil Mask',axisSize=[-border-rad,border+rad,-border-rad,border+rad],xlabel='m', ylabel='m', figureNum=1,showColorbar=False,cmap='binary_r',flipY=True) 

# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getAnnulusMask----------------------------------------
# generates curcular pupil with annulus mask
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# innDiam - diameter of a inner circle of annulus (m)
# border - optional frame border 
# scale - # scale factor (pixels/m)
# output:
#	- int array of (diam+2*border)*scale x (diam+2*border)*scale size where 
#	  1 shows area inside the pupil, 0 - outside
def getAnnulusMask(diam=1.0,innDiam=0.9,border=0.0,scale=1024) :
	''' generates curcular pupil with annulus mask '''
	rad=diam/2
	innRad=innDiam/2
	#checks
	if innDiam >= diam :
		print("Diameter of pupul must be greater than inner diameter of annulus")
		return
	# resolution of the output image and other quantities
	size=int((2*border+diam)*scale)
	center=int((border+rad)*scale)
	pixelRad=int(rad*scale)
	pixelInnRad=int(innRad*scale)
	#calculating distances from the very centre
	xx,yy  = np.meshgrid(np.arange(size)-center, np.arange(size)-center)
	dist = np.hypot(yy,xx)
	# generating result mask
	mask=np.logical_and(dist<=pixelRad,dist>=pixelInnRad)*1		
	return np.transpose(mask)
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getFullMask-----------------------------------------------
# generates curcular pupil
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# border - optional frame border 
# scale - # scale factor (pixels/m)
# output:
#	- int array of (diam+2*border)*scale x (diam+2*border)*scale size where 
#	  1 shows area inside the pupil, 0 - outside
def getFullMask(diam=1.0,border=0.0,scale=1024) :
	''' generates curcular pupil '''
	rad=diam/2
	# resolution of the output image and other quantities
	size=int((2*border+diam)*scale)
	center=int((border+rad)*scale)
	pixelRad=int(rad*scale)
	#calculating distances from the very centre
	xx,yy  = np.meshgrid(np.arange(size)-center, np.arange(size)-center)
	dist = np.hypot(yy,xx)
	# generating result mask
	mask=(dist<=pixelRad)*1		
	return np.transpose(mask)
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getAnnulusHexMask----------------------------------------
# generates curcular pupil with annulus mask splitted in hexagonal segments
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# innDiam - diameter of a inner circle of annulus (m)
# hexDiam - diameter of hexagon (m)
# border - optional frame border 
# scale - # scale factor (pixels/m)
# returnSampling - return coordinates of sampling points
# output:
#	- int array of (diam+2*border)*scale x (diam+2*border)*scale size where 
#	  1 shows area inside the pupil, 0 - outside
#	- sampling points marking the centers of hexagons
def getAnnulusHexMask(diam=1.0,innDiam=0.9,hexDiam=0.1,border=0.0,scale=1024,returnSampling=False) :
	''' generates curcular pupil with annulus mask splitted to hexagonal segments'''
	rad=diam/2
	innRad=innDiam/2
	hexRad=hexDiam/2
	hexRadSmall=hexRad*np.cos(np.pi/6)
	#checks
	if innDiam >= diam :
		print("Diameter of pupul must be greater than inner diameter of annulus")
		return
	if hexDiam >= (diam-innDiam) :
		print("Diameter of hexagon must be smaller than a width of annulus")
		return		
	# resolution of the output image and other quantities
	size=int((2*border+diam)*scale)
	center=int((border+rad)*scale)
	pixelRad=int(rad*scale)
	pixelInnRad=int(innRad*scale)
	pixelHexRad=int(hexRad*scale)
	pixelHexRadSmall=int(hexRadSmall*scale)
	#calculating distances from the very centre
	xx,yy  = np.meshgrid(np.arange(size)-center, np.arange(size)-center)
	dist = np.hypot(yy,xx)
	# generating result mask
	mask=np.logical_and(dist<=pixelRad,dist>=pixelInnRad)*1	
	# getting sampling points
	step_x=2*hexRadSmall
	step_y=3*hexRad/2.0
	sampling=[]
	x=0.0
	y=0.0
	i=0
	while x<=rad and y<=rad-hexRad :
		if mask[int(x*scale)+center,int(y*scale)+center]>0 :
			sampling.append([x,y])
		x+=step_x		
		if x>rad-hexRadSmall :
			i+=1
			x=(i%2)*step_x/2.0
			y+=step_y
	# making sampling points symmetrical	
	nSampl = len(sampling)
	for i in range(0,nSampl) :
		if sampling[i][0]*scale>1 :
			sampling.append([-sampling[i][0],sampling[i][1]])
			if sampling[i][1]*scale>1 :
				sampling.append([-sampling[i][0],-sampling[i][1]])	
		if sampling[i][1]*scale>1 :
			sampling.append([sampling[i][0],-sampling[i][1]])			
	# cutting mask pieces
	# creating hexagon
	hexagonMask = np.ones((pixelHexRadSmall*2+1,pixelHexRad*2+1),dtype='int')
	tgpi6=np.tan(np.pi/6)
	for x in range(0,pixelHexRadSmall) :
		for y in range(0,int(pixelHexRad/2.0-x*tgpi6)) :
			hexagonMask[x,y]=0	
	hexagonMask[pixelHexRadSmall:np.shape(hexagonMask)[0],:]=np.flipud(hexagonMask[0:pixelHexRadSmall+1,:])
	hexagonMask[:,pixelHexRad:np.shape(hexagonMask)[1]]=np.fliplr(hexagonMask[:,0:pixelHexRad+1])
	# cutting
	mask=np.zeros(np.shape(mask),dtype='int')
	for i in range(len(sampling)) :
		min_x=int(sampling[i][0]*scale)+center-pixelHexRadSmall
		max_x=int(sampling[i][0]*scale)+center+pixelHexRadSmall+1
		min_y=int(sampling[i][1]*scale)+center-pixelHexRad
		max_y=int(sampling[i][1]*scale)+center+pixelHexRad+1
		#print("%d-%d,%d-%d" % min_x,max_x,min_y,max_y)
		mask[min_x:max_x,min_y:max_y]=np.logical_or(mask[min_x:max_x,min_y:max_y],hexagonMask)*1
		'''
		for x in range(min_x,max_x) :
			for y in range(min_y,max_y) :
				if (0<=x<np.shape(mask)[0]) and (0<=y<np.shape(mask)[1]) :
					if hexagonMask[x-min_x,y-min_y]>0 :
						mask[x,y]=1
			'''
	# smoothing along x axis
	for x in range(1,np.shape(mask)[0]-1) :
		for y in range(0,np.shape(mask)[1]) :	
			if mask[x,y]==0 :
				if mask[x-1,y]>0 and mask[x+1,y]>0 :
					mask[x,y]=1
	if returnSampling :					
		return np.transpose(mask), np.asarray(sampling)
	else :
		return np.transpose(mask)
# --!EOFunc-----------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getFullHexMask----------------------------------------
# generates curcular pupil with annulus mask splitted to hexagonal segments
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# hexDiam - diameter of hexagon (m)
# border - optional frame border 
# scale - # scale factor (pixels/m)
# returnSampling - return coordinates of sampling points
# output:
#	- int array of (diam+2*border)*scale x (diam+2*border)*scale size where 
#	  1 shows area inside the pupil, 0 - outside
#	- sampling points marking the centers of hexagons
def getFullHexMask(diam=1.0,hexDiam=0.1,border=0.0,scale=1024,returnSampling=False) :
	'''generates curcular pupil splitted in hexagonal segments'''
	return getAnnulusHexMask(diam,0.0,hexDiam,border,scale,returnSampling)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getScaledMask--------------------------------------------
# generates scaled mask for pupil
# --------------------------------------------------------------------------------------------------------------------
# diam - diameter of the pupil (m)
# scale - scale factor (pixels/m)
# flip - flip input pupil horizontally
# output:
#	- int array of diam*scale x diam*scale size where 
#	  1 shows area inside the pupil, 0 - outside
def getScaledMask(in_mask,diam=1.0,scale=100,flip=False) :
	''' generates scaled mask for pupil '''
	size=int(diam*scale)
	mask=sp.misc.imresize(in_mask,(size,size))
	if flip : 
		return mask
	else :
		return mask
# --!EOFunc-----------------------------------------------------------------------------------------------------------	
		
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getSamplingPoints----------------------------------------------------
# generates discrete sampling points array in mask
# --------------------------------------------------------------------------------------------------------------------
# NB: !!!!coordinates in the mask are transposed because it is an image!!!
# in_mask - square array of zeroes and ones where 0=light is blocked
# dens - sampling linear density (points/m)
# scale - scale factor (pixels/m)	
# filename - filename to write array [optional]
# prec  - decimal precision to round results [optional]
# shift_x,shift_y - shift of the very first sampling point in meters (for manual tweaks)
# output:
# res - array Nx2 with sampling points coordinates (in meters). (0.0) is a centre of pupil
def getSamplingPoints(in_mask,dens=100.0,scale=100.0,filename='',prec=6,shift_x=0.0,shift_y=0.0) :
	'''generates discrete sampling points array in mask'''
	step=1.0/dens
	#step_pix = int(scale/dens)
	#xs=int(round(shift_x*scale))
	#ys=int(round(shift_y*scale))
	size_y=in_mask.shape[0]/scale
	size_x=in_mask.shape[1]/scale
	res=[]
	x00=step+shift_x
	y00=step+shift_y
	x01=x00+step/2
	if x01>step :
		x01-=step
	yC=in_mask.shape[0]/scale/2
	xC=in_mask.shape[1]/scale/2
	x=x00
	y=y00
	i=0
	while (x<size_x and y<size_y) :
		if in_mask[int(y*scale),int(x*scale)]>0 :
			res.append([round(x-xC,prec),round(y-yC,prec)])
		x+=step
		if x>=size_x :
			if (i%2)==1 :
				x=x00
			else :
				x=x01
			y+=step
			i+=1
	if len(filename)>0 :
		np.savetxt(filename,res)
	return np.asarray(res)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
	# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getSquareSamplingPoints----------------------------------------------------
# generates discrete sampling points array in mask placedd in a square grid
# --------------------------------------------------------------------------------------------------------------------
# NB: !!!!coordinates in the mask are transposed because it is an image!!!
# in_mask - square array of zeroes and ones where 0=light is blocked
# dens - sampling linear density (points/m)
# scale - scale factor (pixels/m)	
# filename - filename to write array [optional]
# prec  - decimal precision to round results [optional]
# shift_x,shift_y - shift of the very first sampling point in meters (for manual tweaks)
# output:
# res - array Nx2 with sampling points coordinates (in meters). (0.0) is a centre of pupil
def getSquareSamplingPoints(in_mask,dens=100.0,scale=100.0,filename='',prec=6,shift_x=0.0,shift_y=0.0) :
	'''generates discrete sampling points array in mask'''
	step=1.0/dens
	#step_pix = int(scale/dens)
	#xs=int(round(shift_x*scale))
	#ys=int(round(shift_y*scale))
	size_y=in_mask.shape[0]/scale
	size_x=in_mask.shape[1]/scale
	res=[]
	x00=step+shift_x
	y00=step+shift_y
	yC=in_mask.shape[0]/scale/2
	xC=in_mask.shape[1]/scale/2
	x=x00
	y=y00
	i=0
	while (x<size_x and y<size_y) :
		if in_mask[int(y*scale),int(x*scale)]>0 :
			res.append([round(x-xC,prec),round(y-yC,prec)])
		x+=step
		if x>=size_x :
			x=x00
			y+=step
			i+=1
	if len(filename)>0 :
		np.savetxt(filename,res)
	return np.asarray(res)
# --!EOFunc-----------------------------------------------------------------------------------------------------------
	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getCircularSamplingPoints----------------------------------------------------
# generates discrete sampling points array in mask in a circular way
# --------------------------------------------------------------------------------------------------------------------
# NB: !!!!coordinates in the mask are transposed because it is an image!!!
# in_mask - square array of zeroes and ones where 0=light is blocked
# dens - sampling linear density (points/m)
# scale - scale factor (pixels/m)	
# filename - filename to write array [optional]
# prec  - decimal precision to round results [optional]
# shift_a - angle shift from zero	(in radians)
# shift_r - start radius shift from zero	
# output:
# res - array Nx2 with sampling points coordinates (in meters). (0.0) is a centre of pupil
def getCircularSamplingPoints(in_mask,dens=100.0,scale=100.0,filename='',prec=6,shift_a=0.0,shift_r=0.0) :
	''' generates discrete sampling points array in mask in a circular way '''
	step=1.0/dens
	size_y=in_mask.shape[0]/scale
	size_x=in_mask.shape[1]/scale	
	res=[]
	rmax=np.sqrt(size_x**2+size_y**2)/2
	r0=rmax-step+shift_r
	a0=shift_a
	r=r0
	a=a0
	yC=in_mask.shape[0]/scale/2
	xC=in_mask.shape[1]/scale/2	
	i=0
	# optimal step by angle
	a_step0=2*np.arcsin(step/(2*r))
	# fixing step to fit integer number of holes
	a_step0=np.pi/int(np.pi/a_step0)
	a_step=a_step0
	while (r>=0 and a<(a0+2*np.pi)) :
		if r<step :
			if r<(1.0/scale) :
				res.append([0.0,0.0])
			break
		x=r*np.cos(a)
		y=r*np.sin(a)
		if (0<=(x+xC)*scale<np.shape(in_mask)[0]) and (0<=(y+yC)*scale<np.shape(in_mask)[1]) :
			if in_mask[int((y+yC)*scale),int((x+xC)*scale)]>0 :
				res.append([round(x,prec),round(y,prec)])
		a+=a_step
		if a>=(a0+2*np.pi) :
			r-=step
			if (i%2)==1 :
				a=a0+a_step/2
			else :
				a=a0			
			if r>step :
				a_step=2*np.arcsin(step/(2*r))
				a_step=np.pi/int(np.pi/a_step)
			i+=1
	if len(filename)>0 :
		np.savetxt(filename,res)
	return np.asarray(res)
# --!EOFunc-----------------------------------------------------------------------------------------------------------	
# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------placeSamplingPoints----------------------------------------------------
# draws discrete sampling points on an image
# --------------------------------------------------------------------------------------------------------------------
# img - square array of zeroes and ones where 0=light is blocked
# sPoints - sampling points array (Nx2)
# scale - scale factor (pixels/m)	
# output:
# res - result image with sampling points
# NB: here image is a matrix, so x and y axises are swapped	
def placeSamplingPoints(img,sPoints,scale=100.0) :
	size=np.shape(img)[0]
	res=np.copy(img)
	points_pix=np.floor(sPoints*scale)+size//2-1
	points_pix[:,1]=points_pix[:,1]-1
	pointer_len=int(size // 100)
	maxVal=res.max()
	minVal=res.min()
	if minVal==maxVal :
		minVal=0.0
		maxVal=1.0
	midVal=float(maxVal+minVal)/2
	for i in range(np.shape(points_pix)[0]) :
		min_x = int(points_pix[i,0])-pointer_len
		max_x = int(points_pix[i,0])+pointer_len+1
		min_y = int(points_pix[i,1])-pointer_len
		max_y = int(points_pix[i,1])+pointer_len+1
		for y in range(min_y,max_y) :
			if y>=0 and y<size :
				if img[y,points_pix[i,0]]>midVal :
					res[y,points_pix[i,0]]=minVal
				else:
					res[y,points_pix[i,0]]=maxVal
		for x in range(min_x,max_x) :
			if x>=0 and x<size :
				if img[points_pix[i,1],x]>midVal :
					res[points_pix[i,1],x]=minVal
				else:
					res[points_pix[i,1],x]=maxVal					
	return res
	

# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------getPSF----------------------------------------------------
# generates point spread function for arbitarary mask
# --------------------------------------------------------------------------------------------------------------------
# mask - square array of zeroes and ones where 0=light is blocked
# compl=False - if False - return complex array, otherwise return amplitudes and phases separately
# output:
#------- if compl == False ----------
#	- amps - complex amplitudes
#	- phases - complex phases
#-------- complex=True ----------------
#	- psf - complex array
def getPSF(mask,compl=False) :
	''' generates point spread function for arbitarary mask '''
	psf = shift(fft(shift(mask)))
	if compl==True :
		return psf
	else :
		amps=np.sqrt(psf.real**2+psf.imag**2)
		phases=np.angle(psf)
		return amps, phases
# --!EOFunc-----------------------------------------------------------------------------------------------------------
# test
# amps, phases=getPSF(mask)
# displayImage(amps,title='PSF Amplitudes',figureNum=1,showColorbar=True,flipY=True,cmap='gray') 
# displayImage(phases,title='PSF Phases',figureNum=1,showColorbar=True,flipY=True,cmap='gray') 
