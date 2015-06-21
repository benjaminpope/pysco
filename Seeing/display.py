# ------------------------------------------------------------ display.py --------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains functions for displaying different characteristics, animations, images,------
# ------------------- connected with seeing, pupul models, wave propagation, etc. ------------------------------------
# ====================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import time
# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------displayImages----------------------------------------
# display evolution of images over time(e.g., phasescreens over a pupil)
# --------------------------------------------------------------------------------------------------------------------
# frames - array of matrices with data (may contain one image only). NB: Y axis will be reverted while displaying
# delay - delay between frames
# axisSize=[xmin,xmax,ymin,ymax] - min and max values across x and y axises.
# zmin,zmax - normalization of colormap (related to minimum and maximum in data)
#		if zmin==zmax (default) then normalization will be within the frame
# figureNum - number of figure to display on
# flipY - do we need to flip image across Y axis. NB: if you set this value to False make sure that ymin>=ymax
# showColorbar - do we need to display colorbar with depth map. NB: be careful: colorbar may display wrong scheme if zmin==zmax
# cmap - colormap name for the image
# log - use logarithmic scale
def displayImages(frames,delay=0.0, axisSize=[],zmin=0.0,zmax=0.0, 
			xlabel='', ylabel='', title='',colorbarTitle='',
			figureNum=1,flipY=True,showColorbar=True,cmap='gray',log=False) :
	''''display evolution of images over time(e.g., phasescreens over a pupil)'''
	if len(frames) == 0 :
		print("No frames in input array")
		return
	fig = plt.figure(figureNum)
	plt.clf()
	if zmin==zmax :
		zmin=frames[0].min()
		zmax=frames[0].max()
	#
	if log==False :
		norm=mc.Normalize(vmin=zmin,vmax=zmax, clip = False)
	else :
		norm=mc.LogNorm(vmin=zmin,vmax=zmax, clip = False)	
	#
	if len(axisSize)>0 :
		if len(axisSize) !=4 :
			print("Length of axisSize array must be either 0 or 4")
			return
		img = plt.imshow(np.zeros(np.shape(frames[0])),extent=axisSize,interpolation='none',norm=norm)
		plt.xlim(axisSize[0],axisSize[1])
		plt.ylim(axisSize[2],axisSize[3])
	else :
		img = plt.imshow(np.zeros(np.shape(frames[0])),interpolation='none',norm=norm)
	img.set_cmap(cmap)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if showColorbar==True :
		cbar = plt.colorbar()
		cbar.set_clim(zmin, zmax)
		cbar.set_norm(norm)
		cbar.set_label(colorbarTitle, rotation=270)
	#
	nFrames=len(frames)
	#
	if nFrames>0 :
		# starting animation in interactive mode
		plt.ion()
		plt.show()
		for i in range(0,nFrames) :				
			if flipY==True :
				imgFrame=np.flipud(frames[i])
			else :
				imgFrame=frames[i]
			if zmin==zmax :
				zmin=frames[i].min()
				zmax=frames[i].max()
			if log==False :
				norm=mc.Normalize(vmin=zmin,vmax=zmax, clip = False)
			else :
				norm=mc.LogNorm(vmin=zmin,vmax=zmax, clip = False)	
			if len(axisSize)>0 :
				img = plt.imshow(imgFrame,extent=axisSize,interpolation='none', norm = norm)
			else :
				img = plt.imshow(imgFrame,interpolation='none',norm = norm)
			img.set_cmap(cmap)
			fig.canvas.draw()
			print("Frame %d of %d" % (i+1,nFrames))	
			time.sleep(delay)					
			img.remove()			
			plt.ioff()
		raw_input("Press Enter key to continue")
		plt.ioff()		 		
		plt.close()
# --!EOFunc-----------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------displayImage-----------------------------------------
# display single image
# --------------------------------------------------------------------------------------------------------------------
# frame - a matrix with data
# axisSize=[xmin,xmax,ymin,ymax] - min and max values across x and y axises
# zmin,zmax - normalization of colormap (related to minimum and maximum in data)
#		if zmin==zmax (default) then normalization will be within the frame
# showColorbar - do we need to display colorbar with depth map
# flipY - do we need to flip Y axis. if you set this value to False make sure that ymin>=ymax
# cmap - colormap name for the image
# log - use logarithmic scale
# figureNum - number of figure to display on (not used anymore)
def displayImage(frame, axisSize=[],zmin=1.0,zmax=1.0, 
		xlabel='', ylabel='', title='',colorbarTitle='',
		showColorbar=True,flipY=True,cmap='gray',log=False,figureNum=1) :
	''' display single image '''
	if flipY==True :
		imgFrame=np.flipud(frame)
	else :
		imgFrame=frame
	if zmin==zmax :
		zmin=frame.min()
		zmax=frame.max()
	plt.clf()
	#
	if log==False :
		norm=mc.Normalize(vmin=zmin,vmax=zmax, clip = False)
	else :
		norm=mc.LogNorm(vmin=zmin,vmax=zmax, clip = False)	
	#
	if len(axisSize)>0 :
		if len(axisSize) !=4 :
			print("Length of axisSize array must be either 0 or 4")
			return
		img = plt.imshow(imgFrame,extent=axisSize,interpolation='none',norm=norm)
		plt.xlim(axisSize[0],axisSize[1])
		plt.ylim(axisSize[2],axisSize[3])
	else :
		img = plt.imshow(imgFrame,interpolation='none',norm=norm)
	#
	img.set_cmap(cmap)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if showColorbar==True :
		cbar = plt.colorbar()
		cbar.set_clim(zmin, zmax)
		cbar.set_norm(norm)
		cbar.set_label(colorbarTitle, rotation=270)
	plt.show()
# --!EOFunc-----------------------------------------------------------------------------------------------------------

