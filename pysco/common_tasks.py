# ---------------------------------------------------- common_tasks.py -----------------------------------------------
# Author: Alexey Latyshev --------------------------------------------------------------------------------------------
# ------------------- This file contains functions for typical tasks -------------------------------------------------
# ====================================================================================================================

import numpy as np

# =========================================================================
# =======================(c) Benjamine Pope================================
def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return x*np.pi/(180*3600*1000)
# =========================================================================
# =======================(c) Benjamine Pope================================
def rad2mas(x):
    ''' Convenient little function to convert radians to milliarcseconds '''
    return x/np.pi*(180*3600*1000)
# =========================================================================
# =========================================================================
				
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------shift_image-------------------------------------------------------------
# shift image to (possibly) non-discrete number of pixels
# --------------------------------------------------------------------------------------------------------------------				
# img - input image
# x - x axis shift
# y - y axis shift
# doRoll=True - do the shift in the loop so the last pixel becomes the first one in a case of 1 pixel shift
#			if False, replace new values to zeroes				
# output:
#	- res - new image				
def shift_image(img,x=0,y=0,doRoll=True) :
	# step 1 - shifting to the value less than zero
	# flipping image to work with positive shifts only
	in_img=img
	if x<0 :
		in_img=np.fliplr(in_img)
	if y<0 :
		in_img=np.flipud(in_img)
	x_abs=np.abs(x)
	y_abs=np.abs(y)
	float_x=x_abs-int(x_abs)
	float_y=y_abs-int(y_abs)
	if float_x>0 or float_y>0 :
		res=(1-float_x)*(1-float_y)*in_img
		img12=float_x*(1-float_y)*in_img
		img21=(1-float_x)*float_y*in_img
		img22=float_x*float_y*in_img
		res[1:res.shape[0],0:res.shape[1]]+=img12[0:res.shape[0]-1,0:res.shape[1]]
		res[0:res.shape[0],1:res.shape[1]]+=img21[0:res.shape[0],0:res.shape[1]-1]
		res[1:res.shape[0],1:res.shape[1]]+=img22[0:res.shape[0]-1,0:res.shape[1]-1]
		if doRoll :
			res[0,:]+=img12[res.shape[0]-1,:]
			res[:,0]+=img21[:,res.shape[1]-1]
			res[0,0]+=img22[res.shape[0]-1,res.shape[1]-1]
	else :
		res=in_img
	#step2 - shifting to discrete number of pixels
	if not doRoll :
		if x_abs<res.shape[0] and y_abs<res.shape[1] :
			shifted=res[0:res.shape[0]-int(x_abs),0:res.shape[1]-int(y_abs)]
			res=np.zeros(np.shape(res))
			res[int(x_abs):res.shape[0],int(y_abs):res.shape[1]]=shifted	
		else :
			res=np.zeros(np.shape(res))
	else :
		res=np.roll(res,int(x_abs),axis=0)
		res=np.roll(res,int(y_abs),axis=1)
	# flipping back
	if x<0 :
		res=np.fliplr(res)
	if y<0 :
		res=np.flipud(res)
	return res
# ------------------------------------------------EOFunc---------------------------------------------------------------
