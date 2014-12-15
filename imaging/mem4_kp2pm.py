#! /usr/bin/env python
"""
Basically just trying to rewrite Mike I.'s MaxEnt code that's available 
online. To me tt makes sense to make this an object since the variables are 
passed around so much. I'll keep much of the comments from the original 
code and try to fill in some more.

Details:
- The required data input must contain an image file in the first fits 
	extension that is a "v2pm" matrix for kernel phases -- I'm calling it kp2pm
- The header must contain a keyword 'PXSCALE' presumably in mas
- The second extension is a dimension-2 array storing average kernel 
	phase and kernel phase error

by Alex Greenbaum agreenba@pha.jhu.edu Nov 2014
"""
# ===========================
# Legacy comments:
# ===========================
# This Maximum-Entropy method, like any MEM method, has 2 regularizers:
# one for total flux and one for entropy. In an ideal situation, total 
# flux is fixed from the data (not here as we are in the high-contrast
# limit with an unresolved point source), and the MEM regularizer can
# be set to give a reduced chi-squared = 1. Often in a MEM technique, 
# the total flux is constrained using a prior. This is the approach
# that we will take here.

import numpy as np
import pylab as plt
import pyfits
import sys

HOME = "/Users/agreenba/"

def entropy(vector,prior):
    return np.sum(vector/prior*np.log(vector/prior))

def grad_entropy(vector,prior):
	return (np.log(vector/prior) + 1)/prior 

class MemImage():
	#Create a Maximum-Entropy image based on an input fits
	#file and a prior.
	"""
	Methods:
	read_data -- loads in important info from the prepared fits file
	mem_image -- the major driver: calls all the descent functions and 
				 returns the reconstructed image
	make_iterstep -- calculates the new terms in each step
	fline -- returns chi^2 for a given small movement
	line_search -- calls fline on various step sizes to find the best chi^2
	
	"""

	def __init__(self, filename, dataobj=None, **kwargs):
		"""
		Initializing this function requires a set of keyword arguments:
			imsize
			alpha
			gain
			niter
			prior
		If not provided, set to default.
		"""
		# Optional inputs are:
		# alpha: A starting value for the MEM functional multiplier (default=1.0)
		# gain: The servo gain for adjusting alpha to achieve chi^2=1
		# niter: The number of iterations. 
		self.filename = filename
		self.read_data(filename)

		keys = kwargs.keys()
		if ('alpha' in keys):
			self.alpha = kwargs['alpha']
		else:
			self.alpha = 1.0
		if ('gain' in keys):
			self.gain = kwargs['gain']
		else:
        	#A difference of 1.0 in chi^2 gives a difference of 0.1 in log(alpha)
			self.gain = 0.1
		if ('niter' in keys):
			self.niter = kwargs['niter']
		else:
			self.niter = 200
		if ('imsize' in keys):
			self.imsize = kwargs['imsize']
		else:
			self.imsize = 80
		if ('prior' in keys):
			self.prior = kwargs['prior']
		else:
			# This was default set in the original code
			self.prior = np.ones(self.imsize**2)/self.imsize**2.0/50
		print "Initialized:", self.imsize, self.kp.shape, self.kperr.shape

	def read_data(self, filename):
		"""
		gets out the relevant info from 
		"""
		hdulist = pyfits.open(filename)
		
		# the only necessary header data
		self.pxscale = hdulist[0].header['PXSCALE']
		
		# load in the kp data, we'll have to flatten this 
		self.kp2pm = hdulist[0].data
		self.imsize = self.kp2pm.shape[1]
		
		# Sizes will be used later maybe?
		self.dimx = self.kp2pm.shape[2]
		self.dimy = self.kp2pm.shape[1]
		
		# for plotting
		self.extent = [self.dimx/2*self.pxscale,-self.dimx/2*self.pxscale,\
					   -self.dimy/2*self.pxscale,self.dimy/2*self.pxscale]
		
		# Now flatten
		self.kp2pm = self.kp2pm.reshape([self.kp2pm.shape[0],self.dimx*self.dimy])
		
		# Now store the kernel phases and errors
		try:
			kpdata = hdulist[1].data
			self.kp = kpdata[0]
			self.kperr = kpdata[1]
		except: 
			print 'no kp data in',filename
		hdulist.close()

	def mem_image(self):
		"""
		Will iterate to a solution
		"""
		# Start with the prior (pm = 'pixel matrix')
		self.pm = self.prior.copy()
		start_i=5 #still figuring this one out...
		for i in range(self.niter):
			# Give the user a print statement every 10 iteration to see how they are doing
			if ( (i+1) % 10 == 0):
				print 'Done: ' + str(i+1) + ' of ' + str(self.niter) +\
				' iterations. Chi^2: ' + str(self.chi2) + ' alpha: ' +str(self.alpha)
			# =======================================================================
			#---- From original code: ---- 
			#While this loop runs, we need to adjust alpha so that chi^2=1.
			#Lower values of alpha give lower chi^2. Not knowing how to do this,
			#lets use a servo-loop approach.
			if (i >= start_i):
				chi2diff = 1 - self.chi2/len(self.kp) #The difference in chi-squared
				self.alpha = self.alpha*np.exp(self.gain * chi2diff)
			# =======================================================================
			self.make_iterstep()
			# For the first iteration:
			if (i == 0):
				self.steepestdirection = -self.grad.copy()[:] #The direction of the minimum
				self.conjugatedirection = self.steepestdirection.copy()
			# Otherwise follow the Polak-Ribiere method
			else:
				# ---------------------------- #
				#---- from original code: ----
				#The following is the Polak-Ribiere Nonlinear conjugant gradient method. 
				#(http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)
				#It seems to be only about a factor of 2 faster than steepest descent.
				self.lastdir = self.steepestdirection.copy()
				self.steepestdirection = -self.grad.copy()
				# metric 'beta_n' on wikipedia
				self.descentparam = np.sum(np.ma.masked_invalid(self.steepestdirection*\
						(self.steepestdirection - self.lastdir)))/\
						np.sum(np.ma.masked_invalid(self.steepestdirection*self.lastdir))
				self.descentparam = np.max(self.descentparam,0)
				# updates steepest direction (originally 'p', s_n on wikipedia)
				self.conjugatedirection = self.steepestdirection + \
						self.descentparam*self.conjugatedirection
			# store the previous pixel matrix:
			self.lastpm = self.pm.copy()
			self.kp2conj = np.dot(self.kp2pm,self.conjugatedirection)
			# Find us the best stepsize
			self.minstep = self.line_search()
			self.pm = self.lastpm + self.minstep*self.conjugatedirection
		# Now set alpha to zero and keep everything else you've worked so hard to get
		self.alpha=0
		self.make_iterstep()
		print "Chi2: " + str(self.chi2/len(self.kp))
		self.reconstructedimage = self.pm.reshape(self.imsize, self.imsize)
		return self.reconstructedimage

	def make_iterstep(self):
		"""
		"This is the function we are trying to minimize." - original code
		This calculates a new set of kernel phases in the approach to a solution
		We feed in whatever "pixel solution" I'm calling pm (was previously 
		called 'z') to generate 'kp_mod' (previously called theta_mod) & statistics

		The original functions that did this were "f" and "grad_f"
		"""
		self.pm[np.isnan(self.pm)] = 0 # this was necessary for me at one point
		self.kp_mod = np.dot(self.kp2pm, self.pm)
		self.chi2 = np.sum((self.kp_mod - self.kp)**2/self.kperr**2)
		self.errvect = (self.kp_mod - self.kp)/self.kperr**2
		# entropy function is called
		self.stat = self.chi2 + self.alpha*entropy(self.pm,self.prior)

		#The gradient of the function we are trying to minimize.
		# the dot product of our 'error' with the kp2pm matrix + entropy term
		self.grad = (2*np.dot(self.errvect.reshape(1,self.kp.size),self.kp2pm)+\
					self.alpha*grad_entropy(self.pm,self.prior))[0]

	def fline(self,step):
		"""
		move in a direction & calculate chi^2
		I believe t is our step in the line search (alpha_k?)...
		"""
		tmppm = self.pm.copy()
		tmppm = self.lastpm + self.conjugatedirection*step
		newchi2 = np.sum((self.kp_mod + self.kp2conj*step-self.kp)**2 / self.kperr**2) + \
					self.alpha*entropy(tmppm,self.prior)
		return newchi2

	def line_search(self):
		#---- from original code: ----
		#This is a golden section search with a fixed 30 iterations.
		#Returns the t value of the minimum.
		# ---------------------------- 
		# Wikipedia says to choose a variable alph_k to minimize 
		# 	h(alph_k) = f(x_k + alph*p_k) where p_k is the descent direction
		# 	and then update x_k+1 = x_k + alph_k*p_k 
		#	until grad(f(x_k))<tolerance
		#
		# in our case we trying to get the best "pixel matrix" (or image)
		# that represents our data
		niter=30
		if (min(self.conjugatedirection) > 0):
			#This shouldn't really happen
			histep = 2*np.max(self.pm)/np.max(self.conjugatedirection)
		else:
			# need better variable names for ts, tmax, wtf are these...
			division = -self.pm/self.conjugatedirection.clip(-1e12,-1e-12)
			#A somewhat arbitrary limit below to prevent log(0)
			histep = np.min(division) * (1-1e-6)
		# Why these numbers??
		startingpoint =  (1 + np.sqrt(5))/2.0
		res_startingpoint = 2 - startingpoint
		# I guess three different steps to compare
		midstep = histep/startingpoint
		lostep = 0
		# 3 computed moves
		hichi2 = self.fline(histep)
		midchi2 = self.fline(midstep)
		lochi2 = self.fline(lostep)
		for i in range(niter):
			# Check if distance from 'histep' to the mid step size is greater
			if (histep - midstep > midstep - lostep):
				# choose a step that is our midstep + a little bit 
				beststep = midstep + res_startingpoint * (histep - midstep)
				bestchi2 = self.fline(beststep)
				# is our chosen stepsize less than the middle one?
				# if so, bump up starting with mid value
				if bestchi2<midchi2:
					lochi2 = midchi2
					midchi2 = bestchi2
					lostep = midstep
					midstep = beststep
				# otherwise bump down with best chosen step at the top
				else:
					hichi2 = bestchi2
					histep = beststep
			# If distance to mid step to lowest step is larger:
			else:
				# choose a step that is our midstep - a little bit 
				beststep = midstep - res_startingpoint * (midstep - lostep)
				bestchi2 = self.fline(beststep)
				# Is our chosed stepsize greater than the middle one?
				# if so bump up the lowest step to the chosen step
				if midchi2<bestchi2:
					lochi2 = bestchi2
					lostep = beststep
				# If mid value is greater then make that the top
				# and bump the rest down
				else:
					hichi2 = midchi2
					midchi2 = bestchi2
					histep = midstep
					midstep = beststep
		# Now we'll return the step size that produces the lowest chi^2
		return [lostep,midstep,histep][np.argmin([lochi2,midchi2,hichi2])]

if __name__ == "__main__":
	# Initialize the function with whatever keywords I want to set, 
	# choices from: 
	#	imsize -- the size of the image in pixels, needs to match input
	#			  file 'pixel matrix' size
	#	alpha  -- A starting value for the MEM functional multiplier (default=1.0)
	#	gain   -- The servo gain for adjusting alpha to achieve chi^2=1
	#	niter  -- number of iterations
	#	prior  -- as mentioned at the start of this script, the prior
	#			  is used to contrain the flux
	myfitsfile = "LkCa15_K2010_implane.fits"
	## This does is all
	trial = MemImage(myfitsfile, niter=400)
	im = trial.mem_image()
	## the rest is a plot
	plt.imshow(im, interpolation='nearest',cmap=plt.get_cmap('gist_heat'),extent=trial.extent)
	plt.plot(0,0,'w*', ms=15)
	plt.axis(trial.extent)
	plt.xlabel('Delta RA (milli-arcsec)', fontsize='large')
	plt.ylabel('Delta Dec (milli-arcsec)', fontsize='large')
	print "Total contrast (mags): " + str(-2.5*np.log10(np.sum(trial.pm)))
	plt.show()

