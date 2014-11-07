import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits as pf
import copy
import pickle
import os
import sys
import pdb
import glob
import gzip
import pymultinest
import os, threading, subprocess
import matplotlib.pyplot as plt
import json
import oifits
import time
import emcee
import playdoh
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.special import j1, jv
import triangle

'''------------------------------------------------------------------------
cp_tools.py - a collection of functions useful for closure phase analysis
in Python. This includes mas2rad, rad2mas and phase_binary from pysco; 
it must share a directory with oifits.py, and depends on PyMultiNest,
MultiNest (Fortran) and emcee (the Python MCMC Hammer implementation).
------------------------------------------------------------------------'''

def mas2rad(x):
	''' Convenient little function to convert milliarcsec to radians '''
	return x*np.pi/(180*3600*1000)

# =========================================================================
# =========================================================================

def rad2mas(x):
	''' Convenient little function to convert radians to milliarcseconds '''
	return x/np.pi*(180*3600*1000)

# =========================================================================
# =========================================================================

def phase_binary(u, v, wavel, p):
	''' Calculate the phases observed by an array on a binary star
	----------------------------------------------------------------
	p: 3-component vector (+2 optional), the binary "parameters":
	- p[0] = sep (mas)
	- p[1] = PA (deg) E of N.
	- p[2] = contrast ratio (primary/secondary)
	
	optional:
	- p[3] = angular size of primary (mas)
	- p[4] = angular size of secondary (mas)

	- u,v: baseline coordinates (meters)
	- wavel: wavelength (meters)
	---------------------------------------------------------------- '''

	p = np.array(p)
	# relative locations
	th = (p[1] + 90.0) * np.pi / 180.0
	ddec =  mas2rad(p[0] * np.sin(th))
	dra  = -mas2rad(p[0] * np.cos(th))

	# baselines into number of wavelength
	x = np.sqrt(u*u+v*v)/wavel

	# decompose into two "luminosity"
	l2 = 1. / (p[2] + 1)
	l1 = 1 - l2
	
	# phase-factor
	phi = np.zeros(u.size, dtype=complex)
	phi.real = np.cos(-2*np.pi*(u*dra + v*ddec)/wavel)
	phi.imag = np.sin(-2*np.pi*(u*dra + v*ddec)/wavel)

	# optional effect of resolved individual sources
	if p.size == 5:
		th1, th2 = mas2rad(p[3]), mas2rad(p[4])
		v1 = 2*j1(np.pi*th1*x)/(np.pi*th1*x)
		v2 = 2*j1(np.pi*th2*x)/(np.pi*th2*x)
	else:
		v1 = np.ones(u.size)
		v2 = np.ones(u.size)

	cvis = l1 * v1 + l2 * v2 * phi
	phase = np.angle(cvis, deg=True)
	return np.mod(phase + 10980., 360.) - 180.0

# =========================================================================
# =========================================================================

def vis_disk(u, v, wavels, p):
	''' Calculate the visibilities observed by an array on a binary star
	----------------------------------------------------------------
	p: 3-component vector (+2 optional), the binary "parameters":
	- p[0] = angular size of star (mas)
	- p[1] = linear limb-darkening coefficient - not done yet
	- p[2] = quadratic limb-darkening coefficient - not done yet

	- u,v: baseline coordinates (meters)
	- wavel: wavelength (meters)
	---------------------------------------------------------------- '''

	p = np.array(p)
	# relative locations
	th1 = mas2rad(p[0])
	if np.size(p)>1:
		mu = p[1]
	else:
		mu = 0.
	# baselines into number of wavelength
	pref = 1./((1.-mu)/2. + mu/3.)
	eps = 1.e-20
	x = np.sqrt(u*u+v*v)/wavels+eps

	v1 = pref*((1.-mu)*j1(np.pi*th1*x)/(np.pi*th1*x) + mu*np.sqrt(np.pi/2.)*jv(1.5,np.pi*th1*x)/((np.pi*th1*x)**1.5))

	return np.abs(v1)

# =========================================================================
# =========================================================================

def cp_loglikelihood(params,u,v,wavel,t3data,t3err):
	'''Calculate loglikelihood for closure phase data.
	Used both in the MultiNest and MCMC Hammer implementations.'''
	cps = cp_model(params,u,v,wavel)
	chi2 = np.sum(((t3data-cps)/t3err)**2)
	loglike = -chi2/2
	return loglike

# =========================================================================
# =========================================================================

def cp_model(params,u,v,wavel):
	'''Function to model closure phases. Takes a parameter list, u,v triangles and a single wavelength.'''
	ndata = u.shape[0]
	phases = phase_binary(u.ravel(),v.ravel(),wavel,params)
	phases = np.reshape(phases,(ndata,3))
	cps = np.array(np.sum(phases,axis=1))
	return cps

# =========================================================================
# =========================================================================

def cphammer(cpo,ivar=[52., 192., 1.53],ndim=3,nwalcps=50,plot=False):

	'''Default implementation of emcee, the MCMC Hammer, for closure phase
	fitting. Requires a closure phase object cpo, and is best called with 
	ivar chosen to be near the peak - it can fail to converge otherwise.'''

	ivar = np.array(ivar)  # initial parameters for model-fit

	p0 = [ivar + 0.1*ivar*np.random.rand(ndim) for i in range(nwalcps)] # initialise walcps in a ball

	print 'Running emcee now!'

	t0 = time.time()

	sampler = emcee.EnsembleSampler(nwalcps, ndim, cp_loglikelihood, args=[cpo.u,cpo.v,cpo.wavel,cpo.t3data,cpo.t3err])
	sampler.run_mcmc(p0, 1000)

	tf = time.time()

	print 'Time elapsed =', tf-t0,'s'

	seps = sampler.flatchain[:,0]
	ths = sampler.flatchain[:,1]
	cs = sampler.flatchain[:,2]

	meansep = np.mean(seps)
	dsep = np.std(seps)

	meanth = np.mean(ths)
	dth = np.std(ths)

	meanc = np.mean(cs)
	dc = np.std(cs)

	print 'Separation',meansep,'pm',dsep,'mas'
	print 'Position angle',meanth,'pm',dth,'deg'
	print 'Contrast',meanc,'pm',dc

	if plot==True:

		plt.clf()

		paramnames = ['Separation','Position Angle','Contrast']
		paramdims = ['(mas)', '(deg)','Ratio']

		for i in range(ndim):
			plt.figure(i)
			plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
			plt.title(paramnames[i])
			plt.ylabel('Counts')
			plt.xlabel(paramnames[i]+paramdims[i])

		plt.show()

# =========================================================================
# =========================================================================

def cpnest(cpo,paramlimits=[20.,250.,0.,360.,1.0001,10],ndim=3,resume=False,eff=0.3,multi=True):

	'''Default implementation of a MultiNest fitting routine for closure 
	phase data. Requires a closure phase cpo object, parameter limits and 
	sensible keyword arguments for the multinest parameters. 

	This function does very naughty things creating functions inside this 
	function because PyMultiNest is very picky about how you pass it
	data.

	Optional parameter eff tunes sampling efficiency, and multi toggles multimodal 
	nested sampling on and off. Turning off multimodal sampling results in a speed 
	boost of ~ 20-30%. 

	'''

	def myprior(cube, ndim, n_params,paramlimits=paramlimits,kpo=0):
		cube[0] *= (paramlimits[1] - paramlimits[0])+paramlimits[0]
		cube[1] *= (paramlimits[3] - paramlimits[2])+paramlimits[2]
		cube[2] *= (paramlimits[5] - paramlimits[4])+paramlimits[4]

	def myloglike(cube, ndim, n_params):
		loglike = cp_loglikelihood(cube[0:3],cpo.u,cpo.v,cpo.wavel,cpo.t3data,cpo.t3err)
		return loglike

	parameters = ['Separation','Position Angle','Contrast']
	n_params = len(parameters)
	ndim = 3

	tic = time.time() # start timing

	#---------------------------------
	# now run MultiNest!
	#---------------------------------

	pymultinest.run(myloglike, myprior, n_params, wrapped_params=[1], resume=resume, verbose=True, sampling_efficiency=eff, multimodal=multi, n_iter_before_update=1000)

	# let's analyse the results
	a = pymultinest.Analyzer(n_params = n_params)
	s = a.get_stats()

	toc = time.time()

	if toc-tic < 60.:
		print 'Time elapsed =',toc-tic,'s'
	else: 
		print 'Time elapsed =',(toc-tic)/60.,'mins'

	# json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
	print
	print "-" * 30, 'ANALYSIS', "-" * 30
	print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )
	params = s['marginals']

	bestsep = params[0]['median']
	seperr = params[0]['sigma']

	bestth = params[1]['median']
	therr = params[1]['sigma']

	bestcon = params[2]['median']
	conerr = params[2]['sigma']
	
	print ''

	print 'Separation',bestsep,'pm',seperr
	print 'Position angle',bestth,'pm',therr
	print 'Contrast ratio',bestcon,'pm',conerr

	return s

# =========================================================================
# =========================================================================

def detec_limits(cpo,nsim=100,nsep=32,nth=20,ncon=32,smin='Default',smax='Default',
	cmin=1.0001,cmax=500.,addederror=0,threads=0,save=False):

	'''uses a Monte Carlo simulation to establish contrast-separation 
	detection limits given an array of standard deviations per closure phase.

	Because different separation-contrast grid points are entirely
	separate, this task is embarrassingly parallel. If you want to 
	speed up the calculation, use multiprocessing with a threads 
	argument equal to the number of available cores.

	Make nseps a multiple of threads! This uses the cores most efficiently.

	Hyperthreading (2x processes per core) in my experience gets a ~20%
	improvement in speed.

	Written by F. Martinache and B. Pope.'''

	#------------------------
	# first, load your data!
	#------------------------

	error = cpo.t3err + addederror

	u,v = cpo.u,cpo.v

	wavel = cpo.wavel

	ndata = cpo.ndata

	w = np.array(np.sqrt(u**2 + v**2))/wavel

	if smin == 'Default':
		smin = rad2mas(1./4/np.max(w))

	if smax == 'Default':
		smax = rad2mas(1./np.min(w))

	#------------------------
	# initialise Monte Carlo
	#------------------------

	temp_sngl = np.zeros(nsim)
	temp_binr = np.zeros(nsim)

	seps = smin + (smax-smin) * np.linspace(0,1,nsep)
	ths  = 0.0 + 360.0  * np.linspace(0,1,nth)
	cons = cmin  + (cmax-cmin)  * np.linspace(0,1,ncon)

	error_arr = np.outer(error, np.ones(nsim))
	rands = np.random.randn(ndata,nsim)

	#-----------------------------
	# Define your loopfit function
	#-----------------------------

	def loopfit(sep):
		'''Function for multiprocessing in detec_limits. Takes a 
		single separation and full angle, contrast lists.'''
		chi2_diff = np.zeros((nth,ncon,nsim))
		for j,th in enumerate(ths):
			for k,con in enumerate(cons):
				bin_cp = cp_model([sep,th,con],u, v, wavel)
					
					# binary cp model
					# ----------------------
					
				rnd_cp = bin_cp[:,np.newaxis] + error[:,np.newaxis]*rands
				chi2_sngl = np.sum((((rnd_cp)/ error[:,np.newaxis])**2),axis=0) 
				chi2_binr = np.sum((((rnd_cp-bin_cp[:,np.newaxis]) / error[:,np.newaxis])**2),axis=0)
				chi2_diff[j,k] = chi2_binr-chi2_sngl# note not i,j,k - i is for seps
		return chi2_diff

	#------------------------
	# Run Monte Carlo
	#------------------------

	tic = time.time() # start the clock

	if threads ==0:
		chi2_diff = np.zeros((nsep,nth,ncon,nsim))
		for i,sep in enumerate(seps):
			print("iteration # %3d: sep=%.2f" % (i, sep))
			chi2_diff[i,:,:,:]= loopfit(sep)
			toc = time.time()
			if i != 0:
				remaining =  (toc-tic)*(nsep-i)/float(i)
				if remaining > 60:
					print('Estimated time remaining: %.2f mins' % (remaining/60.))
				else: 
					print('Estimated time remaining: %.2f seconds' % (remaining))
	else:
		results = playdoh.map_async(loopfit,seps,cpu=threads)
		chi2_diff = results.get_results()
		for job in results.jobs:
			job.erase_all() # this line is super important! If you don't erase jobs, it fills up your hard drive! Check /.playdoh/jobs if this breaks.
		chi2_diff = np.array(chi2_diff)
	tf = time.time()
	if tf-tic > 60:
		print 'Total time elapsed:',(tf-tic)/60.,'mins'
	elif tf-tic <= 60:
		print 'Total time elapsed:',tf-tic,'seconds'

	ndetec = np.zeros((ncon, nsep))

	nc, ns = int(ncon), int(nsep)

	for k in range(nc):
		for i in range(ns):
			toto = (chi2_diff)[i,:,k,:]
			ndetec[k,i] = (toto < 0.0).sum()

	nbtot = nsim * nth

	ndetec /= float(nbtot)

	# ---------------------------------------------------------------
	#                        contour plot!
	# ---------------------------------------------------------------
	levels = [0.5,0.9, 0.99, 0.999]
	mycols = ('k', 'k', 'k', 'k')

	plt.figure(0)
	contours = plt.contour(ndetec, levels, colors=mycols, linewidth=2, 
                 extent=[smin, smax, cmin, cmax])
	plt.clabel(contours)
	plt.contourf(seps,cons,ndetec,levels,cmap=plt.cm.bone)
	plt.colorbar()
	plt.xlabel('Separation (mas)')
	plt.ylabel('Contrast Ratio')
	plt.title('Contrast Detection Limits')
	plt.draw()
	plt.show()

	data = {'levels': levels,
			'ndetec': ndetec,
			'seps'  : seps,
			'angles': ths,
			'cons'  : cons,
			'name'  : cpo.name}

	if save == True:
		file = 'limit_lowc'+cpo.name+'.pick'
		print file

		myf = open(file,'w')
		pickle.dump(data,myf)
		myf.close()

	return data

# =========================================================================
# =========================================================================

def binary_fit(cpo, p0):
    '''Performs a best binary fit search for the dataset.
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    ------------------------------------------------------------- '''
    
    if np.all(cpo.t3err == 0.0):
        print("Closure phase object instance is not calibrated.\n")
        soluce = leastsq(cpo.bin_fit_residuals, p0, args=(cpo),
                     full_output=1)
    else:
    	def lmcpmodel(index,params1,params2,params3):
    		params = [params1,params2,params3]
    		model = cp_model(params,cpo.u,cpo.v,cpo.wavel)
    		return model[index]
    	soluce = curve_fit(lmcpmodel,range(0,cpo.ndata),cpo.t3data,p0,sigma=cpo.t3err)
    cpo.covar = soluce[1]
    soluce[0][1] = np.mod(soluce[0][1],360.) # to get consistent position angle measurements
    return soluce

# =========================================================================
# =========================================================================

def bin_fit_residuals(params, cpo):
	'''Function for binary_fit without errorbars'''
	test = cp_model(params,cpo.u,cpo.v,cpo.wavel)
	err = (cpo.t3data - test)
	return err

# =========================================================================
# =========================================================================

def vis_loglikelihood(params,u,v,wavel,visdata,viserr):
	'''Calculate loglikelihood for closure phase data.
	Used both in the MultiNest and MCMC Hammer implementations.'''
	if all(params<0):
		return -np.inf
	vismodel = vis_model(params,u,v,wavel)**2. #remember to get vis2!
	chi2 = np.sum(((visdata-vismodel)/viserr)**2)
	loglike = -chi2/2
	return loglike

# =========================================================================
# =========================================================================

def vis_model(params,u,v,wavels):
	'''Function to model closure phases. Takes a parameter list, u,v triangles and a single wavelength.'''
	ndata = u.shape[0]
	vis = vis_disk(u.ravel(),v.ravel(),wavels,params)
	return vis

# =========================================================================
# =========================================================================

def visnest(cpo,paramlimits=[1.,20.,0.,1.],ndim=2,resume=False,eff=0.3,multi=True):

	'''Default implementation of a MultiNest fitting routine for closure 
	phase data. Requires a closure phase cpo object, parameter limits and 
	sensible keyword arguments for the multinest parameters. 

	This function does very naughty things creating functions inside this 
	function because PyMultiNest is very picky about how you pass it
	data.

	Optional parameter eff tunes sampling efficiency, and multi toggles multimodal 
	nested sampling on and off. Turning off multimodal sampling results in a speed 
	boost of ~ 20-30%. 

	'''

	def myprior(cube, ndim, n_params,paramlimits=paramlimits,kpo=0):
		cube[0] *= (paramlimits[1] - paramlimits[0])+paramlimits[0]
		cube[1] *= (paramlimits[3] - paramlimits[2])+paramlimits[2]

	def myloglike(cube, ndim, n_params):
		loglike = vis_loglikelihood(cube[0:3],cpo.u,cpo.v,cpo.wavels,cpo.vis2,cpo.vis2err)
		return loglike

	parameters = ['Diameter','Limb-Darkening']
	n_params = len(parameters)

	tic = time.time() # start timing

	#---------------------------------
	# now run MultiNest!
	#---------------------------------

	pymultinest.run(myloglike, myprior, n_params, resume=resume, verbose=True, sampling_efficiency=eff, multimodal=multi, n_iter_before_update=1000)

	# let's analyse the results
	a = pymultinest.Analyzer(n_params = n_params)
	s = a.get_stats()

	toc = time.time()

	if toc-tic < 60.:
		print 'Time elapsed =',toc-tic,'s'
	else: 
		print 'Time elapsed =',(toc-tic)/60.,'mins'

	# json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
	print
	print "-" * 30, 'ANALYSIS', "-" * 30
	print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )
	params = s['marginals']

	bestd = params[0]['median']
	derr = params[0]['sigma']

	bestmu = params[1]['median']
	muerr = params[1]['sigma']
	
	print ''

	print 'Diameter',bestd,'pm',derr
	print 'Limb-Darkening',bestmu,'pm',muerr

	return s

# =========================================================================
# =========================================================================

def vis2hammer(cpo,ivar=[1., 0.5],ndim=2,nwalcps=50,plot=False,nsteps=1000,burnin=100,paramlimits=[0.5,5.,0.,1.],muprior=[0.64,0.03]):

	'''Default implementation of emcee, the MCMC Hammer, for closure phase
	fitting. Requires a closure phase object cpo, and is best called with 
	ivar chosen to be near the peak - it can fail to converge otherwise.'''

	def lnprior(params):
	    if paramlimits[0] < params[0] < paramlimits[1] and paramlimits[2] < params[1] < paramlimits[3]:
	        return -0.5*((params[1]-muprior[0])/muprior[1])**2. #0.0
	    return -np.inf

	def lnprob(params,u,v,wavels,visdata,viserr):
		return lnprior(params) + vis_loglikelihood(params,u,v,wavels,visdata,viserr)

	ivar = np.array(ivar)  # initial parameters for model-fit

	p0 = [ivar + 0.1*ivar*np.random.rand(ndim) for i in range(nwalcps)] # initialise walcps in a ball

	print 'Running emcee now!'

	t0 = time.time()

	sampler = emcee.EnsembleSampler(nwalcps, ndim, lnprob, args=[cpo.u,cpo.v,cpo.wavels,cpo.vis2,cpo.vis2err])
	
	# burn in
	pos,prob,state = sampler.run_mcmc(p0, burnin)
	print 'Burnt in'
	sampler.reset()

	# restart
	sampler.run_mcmc(pos,nsteps)
	tf = time.time()

	print 'Time elapsed =', tf-t0,'s'

	ds = sampler.flatchain[:,0]
	mus = sampler.flatchain[:,1]

	meand = np.mean(ds)
	derr = np.std(ds)

	meanmu = np.mean(mus)
	dmu = np.std(mus)

	print 'Diameter',meand,'pm',derr,'mas'
	print 'Limb-Darkening',meanmu,'pm',dmu

	if plot==True:
		fig = triangle.corner(sampler.flatchain, labels=["$d$ (mas)", "$\mu$"])
		plt.show()

	return sampler.flatchain