import numpy as np
import matplotlib.pyplot as plt
import pysco
from pysco.core import *
import fitsio

from time import time as clock
from old_diffract_tools import *
import pymultinest

from pysco.diffract_tools import shift_image_ft
from pysco.common_tasks import shift_image
from swiftmask import swiftpupil

import matplotlib as mpl
from astropy.table import Table


mpl.rcParams['figure.figsize']=(8.0,6.0)	#(6.0,4.0)
mpl.rcParams['font.size']= 18			   #10 
mpl.rcParams['savefig.dpi']=200			 #72 
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq
fftn = np.fft.fftn
rfftn = np.fft.rfftn
dtor = np.pi/180.0

# =========================================================================
# =========================================================================

def print_time(t):
		if t>3600:
			print 'Time taken: %d h %d m %3f s'\
			% (np.int(np.floor(t/3600)), np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60))
		elif t>60:
			print 'Time taken: %d m %3f s' % (np.int(np.floor(np.mod(t,3600)/60)),np.mod(t,60) )
		else:
			print 'Time taken: %3f s' % t

# =========================================================================
# =========================================================================

'''------------------------------------------------------------
kergain_sim.py 

Automate a simulation of the effectiveness of raw visibility 
fitting versus kernel amplitudes

------------------------------------------------------------'''

pupil = 'plain'

try:
	a = pysco.kpi('./geometry/'+pupil+'model.pick')
	print 'Loaded kernel phase object'
except:
	a = pysco.kpi('./geometry/'+pupil+'.txt')
	a.name = 'Test'
	a.save_to_file('./geometry/'+pupil+'model.pick')

nbuv, nbh  = a.nbuv, a.nbh

try:
	KerGain = np.loadtxt('KerGain_plain.csv')
	print 'Loaded kernel amplitude matrix'
except:
	gtfm = np.abs(a.TFM)
	U, S, Vh = np.linalg.svd(gtfm.T, full_matrices=1) 

	S1 = np.zeros(nbuv)
	S1[0:nbh-1] = S
	nkg  = np.size(np.where(abs(S1) < 1e-3))
	print nkg

	KGCol  = np.where(abs(S1) < 1e-3)[0]
	KerGain = np.zeros((nkg, nbuv)) # allocate the array
	for i in range(nkg):
		KerGain[i,:] = (Vh)[KGCol[i],:]

	np.savetxt('KerGain_plain.csv',KerGain)
	print 'saved'

def make_ellipse(semi_axis,ecc,thick,sz=256,pscale=36.):
	
	semi_axis, thick = semi_axis/pscale, thick/pscale
	
	b = semi_axis*np.sqrt(1-ecc**2.)
	
	bmin = (semi_axis-thick)*np.sqrt(1-ecc**2)
	
	x = np.arange(sz)-sz/2.
	
	xx, yy = np.meshgrid(x,x)
	
	outer = (np.sqrt((xx/semi_axis)**2 + (yy/b)**2)< 1)
	
	inner = (np.sqrt((xx/(semi_axis-thick))**2 + (yy/bmin)**2) >1)
	   
	plain = np.ones((sz,sz))
	
	plain[~(outer*inner)] = 0
	
	return plain/plain.sum()
	
from scipy.special import j1

def vis_gauss(d,u,v):
	d = mas2rad(d)
	return np.exp(-(np.pi*d*np.sqrt(u**2+v**2))**2/4./np.log(2))

def vis_ud(d,u,v):
	r = np.sqrt(u**2+v**2)
	t = 2*j1(np.pi*d*r)/(np.pi*d*r)
	t[r <=(1/d*1e-5)] = 1.
	return t

def vis_ellipse_disk(semi_axis,ecc,theta,u,v):
	semi_axis = mas2rad(semi_axis)
	thetad = np.pi*theta/180.
	u1, v1 = u*np.cos(thetad)+v*np.sin(thetad), -u*np.sin(thetad)+v*np.cos(thetad)
	
	ad, bd = semi_axis, semi_axis*np.sqrt(1-ecc**2.)
	
	u1, v1 = u1*ad, v1*bd
	
	return vis_ud(0.5,u1,v1)
	
def vis_ellipse_thin(semi_axis,ecc,theta,thick,u,v):
	
	ad, bd = semi_axis, semi_axis*np.sqrt(1.-ecc**2.)
	a2, b2 = semi_axis-thick, (semi_axis-thick)*np.sqrt(1.-ecc**2)
	n1, n2 = ad*bd, a2*b2
	return vis_ellipse_disk(semi_axis,ecc,theta,u,v)-n2/n1*vis_ellipse_disk(semi_axis-thick,ecc,theta,u,v)

def vis_ellipse_gauss(semi_axis,thick,gausswidth,ecc,theta,u,v):
	return vis_gauss(gausswidth,u,v)*vis_ellipse_thin(semi_axis,thick,ecc,theta,u,v)

def my_convolve_2d(array1,array2):
	return shift(ifft(fft(shift(array1))*fft(shift(array2))))

def my_gauss_blur(array1,gausswidth):
	gausswidth *= spaxel
	s = np.shape(array1)[0]
	x = np.arange(s)-s/2.
	xx,yy = np.meshgrid(x,x)
	rr = np.sqrt(xx**2 + yy**2)
	gauss = np.exp(-(rr/gausswidth)**2)
	
	return np.abs(my_convolve_2d(array1,gauss))
	
def mk_star_with_ring(psf_temp,ring,con):
	dummy = np.abs(my_convolve_2d(ring,psf_temp))
	dummy /= dummy.sum()
	ff = psf_temp/psf_temp.sum()+dummy/con
	return ff/ff.sum()	

def make_disk(psf_temp,params,contrast):
	dummy = make_ellipse(*params)
	return mk_star_with_ring(psf_temp,dummy,contrast)

###-----------------------------------------
### now initialize a simulation
###-----------------------------------------

'''------------------------------
First, set all  your parameters.
------------------------------'''
print '\nSimulating a basic PSF'
wavel = 2.5e-6
rprim = 5.093/2.#36903.e-3/2.
rsec= 1.829/2.
pos = [0,0] #m, deg
spaxel = 36.
piston = 0

nimages = 200

reso = rad2mas(wavel/(2*rprim))

print 'Minimum Lambda/D = %.3g mas' % reso

image, imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,seeing=None,verbose=False,\
							 show_pupil=False,mode=None)
# image = recenter(image,sg_rad=25)
imsz = image.shape[0]

images = np.zeros((nimages,imsz,imsz))
psfs = np.zeros((nimages,imsz,imsz))

'''----------------------------------------
Loop over a range of contrasts
----------------------------------------'''

contrast_list = np.linspace(5.,200.,20)

ncalcs = len(contrast_list)

ksemis, keccs, kthetas, kthicks, kcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dksemis, dkeccs, dkthetas, dkthicks, dkcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

vsemis, veccs, vthetas, vthicks, vcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dvsemis, dveccs, dvthetas, dvthicks, dvcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

t0 = clock()

true_vals = (300.,0.95,100)

amp = 0.1

try:
	dummy = fitsio.FITS('psf_cube_scint_%.2f_wavel_%.2f.fits' % (amp,wavel*1e6))
	psfs = dummy[0][:,:,:]
	print 'Loaded PSFs'
except:
	print 'Creating PSFs'
	for j in range(nimages):
		psfs[j,:,:], imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,
			verbose=False,show_pupil=show,mode='amp',
			perturbation=None,amp=amp)
	fitsio.write('psf_cube_scint_%.2f_wavel_%.2f.fits' % (amp,wavel*1e6),psfs)

imsz = image.shape[0]

print_time(clock()-t0)

'''----------------------------------------
Initialise pysco with a pupil model
----------------------------------------'''

# meter to pixel conversion factor
scale = 1.0
m2pix = mas2rad(spaxel) * imsz/ wavel * scale
uv_samp = a.uv * m2pix + imsz/2 # uv sample coordinates in pixels

x = a.mask[:,0]
y = a.mask[:,1]

rev = 1
ac = shift(fft(shift(image)))
ac /= (np.abs(ac)).max() / a.nbh

uv_samp_rev=np.cast['int'](np.round(uv_samp))
uv_samp_rev[:,0]*=rev
data_cplx=ac[uv_samp_rev[:,1], uv_samp_rev[:,0]]

vis2 = np.abs(data_cplx)
vis2 /= vis2.max() #normalise to the origin

'''----------------------------------------
Now loop over simulated disks
----------------------------------------'''

for trial, contrast in enumerate(contrast_list):
	print '\nSimulating for contrast %f' % contrast
	thistime = clock()

	true_params = [true_vals[0]*4.,true_vals[1],0.,true_vals[2]/float(true_vals[0]),contrast]

	for j in range(nimages):
		images[j,:,:] = make_disk(psfs[j,:,:],true_vals,contrast)
		imsz = images.shape[1]

	'''----------------------------------------
	Extract Visibilities
	----------------------------------------'''

	kervises = np.zeros((nimages,KerGain.shape[0]))
	vis2s = np.zeros((nimages,vis2.shape[0]))

	for j in range(nimages):
	    image2 = images[j,:,:]
	    ac2 = shift(fft(shift(image2)))
	    ac2 /= (np.abs(ac2)).max() / a.nbh
	    data_cplx2=ac2[uv_samp_rev[:,1], uv_samp_rev[:,0]]

	    vis2b = np.abs(data_cplx2)
	    vis2b /= vis2b.max() #normalise to the origin
	    vis2s[j,:]=vis2b
	    
	    kervises[j,:] = np.dot(KerGain,vis2b/vis2-1.)

	'''----------------------------------------
	Now Model
	----------------------------------------'''

	paramlimits = [50.,10000.,0.,0.99,-90.,90.,0.02,0.49,contrast/4.,contrast*4.]

	hdr = {'tel':'HST',
		  'filter':wavel,
		  'orient':0}

	def vis_model(cube,kpi):
		con = 1./cube[4]
		u, v = (kpi.uv/wavel).T
		unresolved = 1./(1.+con)
		flux_ratio = con/(1.+con)
		vises = vis_ellipse_thin(cube[0],cube[1],cube[2],cube[0]*cube[3],u,v)
		norm = vis_ellipse_thin(cube[0],cube[1],cube[2],cube[0]*cube[3],np.array([1e-5]),np.array([1e-5]))
		vises = (vises/norm *flux_ratio + unresolved)
		return vises

	### define prior and loglikelihood

	def kg_loglikelihood(cube,kgd,kge,kpi):
		'''Calculate chi2 for single band kernel phase data.
		Used both in the MultiNest and MCMC Hammer implementations.'''
		vises = vis_model(cube,kpi)
		kergains = np.dot(KerGain,vises-1.)
		chi2 = np.sum(((kgd-kergains)/kge)**2)
		return -chi2/2.

	def vis_loglikelihood(cube,vdata,ve,kpi):
		'''Calculate chi2 for single band kernel phase data.
		Used both in the MultiNest and MCMC Hammer implementations.'''
		vises = vis_model(cube,kpi)**2.
		chi2 = np.sum(((vdata-vises)/ve)**2)
		return -chi2/2.

	def myprior(cube, ndim, n_params,paramlimits=paramlimits):
		cube[0] = (paramlimits[1] - paramlimits[0])*cube[0]+paramlimits[0]
		cube[1] = (paramlimits[3] - paramlimits[2])*cube[1]+paramlimits[2]
		cube[2] = (paramlimits[5] - paramlimits[4])*cube[2]+paramlimits[4]
		cube[3] = (paramlimits[7] - paramlimits[6])*cube[3]+paramlimits[6]
		cube[4] = (paramlimits[9] - paramlimits[8])*cube[4]+paramlimits[8]
			
	'''-----------------------------------------------
	First do kernel amplitudes
	-----------------------------------------------'''

	my_observable = np.mean(kervises,axis=0)

	addederror = 0.000001 # in case there are bad frames
	my_error =	  np.sqrt((np.std(kervises,axis=0))**2+addederror**2)
	print 'Error:', my_error 
	
	def myloglike_kg(cube,ndim,n_params):
		try:
			loglike = kg_loglikelihood(cube,my_observable,my_error,a)
			return loglike
		except:
			return -np.inf 

	parameters = ['Semi-major axis','Eccentricity','Position Angle', 'Thickness','Contrast']
	n_params = len(parameters)
	resume=False
	eff=0.3
	multi=True,
	max_iter= 10000
	ndim = n_params

	pymultinest.run(myloglike_kg, myprior, n_params,wrapped_params=[2],
		verbose=False,resume=False,max_iter=max_iter)

	thing = pymultinest.Analyzer(n_params = n_params)
	try:
		s = thing.get_stats()

		ksemis[trial], dksemis[trial] = s['marginals'][0]['median']/4., s['marginals'][0]['sigma']/4.
		keccs[trial], dkeccs[trial] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		kthetas[trial], dkthetas[trial] = s['marginals'][2]['median'], s['marginals'][2]['sigma']
		kthicks[trial], dkthicks[trial] = s['marginals'][3]['median'], s['marginals'][3]['sigma']
		kcons[trial], dkcons[trial] = s['marginals'][4]['median'], s['marginals'][4]['sigma']

		stuff = thing.get_best_fit()
		best_params = stuff['parameters']
		print 'Best params (kg):', best_params

		ksemis[trial] = best_params[0]/4.
		keccs[trial] = best_params[1]
		kthetas[trial] = best_params[2]
		kthicks[trial] = best_params[3]
		kcons[trial] = best_params[4]

		model = np.dot(KerGain,vis_model(best_params,a)-1.)
		true_model = np.dot(KerGain,vis_model(true_params,a)-1.)

		plt.clf()
		plt.errorbar(my_observable,true_model,xerr=my_error,color='b',alpha=0.5,
			ls='',markersize=10,linewidth=2.5)
		plt.errorbar(my_observable,model,xerr=my_error,color='k',
			ls='',markersize=10,linewidth=2.5)
		plt.xlabel('Measured Kernel Amplitudes')
		plt.ylabel('Model Kernel Amplitudes')
		plt.title('Model Fit: Kernel Amplitudes, Contrast %.1f' % contrast)
		plt.savefig('kpfit_%.1f_con.png' % contrast)

	except:
		print 'Failed!'

		ksemis[trial], dksemis[trial] = 0,0
		keccs[trial], dkeccs[trial] = 0,0
		kthetas[trial], dkthetas[trial] = 0,0
		kthicks[trial], dkthicks[trial] = 0,0
		kcons[trial], dkcons[trial] = 0,0



	print 'Kernel amplitudes done'
	print_time(clock()-thistime)
	print ''

	'''-----------------------------------------------
	Now do visibilities
	-----------------------------------------------'''

	my_observable = np.mean((vis2s/vis2)**2,axis=0)

	print '\nDoing raw visibilities'
	my_error =	  np.sqrt((np.std((vis2s/vis2)**2,axis=0))**2+addederror**2)
	print 'Error:', my_error

	def myloglike_vis(cube,ndim,n_params):
		try:
			loglike = vis_loglikelihood(cube,my_observable,my_error,a)
			return loglike
		except:
			return -np.inf

	thistime = clock()

	pymultinest.run(myloglike_vis, myprior, n_params,wrapped_params=[2],
		verbose=False,resume=False,max_iter=max_iter)

	thing = pymultinest.Analyzer(n_params = n_params)
	try:
		s = thing.get_stats()
		vsemis[trial], dvsemis[trial] = s['marginals'][0]['median']/4., s['marginals'][0]['sigma']/4.
		veccs[trial], dveccs[trial] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		vthetas[trial], dvthetas[trial] = s['marginals'][2]['median'], s['marginals'][2]['sigma']
		vthicks[trial], dvthicks[trial] = s['marginals'][3]['median'], s['marginals'][3]['sigma']
		vcons[trial], dvcons[trial] = s['marginals'][4]['median'], s['marginals'][4]['sigma']

		stuff = thing.get_best_fit()
		best_params = stuff['parameters']
		print 'Best params (vis):', best_params

		vsemis[trial] = best_params[0]/4.
		veccs[trial] = best_params[1]
		vthetas[trial] = best_params[2]
		vthicks[trial] = best_params[3]
		vcons[trial] = best_params[4]

		model = vis_model(best_params,a)**2.
		true_model = vis_model(true_params,a)**2.

		plt.clf()
		plt.errorbar(my_observable,true_model,xerr=my_error,color='b',alpha=0.5,
			ls='',markersize=10,linewidth=2.5)
		plt.errorbar(my_observable,model,xerr=my_error,color='k',
			ls='',markersize=10,linewidth=2.5)
		plt.xlabel('Measured Square Visibilities')
		plt.ylabel('Model Square Visibilities')
		plt.title('Model Fit: Visibilities, Contrast %.1f' % contrast)
		plt.savefig('visfit_%.1f_con.png' % contrast)
	except:
		print 'Failed'
		vsemis[trial], dvsemis[trial] = 0,0
		veccs[trial], dveccs[trial] = 0,0
		vthetas[trial], dvthetas[trial] = 0,0
		vthicks[trial], dvthicks[trial] = 0,0
		vcons[trial], dvcons[trial] = 0,0


	print 'Visibilities done'

	print_time(clock()-thistime)

'''------------------------------------
Now save!
------------------------------------'''

cmin, cmax = np.min(contrast_list), np.max(contrast_list)

vdata = Table({'Semis':vsemis,
		 'Eccs':veccs,
		 'Thetas':vthetas,
		 'Thicks':vthicks,
		 'Cons':vcons,
		 'Dsemis':dvsemis,
		 'Deccs':dveccs,
		 'Dthetas':dvthetas,
		 'Dthicks':dvthicks,
		 'Dcons':dvcons})

vdata.write('raw_vis_disk_sims_%.0f_%.0f.csv' %  (cmin,cmax))

print 'Visibility fits saved to raw_vis_disk_sims_%.0f_%.0f.csv' % (cmin,cmax)

kdata = Table({'Semis':ksemis,
		 'Eccs':keccs,
		 'Thetas':kthetas,
		 'Thicks':kthicks,
		 'Cons':kcons,
		 'Dsemis':dksemis,
		 'Deccs':dkeccs,
		 'Dthetas':dkthetas,
		 'Dthicks':dkthicks,
		 'Dcons':dkcons})

kdata.write('kernel_amplitude_disk_sims_%.0f_%.0f.csv' % (cmin,cmax))

print 'Kernel amplitude fits saved to kernel_amplitude_disk_sims_%.0f_%.0f.csv' \
	%  (cmin,cmax)

print 'Finished contrast loop'
print_time(clock()-t0)