import numpy as np
import matplotlib.pyplot as plt
import pysco
from pysco.core import *
import fitsio
from k2_epd_george import print_time

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

def make_ellipse(semi_axis,ecc,thick,sz=1024,pscale=12.):
	
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
	ff = psf_temp/psf_temp.sum()+con*dummy
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
wavel = 1.e-6
rprim = 5.093/2.#36903.e-3/2.
rsec= 1.829/2.
pos = [0,0] #m, deg
spaxel = 12
piston = 0
final_sz = 1024

nimages = 200
nframes = nimages-1

reso = rad2mas(wavel/(2*rprim))

print 'Minimum Lambda/D = %.3g mas' % reso

image, imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,seeing=None,verbose=False,\
                             centre_wavel=wavel,show_pupil=True,dust=False,sz=4096,final_sz=final_sz)

# image = recenter(image,sg_rad=25)
imsz = image.shape[0]

images = np.zeros((nimages,imsz,imsz))
psfs = np.zeros((nimages,imsz,imsz))

k=0
show=False

'''----------------------------------------
Loop over a range of contrasts
----------------------------------------'''

contrast_list = [1.,1.1,1.5,2.,3.,4.,4.5,5.,5.5,6.,6.5,7.,7.5,8.,8.5,9.,9.5,10.,12.,15.,17.,18.,20.]
ncalcs = len(contrast_list)

ksemis, keccs, kthetas, kthicks, kcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dksemis, dkeccs, dkthetas, dkthicks, dkcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

vsemis, veccs, vthetas, vthicks, vcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dvsemis, dveccs, dvthetas, dvthicks, dvcons = np.zeros(ncalcs), np.zeros(ncalcs),np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

t0 = clock()

true_vals = (400.,0.95,50)

for j in range(nimages):
    if k == 10:
        print 'Up to', j
        show=True
        k=0
    psfs[j,:,:], imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,verbose=False,\
                                centre_wavel=wavel,show_pupil=False,dust=True,perturbation=None,
                           amp=0.2,final_sz=final_sz)
    imsz = image.shape[0]
    show=False
    k+=1
      
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

rev = 1.0
ac = shift(fft(shift(image)))
ac /= (np.abs(ac)).max() / a.nbh

uv_samp_rev=np.cast['int'](np.round(uv_samp))
uv_samp_rev[:,0]*=rev
data_cplx=ac[uv_samp_rev[:,1], uv_samp_rev[:,0]]

vis2 = np.abs(data_cplx)
vis2 /= vis2.max() #normalise to the origin

mvis = a.RED/a.RED.max().astype('float')

# calibrator
vis2js = np.zeros((nimages,vis2.shape[0]))

for j in range(nimages):
	image = psfs[j,:,:]
	ac = shift(fft(shift(image)))
	ac /= (np.abs(ac)).max() / a.nbh
	data_cplx=ac[uv_samp_rev[:,1], uv_samp_rev[:,0]]

	vis2cj = np.abs(data_cplx)
	vis2cj /= vis2 #normalise to the origin
	vis2js[j,:]=vis2cj

vis2c = np.mean(vis2js,axis=0)

for trial, contrast in enumerate(contrast_list):
	print '\nSimulating for contrast %f' % contrast
	thistime = clock()

	for j in range(nimages):
		images[j,:,:] = make_disk(psfs[j,:,:],true_vals,contrast)
		  
	'''----------------------------------------
	Extract Visibilities
	----------------------------------------'''

	kervises=np.zeros((nimages,KerGain.shape[0]))
	vis2s = np.zeros((nimages,vis2.shape[0]))

	for j in range(nimages):
		image2 = images[j,:,:]
		ac2 = shift(fft(shift(image2)))
		ac2 /= (np.abs(ac2)).max() / a.nbh
		data_cplx2=ac2[uv_samp_rev[:,1], uv_samp_rev[:,0]]

		vis2b = np.abs(data_cplx2)
		vis2b /= vis2 #normalise to the origin
		vis2s[j,:]= vis2b
		
		kervises[j,:] = np.dot(KerGain,vis2b)#-np.dot(KerGain,vis2c)

	'''----------------------------------------
	Extract Visibilities
	----------------------------------------'''

	paramlimits = [100.,10000.,0.,0.99,-90.,90.,0.02,0.49,contrast/4.,contrast*4.]

	hdr = {'tel':'HST',
		  'filter':wavel,
		  'orient':0}

	def vis_model(cube,kpi):
		u, v = (kpi.uv/wavel).T
		unresolved = 1./(1.+cube[4])
		flux_ratio = cube[4]/(1.+cube[4])
		vises = vis_ellipse_thin(cube[0],cube[1],cube[2],cube[0]*cube[3],u,v)
		norm = vis_ellipse_thin(cube[0],cube[1],cube[2],cube[0]*cube[3],np.array([1e-5]),np.array([1e-5]))
		vises = (vises/norm *flux_ratio + unresolved)
		return vises

	def kg_loglikelihood(cube,kgd,kge,kpi):
		'''Calculate chi2 for single band kernel phase data.
		Used both in the MultiNest and MCMC Hammer implementations.'''
		vises = vis_model(cube,kpi)
		kergains = np.dot(KerGain,vises)
		chi2 = np.sum(((kgd-kergains)/kge)**2)
		return -chi2/2.

	def vis_loglikelihood(cube,vdata,ve,kpi):
		'''Calculate chi2 for single band kernel phase data.
		Used both in the MultiNest and MCMC Hammer implementations.'''
		vises = vis_model(cube,kpi)**2.
		chi2 = np.sum(((vdata-vises)/ve)**2)
		return -chi2/2.

	### define prior and loglikelihood

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

	addederror = 0.001 # in case there are bad frames
	my_error =	  np.sqrt(np.std(kervises,axis=0)**2+addederror**2)
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
	max_iter= 20000
	ndim = n_params

	pymultinest.run(myloglike_kg, myprior, n_params,wrapped_params=[2],
		verbose=True,resume=False,max_iter=max_iter)

	thing = pymultinest.Analyzer(n_params = n_params)
	try:
		s = thing.get_stats()

		ksemis[trial], dksemis[trial] = s['marginals'][0]['median'], s['marginals'][0]['sigma']
		keccs[trial], dkeccs[trial] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		kthetas[trial], dkthetas[trial] = s['marginals'][2]['median'], s['marginals'][2]['sigma']
		kthicks[trial], dkthicks[trial] = s['marginals'][3]['median'], s['marginals'][3]['sigma']
		kcons[trial], dkcons[trial] = s['marginals'][4]['median'], s['marginals'][4]['sigma']

		stuff = thing.get_best_fit()
		best_params = stuff['parameters']
		true_params = [true_vals[0],true_vals[1],0.,true_vals[0]/true_vals[2],contrast]

		model = np.dot(KerGain,vis_model(best_params,a))
		true_model = np.dot(KerGain,vis_model(true_params,a))

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

	my_observable = np.mean((vis2s)**2,axis=0)

	print '\nDoing raw visibilities'
	addederror = 0.001
	my_error =	  np.sqrt(np.std((vis2s)**2,axis=0)**2+addederror**2)
	print 'Error:', my_error

	def myloglike_vis(cube,ndim,n_params):
		try:
			loglike = vis_loglikelihood(cube,my_observable,my_error,a)
			return loglike
		except:
			return -np.inf

	thistime = clock()

	pymultinest.run(myloglike_vis, myprior, n_params,wrapped_params=[2],
		verbose=True,resume=False,max_iter=max_iter)

	thing = pymultinest.Analyzer(n_params = n_params)
	try:
		s = thing.get_stats()
		vsemis[trial], dvsemis[trial] = s['marginals'][0]['median'], s['marginals'][0]['sigma']
		veccs[trial], dveccs[trial] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		vthetas[trial], dvthetas[trial] = s['marginals'][2]['median'], s['marginals'][2]['sigma']
		vthicks[trial], dvthicks[trial] = s['marginals'][3]['median'], s['marginals'][3]['sigma']
		vcons[trial], dvcons[trial] = s['marginals'][4]['median'], s['marginals'][4]['sigma']

		stuff = thing.get_best_fit()
		best_params = stuff['parameters']
		true_params = [true_vals[0],true_vals[1],0.,true_vals[0]/true_vals[2],contrast]
		
		model = vis_model(best_params,a)**2.
		true_model = vis_model(true_params,a)**2.

		plt.clf()
		plt.errorbar(my_observable,true_model,xerr=my_error,color='b',alpha=0.5,
			ls='',markersize=10,linewidth=2.5)
		plt.errorbar(my_observable,model,xerr=my_error,color='k',
			ls='',markersize=10,linewidth=2.5)
		plt.xlabel('Measured Square visibilities')
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