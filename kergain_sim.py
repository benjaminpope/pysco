import numpy as np
import matplotlib.pyplot as plt
import pysco
from pysco.core import *
import fitsio
import triangle
from k2_epd_george import print_time

from time import time as clock
from old_diffract_tools import *
import pymultinest

from pysco.diffract_tools import shift_image_ft
from pysco.common_tasks import shift_image
from swiftmask import swiftpupil

import matplotlib as mpl


mpl.rcParams['figure.figsize']=(8.0,6.0)	#(6.0,4.0)
mpl.rcParams['font.size']= 18			   #10 
mpl.rcParams['savefig.dpi']=100			 #72 
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
nimages = 5 #100
nframes = 1

reso = rad2mas(wavel/(2*rprim))

print 'Minimum Lambda/D = %.3g mas' % reso

image, imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,seeing=None,verbose=False,\
							 centre_wavel=wavel,show_pupil=False,dust=False)

# image = recenter(image,sg_rad=25)
imsz = image.shape[0]

images = np.zeros((nimages,imsz,imsz))
k=0
show=False

'''----------------------------------------
Loop over a range of contrasts
----------------------------------------'''

contrast_list = [10,50,100,150,200,250,300]
ncalcs = len(contrast_list) * nframes

kseps, kthetas, kcons = np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dkseps, dkthetas, dkcons = np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

vseps, vthetas, vcons = np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)
dvseps, dvthetas, dvcons = np.zeros(ncalcs), np.zeros(ncalcs), np.zeros(ncalcs)

t0 = clock()
for trial, contrast in enumerate(contrast_list):
	print '\nSimulating for contrast %f' % contrast
	thistime = clock()

	sep, theta = 48, 45
	x,y = np.cos(theta*np.pi/180)*sep/spaxel, np.sin(theta*np.pi/180)*sep/spaxel

	print 'x',x,',y',y

	for j in range(nimages):
		if k == 10:
			print 'Up to', j
			show=False
			k=0
		psf, imagex = diffract(wavel,rprim,rsec,pos,piston=piston,spaxel=spaxel,verbose=False,\
									centre_wavel=wavel,show_pupil=show,dust=True,perturbation=None,amp=0.2)
	#	 psf = image
		images[j,:,:] = psf + shift_image_ft(psf,[-y,-x])/contrast#shift_image(psf,x=x,y=y,doRoll=True)/contrast

		imsz = image.shape[0]
		show=False
		k+=1
		  
	'''----------------------------------------
	Initialise pysco with a pupil model
	----------------------------------------'''

	# meter to pixel conversion factor
	scale = 1.0
	m2pix = mas2rad(spaxel) * imsz/ wavel * scale
	uv_samp = a.uv * m2pix + imsz/2 # uv sample coordinates in pixels

	x = a.mask[:,0]
	y = a.mask[:,1]

	'''----------------------------------------
	Extract Visibilities
	----------------------------------------'''


	rev = 1.0
	ac = shift(fft(shift(image)))
	ac /= (np.abs(ac)).max() / a.nbh

	uv_samp_rev=np.cast['int'](np.round(uv_samp))
	uv_samp_rev[:,0]*=rev
	data_cplx=ac[uv_samp_rev[:,1], uv_samp_rev[:,0]]

	vis2 = np.abs(data_cplx)
	vis2 /= vis2.max() #normalise to the origin

	mvis = a.RED/a.RED.max().astype('float')

	# kpd_phase = np.angle(data_cplx)/dtor
	# kpd_signal = np.dot(a.KerPhi, kpd_phase)

	kervises=np.zeros((nimages,KerGain.shape[0]))
	vis2s = np.zeros((nimages,vis2.shape[0]))
	kpd_signals = np.zeros((nimages,a.KerPhi.shape[0]))
	phases = np.zeros((nimages,vis2.shape[0]))
	randomGain = np.random.randn(np.shape(KerGain)[0],np.shape(KerGain)[1])

	for j in range(nimages):
	    image2 = images[j,:,:]
	    ac2 = shift(fft(shift(image2)))
	    ac2 /= (np.abs(ac2)).max() / a.nbh
	    data_cplx2=ac2[uv_samp_rev[:,1], uv_samp_rev[:,0]]

	    vis2b = np.abs(data_cplx2)
	    vis2b /= vis2b.max() #normalise to the origin
	    vis2s[j,:]=vis2b
	    
	#     log_data_complex_b = np.log(np.abs(data_cplx2))+1.j*np.angle(data_cplx2)
	    
	    phases[j,:] = np.angle(data_cplx2)/dtor
	    kervises[j,:] = np.dot(KerGain,vis2b/vis2)
	#     kervises[j,:] = np.dot(randomGain, np.sqrt(vis2b)-mvis)
	#     kpd_signals[j,:] = np.dot(a.KerPhi,np.angle(data_cplx2))/dtor
	#     kercomplexb = np.dot(KerBispect,log_data_complex_b)
	#     kervises_cplx[j,:] = np.abs(kercomplexb)

	'''----------------------------------------
	Extract Visibilities
	----------------------------------------'''

	paramlimits = [20.,80.,30.,60.,contrast/2.,contrast*2.]

	hdr = {'tel':'HST',
	      'filter':wavel,
	      'orient':0}

	def myprior(cube, ndim, n_params,paramlimits=paramlimits):
	    cube[0] = (paramlimits[1] - paramlimits[0])*cube[0]+paramlimits[0]
	    cube[1] = (paramlimits[3] - paramlimits[2])*cube[1]+paramlimits[2]
	    for j in range(2,ndim):
	        cube[j] = (paramlimits[5] - paramlimits[4])*cube[j]+paramlimits[4]
	        
	def kg_loglikelihood(cube,kgd,kge,kpi):
	    '''Calculate chi2 for single band kernel phase data.
	    Used both in the MultiNest and MCMC Hammer implementations.'''
	    vises = np.sqrt(pysco.binary_model(cube[0:3],kpi,hdr,vis2=True))
	    kergains = np.dot(KerGain,vises)
	    chi2 = np.sum(((kgd-kergains)/kge)**2)
	    return -chi2/2.

	def vis_loglikelihood(cube,vdata,ve,kpi):
	    '''Calculate chi2 for single band kernel phase data.
	    Used both in the MultiNest and MCMC Hammer implementations.'''
	    vises = pysco.binary_model(cube[0:3],kpi,hdr,vis2=True)
	    chi2 = np.sum(((vdata-vises)/ve)**2)
	    return -chi2/2.

	'''-----------------------------------------------
	Loop over a set of frames
	-----------------------------------------------'''

	for frame in range(nframes):
	
		'''-----------------------------------------------
		First do kernel amplitudes
		-----------------------------------------------'''

		if frame == 0:
			my_observable = np.mean(kervises,axis=0)
		else:
			my_observable = kervises[frame+1,:]

		my_error =      np.std(kervises,axis=0)
		
		def myloglike_kg(cube,ndim,n_params):
		    loglike = kg_loglikelihood(cube,my_observable,my_error,a)
		    # loglike = vis_loglikelihood(cube,my_observable,my_error,a)
		    return loglike

		parameters = ['Separation','Position Angle','Contrast']
		n_params = len(parameters)

		resume=False
		eff=0.3
		multi=True,
		max_iter= 0
		ndim = n_params

		pymultinest.run(myloglike_kg, myprior, n_params, wrapped_params=[1],verbose=True,resume=False)

		thing = pymultinest.Analyzer(n_params = n_params)
		s = thing.get_stats()

		this_j = trial*nframes + frame

		kseps[this_j], dkseps[this_j] = s['marginals'][0]['median'], s['marginals'][0]['sigma']
		kthetas[this_j], dkthetas[this_j] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		kcons[this_j], dkcons[this_j] = s['marginals'][2]['median'], s['marginals'][2]['sigma']

		print 'Kernel amplitudes done'

		'''-----------------------------------------------
		Now do visibilities
		-----------------------------------------------'''

		if frame == 0:
			my_observable = np.mean((vis2s/vis2)**2,axis=0)
		else:
			my_observable = (vis2s[frame+1,:]/vis2)**2

		my_error =      np.std((vis2s/vis2)**2,axis=0)

		def myloglike_vis(cube,ndim,n_params):
		    # loglike = kg_loglikelihood(cube,my_observable,my_error,a)
		    loglike = vis_loglikelihood(cube,my_observable,my_error,a)
		    return loglike

		pymultinest.run(myloglike_vis, myprior, n_params, wrapped_params=[1],verbose=True,resume=False)

		thing = pymultinest.Analyzer(n_params = n_params)
		s = thing.get_stats()

		this_j = trial*nframes + frame

		vseps[this_j], dvseps[this_j] = s['marginals'][0]['median'], s['marginals'][0]['sigma']
		vthetas[this_j], dvthetas[this_j] = s['marginals'][1]['median'], s['marginals'][1]['sigma']
		vcons[this_j], dvcons[this_j] = s['marginals'][2]['median'], s['marginals'][2]['sigma']

		print 'Visibilities done'

		print_time(clock()-thistime)

'''------------------------------------
Now save!
------------------------------------'''

vdata = Table({'Seps':vseps,
		 'Thetas':vthetas,
		 'Cons':vcons,
		 'Dseps':dvseps,
		 'Dthetas':dvthetas,
		 'Dcons':dvcons})

vdata.write('raw_vis_sims_%2f_%2f.csv' %  (contrasts.min(),contrasts.max()))

print 'Visibility fits saved to raw_vis_sims_%2f_%2f.csv' % (contrasts.min(),contrasts.max())

kdata = Table({'Seps':kseps,
		 'Thetas':kthetas,
		 'Cons':kcons,
		 'Dseps':dkseps,
		 'Dthetas':dkthetas,
		 'Dcons':dkcons})
kdata.write('kernel_amplitude_sims_%2f_%2f.csv' % (contrasts.min(),contrasts.max()))

print 'Kernel amplitude fits saved to kernel_amplitude_sims_%2f_%2f.csv' \
	%  (contrasts.min(),contrasts.max())

print 'Finished contrast loop'
print_time(clock()-t0)