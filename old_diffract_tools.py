import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
from frebin import spaxel_scale
from swiftmask import swiftpupil


shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq

dtor = np.pi/180.0

'''------------------------------------------------
diffract_tools.py 

code to simulate a narrowband image
------------------------------------------------'''

# =========================================================================
# =========================================================================

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

def kolmogorov_spectrum(sz,r0 = 0.01):

    xs, ys = np.linspace(-5./r0,5./r0,sz), np.linspace(-5./r0,5./r0,sz)

    xx, yy = np.meshgrid(xs,ys)

    rr = np.sqrt(xx**2+yy**2)

    newspec = shift(0.0229*r0**(5./3.)*(rr**(-11./3.)))
    newspec[~np.isfinite(newspec)] = 0

    return newspec

# =========================================================================
# =========================================================================

def screen(seeingfile,xs=None):
	'''Generate a phase screen'''
	try:
		seeing = pf.getdata(seeingfile)
	except:
		seeing = kolmogorov_spectrum(2080)

	seeing = np.sqrt(shift(seeing)) + 0j # center and square root to get amplitudes; 0j to make it complex!

	seeingx = np.arange(2080)*0.5
	seeingx -= seeingx.max()
	sxx,syy = np.meshgrid(seeingx,seeingx)

	# #interpolate seeing to the correct basis

	noise = np.random.standard_normal(np.shape(seeing))+0j # generate noise in pupil plane

	noise = shift(fft(shift(noise)))

	rprim = 5.093/2. # E-ELT

	seeing *= 2.*rprim*noise # multiply amplitudes by pupil plane noise
	# remember seeing is normalised to the pupil diameter in metres!

	seeing = shift(ifft(shift(seeing)))
	if xs == None:
		newx = np.linspace(-4*rprim,4*rprim,2080)
	else:
		newx = np.linspace(xs.min(),xs.max(),2080)

	interpfun = interp(newx,newx,seeing)

	return interpfun

# =========================================================================
# =========================================================================

def diffract(wavel,rprim,rsec,pos=[0,0],piston=100.e-9,spaxel=40.,seeing=None,verbose=True,
	show_pupil=False,telescope=None,centre_wavel=900e-9,dust=None,
	perturbation='phase',amp=0.3):
	'''Run a diffraction simulation!'''

	# wavel = params[0]
	# rprim = params[1]

	# spacing = 3.

	reso = rad2mas(wavel/(2*rprim))

	if verbose:
		print 'Lambda/D =',reso,'mas'

	'''----------------------------------------
	Calculate an input pupil.
	----------------------------------------'''

	sz = 4096/2

	pupil = np.ones((sz,sz),dtype='complex')

	# get coordinates for the full un-cropped image
	xs, ys = np.linspace(-sz/2,sz/2,sz), np.linspace(-sz/2,sz/2,sz)
	xs *= 4*rprim/xs.max()
	ys *= 4*rprim/ys.max()

	th = pos[1]*np.pi/180.
	poke = pos[0]*np.array([np.cos(th),np.sin(th)])

	xx,yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy**2)
	rr2 = np.sqrt((xx-poke[0])**2 + (yy-poke[1])**2)

	m2pix = sz/(xx.max()-xx.min())
	pix2m = 1./m2pix

	freqs = shift(fftfreq(sz,d=pix2m))

	'''----------------------------------------
	Apply a phase screen.
	----------------------------------------'''

	# seeing = np.exp(1.j*seeing)

	# pupil *= seeing 

	# # where do you poke the phases?
	# angle = 45
	# dist = 0.5

	# angle *= np.pi/180.
	# poke = np.array([np.sin(angle),np.cos(angle)])
	# poke = poke*(rprim-rsec)*dist + poke*rsec

	# #phase piston a 1x1 m region cetred at the poke point
	# pupil[np.sqrt((xx-poke[0])**2 +(yy-poke[1])**2)<3] *= np.exp(0+1j)

	# display image

	'''----------------------------------------
	Apply the pupil mask.
	----------------------------------------'''

	# now limit the bandpass of the pupil

	if telescope == 'swift':
		mask,xs,m2pix=swiftpupil(sz=sz)
		pix2m = 1./m2pix
		freqs = shift(fftfreq(sz,d=pix2m))

	else:
		mask = np.ones(rr.shape,dtype='complex')
		# mask *= np.exp(-(rr-(rprim+rsec)/2.)**2/2./gausswidth**2)
		mask /= mask.max()
		mask[rr>rprim] = 0
		mask[rr<rsec] = 0

		# if dust ==True:
		# 	# mask*= 1 + 0.6*np.sin((rr-(rprim+rsec)/2.)*10.)

		# 	interpfun = screen(seeing)

		# 	seeing = interpfun(xs+xs.min(),ys+ys.min())
		# 	seeing /= seeing.max()

		# 	# seeing = np.exp(seeing)

		# 	mask *= np.abs(seeing) 
		# 	mask /= mask.max()


	# mask = np.ones(rr1.shape,dtype='complex')
	# mask *= np.exp(-(rr-(rprim+rsec)/2.)**2/2./gausswidth**2)
	# mask /= mask.max()
	# mask[(rr1>rprim)* (rr2 > rprim)] = 0

	if dust:
		interpfun = screen(None,xs=xs+xs.min())
		scintillation = interpfun(xs+xs.min(),ys+ys.min())
		scintillation = (scintillation-scintillation.min())/(scintillation.max()-scintillation.min())
		scintillation *= mask
		keep = np.isfinite(scintillation) * (scintillation>0)
		scintillation -= np.median(scintillation[keep])
		pupil *= (1+amp*scintillation)

	pupil *= mask

	if perturbation == 'phase':
		if verbose:
			print 'perturbing phase'
		pupil[rr2<0.5] *= np.exp(2.*np.pi*1.j*piston/wavel)
	elif perturbation == 'amplitude':
		if verbose:
			print 'perturbing amplitude'
		pupil[rr2<0.5] *= 1.03

	# plt.figure(0)
	# plt.clf()
	# plt.imshow(np.abs(pupil),extent=[xx.min(),xx.max(),xx.min(),xx.max()],
	# 	cmap=plt.cm.gray,interpolation='none')
	# cbar = plt.colorbar()
	# # plt.scatter(poke[0],poke[1],s=5)
	# plt.xlim(xx.min(),xx.max())
	# plt.ylim(xx.min(),xx.max())
	# plt.xlabel('m')
	# plt.ylabel('m')
	# plt.title('Input Pupil')
	# plt.show()

	if show_pupil:
		toy = np.copy(pupil)
		toy[mask==0] = np.nan
		plt.imshow(np.abs(toy),extent=[xs.min(),xs.max(),xs.min(),xs.max()],
			interpolation='none')
		# plt.colorbar(ticks=np.linspace(0.98,1.03,11))
		# plt.clim([0.98,1.03])
		plt.colorbar()
		plt.xlim([-4,4])
		plt.ylim([-4,4])
		plt.xlabel('m')
		plt.ylabel('m')
		plt.title('Input Pupil')
		# cbar = plt.colorbar()
		plt.show()


	if verbose:
		rmsphase = np.sqrt(np.mean(np.angle(pupil[np.abs(pupil)>0]))**2)
		print 'RMS Phase %.3g rad' %rmsphase

	# insert in a padded array

	# newpupil = np.zeros((sz*4,sz*4),dtype='complex')
	# newpupil[3*sz/2:5*sz/2,3*sz/2:5*sz/2] = pupil
	# pupil = newpupil

	'''----------------------------------------
	Propagate to the first focal plane.
	----------------------------------------'''

	focal = shift(fft(shift(pupil)))

	# get coordinates for the full un-cropped image
	focx = rad2mas(freqs*wavel)
	focxx,focyy = np.meshgrid(focx,focx)
	focrr = np.sqrt(focxx**2 + focyy**2)

	image = focal.real**2 + focal.imag**2

	imsz = image.shape[0]

	# image = image[3*imsz/8:5*imsz/8,3*imsz/8:5*imsz/8]
	# imagex = focx[3*imsz/8:5*imsz/8]

	# display image
	# plt.figure(1)
	# plt.clf()
	# plt.imshow(image**0.2,extent=[focx.min(),focx.max(),focx.min(),focx.max()],
	# 	cmap=plt.cm.gray,interpolation='none')
	# plt.xlabel('mas')
	# plt.ylabel('mas')
	# plt.title('HARMONI input PSF')
	# cbar = plt.colorbar()
	# plt.show()

	'''----------------------------------------
	Rebin
	----------------------------------------'''

	pscale = (focx.max()-focx.min())/focx.size

	rescale = wavel/centre_wavel

	if verbose:
		print 'Image dimension', imsz
		print 'pscale',pscale
		print 'rescale', rescale
		print 'virtual spaxel size',spaxel*rescale

	# try:
	# 	rebin = fits_spaxel_scale(image2,pscale,spaxel,'config3.fits')
	# 	rebinx = focx/pscale*spaxel 
	# 	print 'saved!'
	# except:

	rebin = spaxel_scale(image,pscale,spaxel*rescale,verbose=verbose)

	rebin = rebin[rebin.shape[0]/2-128:rebin.shape[0]/2+128,rebin.shape[0]/2-128:rebin.shape[0]/2+128]

	rebinx = spaxel*np.arange(256)
	rebinx -= rebinx.max()/2.


	return rebin, rebinx