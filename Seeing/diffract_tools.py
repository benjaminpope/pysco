import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
from frebin import *
from wfirst import *
from jwstpupil import *
import time

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

def screen(seeingfile):
	'''Generate a phase screen'''
	seeing = pf.getdata(seeingfile)

	seeing = np.sqrt(shift(seeing)) + 0j # center and square root to get amplitudes; 0j to make it complex!
	
	seeingx = np.arange(2080)*0.5
	seeingx -= seeingx.max()
	sxx,syy = np.meshgrid(seeingx,seeingx)

	# #interpolate seeing to the correct basis

	noise = np.random.standard_normal(np.shape(seeing))+0j # generate noise in pupil plane
	
	noise = shift(fft(shift(noise)))

	rprim = 36903.e-3/2. # E-ELT

	seeing *= 2.*rprim*noise # multiply amplitudes by pupil plane noise
	# remember seeing is normalised to the pupil diameter in metres!

	seeing = shift(ifft(shift(seeing))).real

	interpfun = interp(seeingx,seeingx,seeing)

	return interpfun

# =========================================================================
# =========================================================================

def phase_jwst(pupil,xs,phases):
	'''Generate a JWST phase screen. Format: piston, tip (x), tilt (y). Piston 
	in rad, tip-tilt in total amplitude in radians. '''

	tol = 0.01
	d = 1.32
	
	pscreen = np.copy(pupil)
	pscreen[pscreen>0.1] = 1 # make a screen object
	tiptemplate = np.ones(np.shape(pscreen))*xs
	tilttemplate = tiptemplate.T 

	for j in range(18):
		j+=1
		piston, tip, tilt = phases[0,j], phases[1,j]/d, phases[2,j]/d
		try:
			pscreen[np.abs(pupil-j*1.)<=tol] += tip*tiptemplate[np.abs(pupil-j*1.)<=tol]
			pscreen[np.abs(pupil-j*1.)<=tol] += tilt*tilttemplate[np.abs(pupil-j*1.)<=tol]
			pscreen[np.abs(pupil-j*1.)<=tol] -= np.mean(pscreen[np.abs(pupil-j*1.)<=tol])
			pscreen[np.abs(pupil-j*1.)<=tol] += piston
		except:
			print 'Failed'

	pscreen[pupil<=tol] = 0

	return pscreen

# =========================================================================
# =========================================================================

def diffract(wavel=2.2e-6,spaxel=4.,seeingfile=None,rprim = 2.4,sz=4096,tel='wfirst',
	jwstphases=np.zeros((3,19))):
	'''Run a diffraction simulation!'''

	# wavel = params[0]
	# rprim = params[1]
	# rsec = params[2]
	# stop = params[3] #stop size in lambda/D
	# spaxel = params[4]
	
	'''----------------------------------------
	Calculate an input pupil.
	----------------------------------------'''

	# sz = 4096

	if tel == 'wfirst':
		pupil,xs,m2pix = wfirstpupil(sz=sz)#np.ones((sz,sz),dtype='complex')+0j
		rprim = 2.4
	elif tel == 'jwst':
		pupil,xs,m2pix = jwstpupil(sz=sz)
		rprim = 6.5/2.
	else: 
		print 'Telescope must be wfirst or jwst'

	reso = rad2mas(wavel/(2*rprim))

	print 'Lambda/D =',reso,'mas'

	pupil = pupil+0j

	# get coordinates for the full un-cropped image
	# xs, ys = np.linspace(-sz/2,sz/2,sz), np.linspace(-sz/2,sz/2,sz)
	# xs *= 3*rprim/xs.max()
	# ys *= 3*rprim/ys.max()
	ys = xs

	xx,yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy**2)

	# m2pix = sz/(xx.max()-xx.min())
	pix2m = 1./m2pix

	freqs = shift(fftfreq(sz,d=pix2m))

	'''----------------------------------------
	Apply a phase screen.
	----------------------------------------'''

	if tel == 'jwst':
		phases = jwstphases 
		pscreen = phase_jwst(pupil,xs,phases)
		pscreen = np.exp(1.j*(pscreen))
		pupil[pupil>=0.01] = 1.+0j # normalise
		pupil *= pscreen

	else:
		interpfun = screen(seeingfile)

		seeing = interpfun(xs+xs.min(),ys+ys.min())

		seeing = np.exp(1.j*(seeing))

		pupil *= seeing 

	pupil[np.abs(pupil.real)<=0.01] = 0+0j

	# # where do you poke the phases?
	# angle = 45
	# dist = 0.5

	# angle *= np.pi/180.
	# poke = np.array([np.sin(angle),np.cos(angle)])
	# poke = poke*(rprim-rsec)*dist + poke*rsec

	# #phase piston a 1x1 m region cetred at the poke point
	# pupil[np.sqrt((xx-poke[0])**2 +(yy-poke[1])**2)<3] *= np.exp(0+1j)

	# display image
	plt.figure(0)
	plt.clf()
	plt.imshow(np.angle(pupil),extent=[xs.min(),xs.max(),xs.min(),xs.max()],
		interpolation='none',origin='lower')
	plt.xlabel('m')
	plt.ylabel('m')
	plt.title('Input Phase Screen')
	cbar = plt.colorbar()
	plt.draw()
	plt.show()

	'''----------------------------------------
	Apply the pupil mask.
	----------------------------------------'''

	# now limit the bandpass of the pupil

	# mask = np.ones(rr.shape,dtype='complex')
	# #mask *= np.exp(-(rr-(rprim+rsec)/2.)**2/2./gausswidth**2)
	# # mask /= mask.max()
	# mask[rr>rprim] = 0
	# mask[rr<rsec] = 0

	# pupil *= mask

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

	rmsphase = np.sqrt(np.mean(np.angle(pupil[np.abs(pupil)>0]))**2)
	print 'RMS Phase',rmsphase,'rad'

	# insert in a padded array

	# newpupil = np.zeros((sz*4,sz*4),dtype='complex')
	# newpupil[3*sz/2:5*sz/2,3*sz/2:5*sz/2] = pupil
	# pupil = newpupil

	'''----------------------------------------
	Propagate to the first focal plane.
	----------------------------------------'''

	focal = shift(fft((pupil)))

	# get coordinates for the full un-cropped image
	focx = rad2mas(freqs*wavel)
	focxx,focyy = np.meshgrid(focx,focx)
	focrr = np.sqrt(focxx**2 + focyy**2)

	image = focal.real**2 + focal.imag**2

	imsz = image.shape[0]

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

	print 'plate scale = ', pscale
	# spaxel = 4. # mas

	# try:
	# 	rebin = fits_spaxel_scale(image2,pscale,spaxel,'config3.fits')
	# 	rebinx = focx/pscale*spaxel 
	# 	print 'saved!'
	# except:
	rebin = spaxel_scale(image,pscale,spaxel)
	rebinx = focx/pscale*spaxel

	return rebin, rebinx