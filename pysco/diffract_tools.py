import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
from frebin import *
# from wfirst import *
# from jwstpupil import *
from simpupil import *
import time
from common_tasks import shift_image
from astropy.io import fits

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq

dtor = np.pi/180.0

'''------------------------------------------------
diffract_tools.py 

code to simulate a narrowband image
------------------------------------------------'''


def subpix_fftshift(ftim,xy,shift):
	"""Shift an image by fractional pixels using an FFT
	(why isn't this in scipy?)"""

	return np.fft.irfft2(ftim*np.exp(1j*(xy[0]*shift[0] + xy[1]*shift[1])))


def shift_image_ft(image,shift):
	"""Shift an image by fractional pixels using an FFT
	(why isn't this in scipy?)"""

	calim_ft = np.fft.rfft2(image[:,:])
	x = np.arange(calim_ft.shape[1],dtype=np.double)
	y = ((np.arange(calim_ft.shape[0]) + calim_ft.shape[0]/2.0) % calim_ft.shape[0]) - calim_ft.shape[0]/2.0
	x *= 2*np.pi/calim_ft.shape[0]
	y *= 2*np.pi/calim_ft.shape[0]
	xy = np.meshgrid(x,y)

	return subpix_fftshift(calim_ft,xy,shift)

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

def make_binary(sep,theta,contrast,spaxel=25.2,wavel=2.145e-6,sz=4096,tel='palomar'):

	psf, xx = diffract(wavel=wavel,spaxel=spaxel,sz=sz,tel=tel)

	x,y = np.cos(theta*np.pi/180)*sep/spaxel, np.sin(theta*np.pi/180)*sep/spaxel

	print 'x',x,',y',y
		
	companion = shift_image_ft(psf,[-y,-x])/contrast

	binary_image = psf + companion - companion.min()#shift_image(psf,x=x,y=y,doRoll=True)/contrast

	return binary_image/binary_image.max(), xx

# =========================================================================
# =========================================================================

def diffract(wavel=2.145e-6,spaxel=25.2,seeingfile=None,sz=4096,tel='palomar',phases=False):
	'''Run a diffraction simulation!'''

	# wavel = params[0]
	# rprim = params[1]
	# rsec = params[2]
	# stop = params[3] #stop size in lambda/D
	# spaxel = params[4]
	
	'''----------------------------------------
	Calculate an input pupil.
	----------------------------------------'''

	if tel == 'palomar':
		pupil,xs,m2pix = palomarpupil(sz=sz)#np.ones((sz,sz),dtype='complex')+0j
		rprim = 5.093/2.  * 15.4/16.88 		
	elif tel == 'wfirst':
		pupil,xs,m2pix = wfirstpupil(sz=sz)#np.ones((sz,sz),dtype='complex')+0j
		rprim = 2.4
	elif tel == 'jwst':
		pupil,xs,m2pix = jwstpupil(sz=sz)
		rprim = 6.5/2.
	elif tel == 'wfc3':
		pupil,xs,m2pix = wfc3pupil(sz=sz)
		rprim = 1.2
	else: 
		print 'Telescope must be palomar, wfirst or jwst'

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
	if phases:
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

		rmsphase = np.sqrt(np.mean(np.angle(pupil[np.abs(pupil)>0]))**2)
		print 'RMS Phase',rmsphase,'rad'

	# # where do you poke the phases?
	# angle = 45
	# dist = 0.5

	# angle *= np.pi/180.
	# poke = np.array([np.sin(angle),np.cos(angle)])
	# poke = poke*(rprim-rsec)*dist + poke*rsec

	# #phase piston a 1x1 m region cetred at the poke point
	# pupil[np.sqrt((xx-poke[0])**2 +(yy-poke[1])**2)<3] *= np.exp(0+1j)


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
	# plt.title('Input PSF')
	# cbar = plt.colorbar()
	# plt.show()

	'''----------------------------------------
	Rebin
	----------------------------------------'''

	pscale = (focx.max()-focx.min())/focx.size

	print 'Native plate scale = ', pscale
	# spaxel = 4. # mas

	# try:
	# 	rebin = fits_spaxel_scale(image2,pscale,spaxel,'config3.fits')
	# 	rebinx = focx/pscale*spaxel 
	# 	print 'saved!'
	# except:
	rebin = spaxel_scale(image,pscale,spaxel)
	rebinx = focx/pscale*spaxel
	try:
		rebin = rebin[rebin.shape[0]/2-256:rebin.shape[0]/2+256,rebin.shape[1]/2-256:rebin.shape[1]/2+256]
	except:
		rebin = rebin[rebin.shape[0]/2-128:rebin.shape[0]/2+128,rebin.shape[1]/2-128:rebin.shape[1]/2+128]

	return rebin, rebinx

def imageToFits(image,path='./',filename='image.fits',
	tel='simu',pscale=25.2,odate='Jan 1, 2000', otime= "0:00:00.00",
	tint=1.0,coadds=1,RA=0.0,DEC=0.0,wavel=2.145e-6,orient=0.0,clobber=True) :
	''' saving generated image to a fits file '''
	prihdr = fits.Header()
	prihdr.append(('TELESCOP',tel))
	prihdr.append(('PSCALE',pscale))
	prihdr.append(('ODATE',odate))
	prihdr.append(('OTIME',otime))
	prihdr.append(('TINT',tint))
	prihdr.append(('FNAME',filename))
	prihdr.append(('COADDS',coadds))
	prihdr.append(('RA',RA))
	prihdr.append(('DEC',DEC))
	prihdr.append(('FILTER',wavel))
	prihdr.append(('ORIENT',orient))
	hdu = fits.PrimaryHDU(image,prihdr)
	hdu.writeto(path+filename,clobber=clobber)