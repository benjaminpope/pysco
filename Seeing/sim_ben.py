import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
import Image
import ImageDraw
from wfirst import wfirstpupil 
from jwstpupil import jwstpupil
from swiftmask import swiftpupil
from frebin import spaxel_scale

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2
fftfreq = np.fft.fftfreq

dtor = np.pi/180.0

'''------------------------------------------------
sim.py 

code to simulate WFIRST
------------------------------------------------'''

wavel = 1.0e-6
seeingfile = 'DSP_AO_seeing=0p65_L0=50_z=30_nph=100_500Hz_Td=3ms_1040pixels_2p2microns.fits'
spaxel = 16 #mas 
doseeing = False

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

'''----------------------------------------
Calculate an input pupil.
----------------------------------------'''

sz = 4096

pupil = np.ones((sz,sz),dtype='complex') + 0j

mask,xs,m2pix = wfirstpupil(sz=sz)#swiftpupil(sz=sz,whichmask=0)#
# mask = np.array(mask!=0,dtype='float') # for jwst

# get coordinates for the full un-cropped image
ys = xs 

xx,yy = np.meshgrid(xs,ys)

pix2m = 1./m2pix

freqs = shift(fftfreq(sz,d=pix2m))


'''----------------------------------------
Apply a phase screen.
----------------------------------------'''

if doseeing:

	seeing = pf.getdata(seeingfile)

	seeing = np.sqrt(shift(seeing)) + 0j # center and square root to get amplitudes; 0j to make it complex!

	seeingx = np.arange(2080)*0.05
	seeingx -= seeingx.max()
	sxx,syy = np.meshgrid(seeingx,seeingx)

	# #interpolate seeing to the correct basis

	noise = np.random.standard_normal(np.shape(seeing)) +0j # generate noise in pupil plane

	noise = shift(fft(shift(noise)))

	seeing *= noise # multiply amplitudes by pupil plane noise

	seeing = shift(ifft(shift(seeing)))

	interpfun = interp(seeingx,seeingx,seeing.real)

	seeing = interpfun(xs+xs.min(),ys+ys.min())+0j

	seeing = np.exp(1.j*seeing)

	pupil *= seeing 

	'''----------------------------------------
	Optionally, add extra phase pokes
	----------------------------------------'''

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
		cmap=plt.cm.gray,interpolation='none')
	plt.xlabel('m')
	plt.ylabel('m')
	plt.title('Input Phase Screen')
	cbar = plt.colorbar()
	plt.show()

'''----------------------------------------
Apply the pupil mask.
----------------------------------------'''

pupil *= (mask+0j)

'''----------------------------------------
Plot the pupil
----------------------------------------'''

plt.figure(1)
plt.clf()
plt.imshow(np.abs(pupil),extent=[xx.min(),xx.max(),xx.min(),xx.max()],
	cmap=plt.cm.gray,interpolation='none')
cbar = plt.colorbar()
# plt.scatter(poke[0],poke[1],s=5)
plt.xlim(xx.min(),xx.max())
plt.ylim(xx.min(),xx.max())
plt.xlabel('m')
plt.ylabel('m')
plt.title('Input Pupil')
plt.show()

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

pscale = (focx.max()-focx.min())/focx.size

image = focal.real**2 + focal.imag**2

imsz = image.shape[0]

# display image
plt.figure(2)
plt.clf()
plt.imshow(image**0.1,extent=[focx.min(),focx.max(),focx.min(),focx.max()],
	cmap=plt.cm.gray,interpolation='none')
plt.xlabel('mas')
plt.ylabel('mas')
plt.title('Input PSF')
cbar = plt.colorbar()
plt.show()

'''----------------------------------------
Re-bin to new spaxel scale
using Simon's function
----------------------------------------'''

rebin = spaxel_scale(image,pscale,spaxel)
rebinx = focx/pscale*spaxel 

# display image
plt.figure(3)
plt.clf()
plt.imshow(rebin**0.2,extent=[rebinx.min(),rebinx.max(),rebinx.min(),rebinx.max()],
	cmap=plt.cm.gray,interpolation='none')
plt.xlabel('mas')
plt.ylabel('mas')
plt.title('Input PSF, rebinned')
cbar = plt.colorbar()
plt.show()