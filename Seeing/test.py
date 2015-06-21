from display import displayImage
from pupil import getNHolesMask
from pupil import getAnnulusMask
from pupil import getFullMask
from sim import getFocalImage
from common_tasks import mas2rad
import numpy as np
from astropy.io import fits

pupilSize=5.0		# pupil diameter in meters (NB: MUST be smaller than phasescreen)
scale=100.0 		# scale factor (pixels/m)
plateScale=11.5		# plate scale (mas/pixel)
wl=1.6e-6			#base wavelength
chip_px=512			# number of elements per chip (1dim)

#input full pupil Mask 
mask=getFullMask(diam=pupilSize,border=0.0,scale=scale)
pupil = np.ones((scale*pupilSize,scale*pupilSize),dtype='complex') + 0j # plane wave
# scale factor for FFT
#scale_f=1.22*wl/(np.power(pupilSize,2)*mas2rad(plateScale)*scale)
scale_f=wl/(pupilSize*mas2rad(plateScale))


#phaseScreen
f = fits.open('/suphys/latyshev/Code/idl_toy_imaging/screen.fits')
phases=f[0].data

a,p=getFocalImage(pupil,mask,seeing=np.cos(phases)+1j*np.sin(phases),compl=False,scale=scale_f,cropPix=chip_px)


displayImage(abs(a)**0.2,
	axisSize=[-plateScale*len(a)/2,plateScale*len(a)/2,-plateScale*len(a)/2,plateScale*len(a)/2],
	xlabel='mas', ylabel='mas', title='Focal Plane Amplitudes **0.2',showColorbar=True,flipY=True,cmap='gray') 	
	

f = fits.open('/suphys/latyshev/Code/idl_toy_imaging/image.fits')
a1=f[0].data


a2=a**2
a1_norm=(a1-a1.min())/(a1.max()-a1.min())
a2_norm=(a2-a2.min())/(a2.max()-a2.min())

displayImage(abs(a2_norm)**0.1-abs(a1_norm)**0.1,
	axisSize=[-plateScale*len(a)/2,plateScale*len(a)/2,-plateScale*len(a)/2,plateScale*len(a)/2],
	xlabel='mas', ylabel='mas', title='Focal Plane Amplitudes **0.2',showColorbar=True,flipY=True,cmap='gray') 