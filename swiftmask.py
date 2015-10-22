import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
import Image
import ImageDraw
import pickle

def swiftpupil(sz=1024, whichmask=0, show_pupil = False):
	'''------------------------------------------------
	jwstpupil.py 

	code to generate a SWIFT pupil mockup
	------------------------------------------------'''

	#things to edit
	rprim = 5.093/2.
	rsec = 0.36*rprim
	secpos = np.array([0,0])
	# spiderthick = 0.025*rsec

	'''-------------------------------------------
	Initialise your arrays
	-------------------------------------------'''

	mask = np.ones((sz,sz))

	# get coordinates for the full un-cropped image
	xs, ys = np.linspace(-sz/2,sz/2,sz), np.linspace(-sz/2,sz/2,sz)
	xs *= 6*rprim/xs.max()
	ys *= 6*rprim/ys.max()

	xx,yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy**2)
	rr2 = np.sqrt((xx-secpos[0])**2 + (yy-secpos[1])**2)

	m2pix = sz/(xx.max()-xx.min())
	pix2m = 1./m2pix

	'''-------------------------------------------
	Put in the primary and secondary sizes
	-------------------------------------------'''

	mask /= mask.max()
	mask[rr>rprim] = 0
	mask[rr2<rsec] = 0

	'''-------------------------------------------
	Now do the spiders
	-------------------------------------------'''

	if whichmask == 0:
		angles = [90, 180, 322-90]
		spiderthicks = rsec/4.*np.array([0.2,0.6,0.2])
		angles = [180]
		spiderthicks = rsec/4.*np.array([0.6])
	elif whichmask == 1:
		angles = [180, 270, 23]
		spiderthicks = rsec/4.*np.array([0.1,0.4,0.1])
	else:
		angles = [90, 180, 322-90]
		spiderthicks = rsec/4.*np.array([0.2,0.6,0.2])

	img = Image.fromarray(mask)
	draw = ImageDraw.Draw(img)

	for j, angle in enumerate(angles):
		
		angle *= np.pi/180.

		start = 1.1*rprim*np.array([np.cos(angle),np.sin(angle)])-xx.min()
		stop = np.array([0,0])-xx.min()

		normvec = np.array([start[1]-stop[1],stop[0]-start[0]])*spiderthicks[j]
		corners = np.round(m2pix*np.array([start+normvec,start-normvec,stop-normvec,stop+normvec]))

		#convert to pixels

		draw.polygon([tuple(p) for p in corners], fill=0)

	newmask = np.asarray(img)


	'''-------------------------------------------
	Display
	-------------------------------------------'''

	if show_pupil:

		plt.figure(0)
		plt.imshow(newmask,cmap=plt.cm.gray,extent=[xx.min(),xx.max(),xx.min(),xx.max()])
		plt.xlabel('x (m)')
		plt.ylabel('y (m)')
		plt.title('SWIFT Pupil')
		plt.draw()
		plt.show()

	return newmask,xs,m2pix
