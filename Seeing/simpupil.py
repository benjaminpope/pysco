import numpy as np 
import matplotlib.pyplot as plt 
import pyfits as pf
from scipy.interpolate import RectBivariateSpline as interp 
import Image
import ImageDraw
import pickle

def palomarpupil(sz=1024,spiders=True):
	'''------------------------------------------------
	simpupil.py 

	generates a Palomar pupil

	Owes a lot to Anthony Cheetham's IDL script
	------------------------------------------------'''

	#things to edit
	rmax   = 2.3918*1.5 # from optimization #5.093/2.  * 15.4/16.88        # outer diameter:      5.093 m
	rmin   = rmax/2.11885 # 1.829/2. * 7.6/6.08         # central obstruction: 1.829 m
	thick  = 0.2292*(rmax-rmin)/2.# 0.83 *15.4/16.88/4.         # adopted spider thickness (meters)


	'''-------------------------------------------
	Initialise your arrays
	-------------------------------------------'''

	mask = np.ones((sz,sz))

	# get coordinates for the full un-cropped image
	xs, ys = np.linspace(-sz/2,sz/2,sz), np.linspace(-sz/2,sz/2,sz)
	xs *= 7*rmax/xs.max()
	ys *= 7*rmax/ys.max()

	xx,yy = np.meshgrid(xs,ys)
	rr = np.sqrt(xx**2 + yy**2)

	m2pix = sz/(xx.max()-xx.min())
	pix2m = 1./m2pix

	'''-------------------------------------------
	Put in the primary and secondary sizes
	-------------------------------------------'''

	mask /= mask.max()
	mask[rr>rmax] = 0
	mask[rr<rmin] = 0

	'''-------------------------------------------
	Now do the spiders
	-------------------------------------------'''
	if spiders:
		angles = [0., 90., 180., 270.]

		img = Image.fromarray(mask)
		draw = ImageDraw.Draw(img)

		for j, angle in enumerate(angles):
			
			angle *= np.pi/180.

			start = 1.1*rmax*np.array([np.cos(angle),np.sin(angle)])-xx.min()
			stop = np.array([0,0])-xx.min()

			normvec = np.array([start[1]-stop[1],stop[0]-start[0]])*thick
			corners = np.round(m2pix*np.array([start+normvec,start-normvec,stop-normvec,stop+normvec]))

			#convert to pixels

			draw.polygon([tuple(p) for p in corners], fill=0)

		newmask = np.copy(np.asarray(img))
	else:
		newmask = np.copy(mask)


	'''-------------------------------------------
	Display
	-------------------------------------------'''

	# plt.figure(0)
	# plt.imshow(newmask,cmap=plt.cm.gray,extent=[xx.min(),xx.max(),xx.min(),xx.max()])
	# plt.xlabel('x (m)')
	# plt.ylabel('y (m)')
	# plt.title('WFIRST Pupil')
	# plt.draw()
	# plt.show()

	return newmask,xs,m2pix
