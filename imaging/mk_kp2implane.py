"""
Should be able to give this a kpo or something like that eventually.
"""
import numpy as np
from astropy.io import fits
import sys, os

def rad2mas(rad):
	return rad*(3600*180/np.pi) * (10**3)
def mas2rad(mas):
	return mas * (10**(-3)) / (3600*180/np.pi)

class kerphimobj():
	def __init__(self, kpo, **kwargs):
		"""
		kerphimobj takes a kpi object from pysco and key words

		--- relevant key words ---
		fov: size of image in pixels, default=80
		"""
		self.keys = kwargs.keys()
		if 'fov' in self.keys:
			self.fov = kwargs['fov']
		else:
			self.fov=80
			print "Default FOV assigned: {0} pixels".format(self.fov)
		try:
			self.name = kpo.kpi.name
		except:
			self.name = ''
		
		# Geometry (kpi) stuff
		self.Kmat = kpo.kpi.KerPhi
		self.nkphi = kpo.kpi.nkphi
		self.ctrs = kpo.kpi.mask
		self.uv = kpo.kpi.uv
		print self.uv
		sys.exit()
		self.red = kpo.kpi.RED

		# Measured (kpd) stuff
		self.nframes = kpo.kpd.shape[0] # how many frames
		self.kerph = np.mean(kpo.kpd, axis=0)*np.pi/180. # mean kernel phase
		#print kpo.kpd.shape
		#sys.exit()
		self.kerpherr = kpo.kpe*np.sqrt(self.nframes)*np.pi/180. # standard deviation
		self.pitch = kpo.hdr[0]['pscale'] # mas/pixel
		self.wavl = kpo.hdr[0]['filter'] # in m
		
	def ffs(self, kx,ky):
		"""
		Returns sine given the kx,ky image coordinates in pixels
		kx,ky are visibility (transform) coordinates
		"""
		return -np.sin(np.pi*mas2rad(self.pitch)*((kx - self.off[0])*(self.rcoord[0]) + \
						(ky - self.off[1])*(self.rcoord[1]))/self.wavl)

	def ffc(self, kx,ky):
		"""
		Returns cosine given the kx,ky image coordinates in pixels
		"""
		return np.cos(np.pi*mas2rad(self.pitch)*((kx - self.off[0])*(self.rcoord[0]) + 
					    (ky - self.off[1])*(self.rcoord[1]))/self.wavl)

	def kerph2im(self):
		"""
		adds up the sine transform for uv phases & then multiplies Kmat,
		the transfer matrix from uv phases to kernel phases.
	
		Returns image to kernel phase transfer matrix.
		"""
		# To make the image pixel centered:
		self.off = np.array([0.5, 0.5])
		# empty sine transform matrix:
		self.ph2im = np.zeros((len(self.uv), self.fov,self.fov))
		self.sym2im = np.zeros((len(self.uv), self.fov,self.fov))
		for q,one_uv in enumerate(self.uv):
			self.rcoord = one_uv
			self.ph2im[q,:,:] = self.red[q]*np.fromfunction(self.ffs, (self.fov, self.fov))
			self.sym2im[q,:,:] = self.red[q]*np.fromfunction(self.ffc, (self.fov,self.fov))
		# flatten for matrix multiplication
		#self.ph2im = self.ph2im.reshape(len(self.uv), self.fov*self.fov)
		# Matrix multiply Kmat & flattened sin transform matrix
		self.kerim = np.dot(self.Kmat, self.ph2im.reshape(len(self.uv), self.fov*self.fov))
		# Reshape back to image dimensions
		self.kerim = self.kerim.reshape(len(self.Kmat), self.fov,self.fov)
		return self.kerim, self.sym2im

	def write(self, fn = 'kerphim'):
		"""
		Writes file out into fits
		optional file name suffix, default = kerphim
		"""
		# primary extension is the kerphi x image transfer matrix
		imhdu = fits.PrimaryHDU(data=self.kerim)
		# mem code needs PXSCALE header keyword in mas/pixel
		imhdu.header.update('PXSCALE', self.pitch, "mas per pixel")
		imhdu.header.update('WAVL', self.wavl, "monochromatic wavelength")

		# extra extension stores kernel phase measurements
		datahdu = fits.ImageHDU(data=[self.kerph, self.kerpherr])

		# put it all together and write it out
		hdulist = fits.HDUList(hdus = [imhdu,datahdu])
		hdulist.writeto(fn+"_"+self.name+".fits", clobber=True)
		return fn+'_'+self.name+'.fits'
