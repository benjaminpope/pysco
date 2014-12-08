"""
Should be able to give this a kpo or something like that.
"""
import numpy as np

def rad2mas(rad):
	return rad*(3600*180/np.pi) * (10**3)

class kerphimobj():
	def __init__(self, kpi, kws=[]):
		"""
		kerphimobj takes a kpi object from pysco and key words

		--- relevant key words ---
		fov: size of image in pixels, default=80
		wavl: wavelength in m
		pitch: radians/pixel
		"""
		if 'FOV' in kws:
			self.fov = kws['FOV']
		else:
			self.fov=80
			print "Default FOV assigned: {0} pixels".format(self.fov)
		if 'wavl' in kws:
			self.wavl = kws['pitch']
		else:
			print "Warning: please assign a wavelength for this dataset"
		if 'pitch' in kws:
			self.pitch = kws['pitch']
		else:
			print "Warning: please assign a pixel pitch in radians/pixel"
		if 'kerph' in kws:
			self.kerph = kws['kerph']
		else:
			print "Warning: please assign a set of kernel phase values\
				   before proceeding"
		if 'kerpherr' in kws:
			self.kerpherr = kws['kerpherr']
		else:
			print "Warning: please assign a set of kernel phase errors\
				   before proceeding"
		self.Kmat = kpi.KerPhi
		self.nkphi = kpi.nkphi
		self.ctrs = kpi.mask
		self.uv = kpi.uv
		self.red = kpi.RED
		
	def ffs(self, kx,ky):
		"""
		Returns sine given the kx,ky image coordinates in pixels
		"""
		return np.sin(2*np.pi*self.pitch*((kx - self.off[0])*(self.rcoord[0]) + 
						(ky - self.off[1])*(self.rcoord[1]))/self.wavl)

	def kerph2im(self):
		"""
		adds up the sine transform for uv phases & then multiplies Kmat,
		the transfer matrix from uv phases to kernel phases.
	
		Returns image to kernel phase transfer matrix.
		"""
		# To make the image pixel centered:
		off = np.array([0.5, 0.5])
		# empty sine transform matrix:
		self.ph2im = np.zeros((self.len(uv), self.fov,self.fov))
		for q,uv in enumerate(self.uv):
			self.rcoord = self.uv
			self.ph2im[q,:,:] = self.red[q]*np.fromfunction(self.ffs, (self.fov, self.fov))
		# flatten for matrix multiplication
		self.ph2im = self.ph2im.reshape(len(self.uuvs), fov*fov)
		# Matrix multiply Kmat & sin transform matrix
		self.kerim = np.dot(self.Kmat, self.ph2im)
		# Reshape back to image dimensions
		self.kerim = self.kerim.reshape(len(self.Kmat), fov,fov)
		return self.kerim

	def write(self, fn = 'kerphim.fits'):
		"""
		Writes file out into fits
		optional file name suffix, default = kerphim
		"""
		hdulist = fits.PrimaryHDU(data=self.kerim)
		# mem code needs PXSCALE header keyword in mas/pixel
		hdulist.header.update('PXSCALE', rad2mas(self.pitch), "mas per pixel")
		hdulist.header.update('WAVL', self.wavl, "monochromatic wavelength")
		hdulist.writeto(self.name+"_"+fn, clobber=True)
