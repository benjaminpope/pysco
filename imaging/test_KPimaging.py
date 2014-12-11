#! /usr/bin/env python

"""
Goal: test Alexey's simulation (kpo files that have all kpi geometry + measured kps encoded
"""

import numpy as np
import pickle
from whisky import mas2rad
import whisky
from mk_kp2implane import kerphimobj
from mem4_kp2pm import MemImage
import matplotlib.pyplot as plt


if __name__ == "__main__":

	kpd_filename = "alexey_sims/golay9_scale50_0.1.kpd.gz"
	kpo = kpo=whisky.kpo(kpd_filename)
	kio = kerphimobj(kpo)
	kio.kerph2im()
	fn = kio.write()

	testgolay9 = MemImage(fn)

	im = testgolay9.mem_image()
	## the rest is a plot
	plt.imshow(im, interpolation='nearest',cmap=plt.get_cmap('gist_heat'),extent=testgolay9.extent)
	plt.plot(0,0,'w*', ms=15)
	plt.axis(testgolay9.extent)
	plt.xlabel('Delta RA (milli-arcsec)', fontsize='large')
	plt.ylabel('Delta Dec (milli-arcsec)', fontsize='large')
	print "Total contrast (mags): " + str(-2.5*np.log10(np.sum(testgolay9.pm)))
	plt.show()	
