import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits as pf
import copy
import pickle
import os
import sys
import pdb
import glob
import gzip
import pymultinest
import os, threading, subprocess
import matplotlib.pyplot as plt
import json
import oifits
import time

from cp_tools import *

'''------------------------------------------------------------------------
cpo.py - Python class for manipulating oifits format closure phase data.
------------------------------------------------------------------------'''


class cpo():
	''' Class used to manipulate multiple closure phase datasets'''

	def __init__(self, filename,flag='classic'):
		# Default instantiation.

		# if the file is a complete (kpi + kpd) structure
		# additional data can be loaded.
		try:
		   self.extract_from_oifits(filename)
		except:
			try: 
				self.extract_from_dat(filename,flag=flag)
				print 'Extracting from .dat file'
			except:
				print('Invalid file.')


	def extract_from_dat(self,filename,flag='classic'):

		data = np.loadtxt(filename,skiprows=1)
		data = data.transpose()
		self.name = ''

		if flag == 'classic':
			print 'extracting classic'
			try:
				self.u = data[1,:]/np.sqrt(2)
				self.v = data[1,:]/np.sqrt(2)
				self.vis2 = data[2,:]**2.
				self.vis2err = 2*(data[3,:]*data[2,:])
				self.wavels = 1.67e-6*np.ones(len(self.u)) # unless you have a better idea!
			except: print 'failed'

		elif flag=='vega':
			print 'extracting vega'
			try:
				self.vis2 = data[0,:]
				self.vis2err = np.sqrt(data[1,:]**2 + data[2,:]**2)
				self.wavels = 1.e-9*data[4,:]
				self.u = data[3,:]/np.sqrt(2)
				self.v = data[3,:]/np.sqrt(2)
			except: print 'failed'

		else:
			print 'extracting pavo'
			norm_b = data[0,:]
			self.vis2 = data[1,:]
			self.vis2err = data[2,:]
			self.u = data[3,:]
			self.v = data[4,:]
			self.wavels = np.sqrt(self.u**2. + self.v**2.)/norm_b


	def extract_from_oifits(self,filename):
		'''Extract closure phase data from an oifits file.'''

		data = oifits.open(filename)
		self.name = ''

		self.ndata = len(data.t3)

		for j in data.wavelength:
			wavel = data.wavelength[j].eff_wave
			self.wavel = wavel[0]
			break

		self.target = data.target[0].target

		t3data = []
		t3err = []
		self.u = np.zeros((self.ndata,3))
		self.v = np.zeros((self.ndata,3))

		for j, t3 in enumerate(data.t3):
			t3data.append(t3.t3phi[0])
			t3err.append(t3.t3phierr[0])
			self.u[j,:] = [t3.u1coord,t3.u2coord,-(t3.u1coord+t3.u2coord)]
			self.v[j,:] = [t3.v1coord,t3.v2coord,-(t3.v1coord+t3.v2coord)]

		self.t3data = np.array(t3data)
		self.t3err = np.array(t3err)
