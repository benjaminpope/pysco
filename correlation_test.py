################-----------------#################
### Created for phases correlation estimation with Kernel or Bispectral phases ########
###  Author: Alexey Latyshev. Date: 05.2014 ###
################-----------------#################

import numpy as np
import pyscosour as pysco 

ddir = '/import/pendragon1/latyshev/Data/KerPhases/'
kpidir='kpi/'
kpddir='outputs/frozen_noiseless7000/kpd/binaries/bsp/'
D=8.0
wl=2.6e-6
lambdaD=wl/D
sg_ld=10.0 # windowing radius in lambda/D


# creating kpd with bispectral phases	(100000 phases)
kpo=pysco.kpo(ddir+kpidir+'full_hex15.kpi.gz')
kpo.extract_kpd(ddir+'outputs/frozen_noiseless7000/PSF/binaries/full_hex15_frozen_s135_c5_a0_7000.fits',manual=0,D=D,sg_ld=sg_ld,
		plotim=False,re_center=True,window=True,use_main_header=True,ave='median',bsp=True,range=(0,100000))
kpo.save_to_file(ddir+kpddir+'full_hex15_frozen_'+s+'_s135_c5_a0_7000.kpd.gz')

'''	
#loading kpd file
kpo=pysco.kpo(ddir+kpddir+'full_hex15_frozen_s135_c5_a0_7000.kpd.gz'))	
'''	
# plot with bispectral phases
#pysco.correlation_plot_bsp(kpo,params=[135.,0.,5.])		

# plot with kernel phases
#pysco.correlation_plot(kpo,params=[135.,0.,5.])