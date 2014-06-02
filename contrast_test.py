################-----------------#################
### Created for detecting contrast limits ########
###  Author: Alexey Latyshev. Date: 02.2014 ###
################-----------------#################
import numpy as np
import pyscosour as pysco # for compatibility and ease of use

# Parameters for contrasts plot and for detection limits
D=8.0
wl=2.6e-6
lambdaD=wl/D
sg_ld=10.0 # windowing radius in lambda/D

ddir = '/import/pendragon1/latyshev/Data/KerPhases/'	# root directory

txtdir='sampling_points/'	# a directory with sampling points in text format
kpidir='kpi/'				# a directory to store kpi
outdir='outputs/25t0_noiseless500/' # a directory with outputs 
kpddir=outdir+'kpd/'				# a directory to store kpd
fitsdir=outdir+'PSF/'				# a directory to store fits files
limitsdir=outdir+'contrasts/limits/'# a directory to put limits
fits_ext='.fits'
kpd_ext='.kpd.gz'
kpi_ext='.kpi.gz'
txt_ext='.txt'

# setting up working data
# (kpifile,kpifile_txt,kpdfile,fitsfile)
data=[]
data.append(('full_hex15','full_hex15','full_hex15_scale50','full_hex15_scale50'))
data.append(('ann_hex15','ann_hex15','ann_hex15_scale50','ann_hex15_scale50'))
data.append(('ann_hex15_w05','ann_hex15_w05','ann_hex15_w05_scale50','ann_hex15_w05_scale50'))
data.append(('golay9','golay9','golay9_scale50','golay9_scale50'))
# lines in data array to analyse
active = range(0,len(data))	



'''
# creating kpi from txt files if required
#calculating kpis
for i in active :
	kpi=pysco.kpi(ddir+txtdir+data[i][1]+txt_ext)
	kpi.save_to_file(ddir+kpidir+data[i][0]+kpi_ext)
'''

'''
# kreating kpd files from kpi if required
for i in active :
	#loading kpo from kpi/load pre-calculated ker phases
	kpo=pysco.kpo(ddir+kpidir+data[i][0]+kpi_ext)
	# extracting kpd	kpo.extract_kpd(ddir+fitsdir+data[i][3]+'_'+suff+fits_ext,manual=0,D=D,sg_ld=sg_ld,plotim=False,re_center=True,window=True,use_main_header=True)
	# saving kpd
	kpo.save_to_file(ddir+kpddir+data[i][2]+'_'+suff+kpd_ext)
'''		


#loading kpos from saved kpd
kpos=[]
num=0
for i in active :	
	kpos.append(pysco.kpo(ddir+kpddir+data[i][2]+kpd_ext))
	

#parameters for limits detection
nsim=1000
nsep=32
nsim=40
cmin=1.01
cmax=1000
ncon=80

# calculating limits
limits=[]
num=0
for i in active :	
	limits.append(pysco.detec_limits(kpos[num],nsim=nsim,nsep=nsep,nth=nth,ncon=ncon,smin='Default',smax='Default',
			cmin=cmin,cmax=cmax,addederror=0,threads=4,save=True,draw=False,name=data[i][2][:-7]))
	num+=1

# loading stored limits and drawing plots
'''
res=[]
restxt=[]
limits=[]
num=0
for i in active :	
	print(data[i][2][:-7])
	# loading limits
	limits.append(np.load(ddir+limitsdir+'limit_lowc_'+data[i][2][:-7]+suff+'.pick'))
	res.append(pysco.calc_contrast(limits[num],level=0.999,lambdaD=lambdaD,maxSep=sg_ld,minSep=1.0))
	restxt.append(str(np.round(res[num]['mean'],1))+' ('+str(np.round(res[num]['std'],2))+')')
	pysco.draw_limits(limits[num], levels=[0.5,0.9, 0.99, 0.999], lambdaD=lambdaD, maxSep=sg_ld)	
	num+=1
'''
