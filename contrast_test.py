################-----------------#################
### Created for contrast limits estimation ########
###  Author: Alexey Latyshev. Date: 02.2014 ###
################-----------------#################
import numpy as np

import whisky as pysco # for compatibility and ease of use

# Parameters for contrasts plot and for detection limits
D=8.0
wl=2.6e-6
lambdaD=wl/D
sg_ld=10.0 # windowing radius in lambda/D
#sg_ld=15.7595/3. # from IDL

ddir = '/import/pendragon1/latyshev/Data/KerPhases/'

txtdir='sampling_points/'
kpidir='kpi/'
outdir='outputs/frozen_noiseless500_cut/'
#outdir='outputs/frozen_noiseless500/'
kpddir=outdir+'kpd/'
fitsdir=outdir+'PSF/'
limitsdir=outdir+'contrasts/limits/'
fits_ext='.fits'
kpd_ext='.kpd.gz'
kpi_ext='.kpi.gz'
txt_ext='.txt'

# (kpifile,kpifile_txt,kpdfile,fitsfile)
data=[]
data.append(('full_hex15','full_hex15','full_hex15_scale50','full_hex15_scale50'))
data.append(('ann_hex15','ann_hex15','ann_hex15_scale50','ann_hex15_scale50'))
data.append(('ann_hex15_w05','ann_hex15_w05','ann_hex15_w05_scale50','ann_hex15_w05_scale50'))
data.append(('golay9','golay9','golay9_scale50','golay9_scale50'))
# lines in data array to analyse
active = range(0,len(data))	

'''
# creating kpi from txt file
#calculating kpis
for i in active :
	kpi=pysco.kpi(ddir+txtdir+data[i][1]+txt_ext)
	kpi.save_to_file(ddir+kpidir+data[i][0]+kpi_ext)
'''

#coeff=1.0 # 25*t0 # max coefficient reduction for different integration times
#coeff=0.7 # 3*t0
coeff=0.6 #frozen
#coeff=1.0

nsim=1000
nsep=32
nth=12

#nsim=40
#nsep=16
# calculating limits
#calculating kpds
num=0
#for s in [0,0.05,0.1,0.2,0.4,0.6] :
#for s in [0.2,0.4,0.6] :
#for s in [0,0.05,0.1] :
for s in [0.8,0.9] :	
	if s==0 : suff='no_ao'
	else : suff=str(s)
	print(suff)			
	for i in active :	
		'''
		#loading kpo from kpi/load pre-calculated ker phases
		kpo=pysco.kpo(ddir+kpidir+data[i][0]+kpi_ext)
		# extracting kpd
		kpo.extract_kpd(ddir+fitsdir+data[i][3]+'_'+suff+fits_ext,manual=0,D=D,sg_ld=sg_ld,plotim=False,re_center=True,window=True,use_main_header=True,unwrap_kp=False)	
		# saving kpd
		#kpo.save_to_file(ddir+kpddir+data[i][2]+'_'+suff+kpd_ext)
		'''			
		kpo=pysco.kpo(ddir+kpddir+data[i][2]+'_'+suff+kpd_ext)		
		cmin=1.01
		cmax=1000
		ncon=80
		if s<0.05 : 
			cmax=50
			ncon=25	
		elif s==0.05 : 
			cmax=150	
			ncon=40				
		elif s==0.1 : 
			cmax=200
			ncon=50
		elif s==0.2:
			cmax=300
			ncon=60
		elif s==0.4:
			cmax=400
			ncon=80
		elif s==0.6 :
			cmax=2500
			ncon=200
		elif s==0.8 :
			cmax=2000
			ncon=200
		elif s==0.9 :
			cmax=4000
			ncon=250	
		# golay9
		if data[i][0]=='golay9' and s<0.4:
			cmax/=5
			#ncon=25
		if data[i][0]=='golay9' and s >= 0.8:
			cmax=25000
			ncon=500
		cmax*=coeff
		limits=pysco.detec_limits(kpo,nsim=nsim,nsep=nsep,nth=nth,ncon=ncon,smin='Default',smax='Default',
			cmin=cmin,cmax=cmax,addederror=0,threads=4,save=True,draw=False,name=data[i][2][ :-7]+suff)
		num+=1

# drawing
'''
res=[]
restxt=[]
limits=[]
num=0
#for s in [0,0.05,0.1,0.2,0.4,0.6] :
for s in [0.2,0.4,0.6] :
	if s==0 : suff='no_ao'
	else : suff=str(s)
	for i in active :	
		print(data[i][2][:-7]+suff)
		# loading limits
		limits.append(np.load(ddir+limitsdir+'limit_lowc_'+data[i][2][:-7]+suff+'.pick'))
		res.append(pysco.calc_contrast(limits[num],level=0.999,lambdaD=lambdaD,maxSep=sg_ld,minSep=1.0))
		restxt.append(str(np.round(res[num]['mean'],1))+' ('+str(np.round(res[num]['std'],2))+')')
		pysco.draw_limits(limits[num], levels=[0.5,0.9, 0.99, 0.999], lambdaD=lambdaD, maxSep=sg_ld)	
		num+=1
np.savetxt('frozen.txt',restxt,fmt="%s")
'''
