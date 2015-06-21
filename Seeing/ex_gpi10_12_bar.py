# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:00:28 2015

@author: latyshev
"""

from common_tasks import shift_image


cont=10
sep=20

#asymmetric
bar=np.zeros(res.shape,res.dtype)
for i in range(bar.shape[0]) :
	print(i)
	bar[i]=res[i]
	for j in range(1,sep+1) :
		bar[i]+=(shift_image(res[i],x=j,y=0,doRoll=False)/(cont*sep))

dt=datetime.datetime.now()
imageToFits(bar,path=ddir+fits_dir,filename='gpi12_10_bar_asymm_c'+str(cont)+'_a0_s5'+fits_ext,
		tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),	
		tint=exp_time,filter=wl)
		
#symmetric
bar=np.zeros(res.shape,res.dtype)
for i in range(bar.shape[0]) :
	print(i)
	bar[i]=res[i]
	for j in range(1,sep+1) :
		bar[i]+=(shift_image(res[i],x=j,y=0,doRoll=False)/(2*cont*sep))
		bar[i]+=(shift_image(res[i],x=-j,y=0,doRoll=False)/(2*cont*sep))

dt=datetime.datetime.now()
imageToFits(bar,path=ddir+fits_dir,filename='gpi12_10_bar_symm_c'+str(cont)+'_a0_s5'+fits_ext,
		tel='simu',pscale=plateScale,odate=dt.strftime("%b %d, %Y"), otime=dt.strftime("%H:%M:%S.%f"),	
		tint=exp_time,filter=wl)		
