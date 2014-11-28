import numpy as np
from scipy.optimize import leastsq
from core import *
from fitting import *
from grid import *
from calibration import *
from kpo import *
from kpi import *
import sys
from numpy.random import rand, randn
from random import choice, shuffle
import multiprocessing
import time

'''---------------------------------------------------------------------------
limits.py - pyscosour module for calculating detection limits.

This is based on code by A. Cheetham, F. Martinache and B. Pope. 
---------------------------------------------------------------------------'''
# [AL, New, faster version]
def detec_sim_loopfit(everything):
    #Function for multiprocessing in detec_limits. Takes a 
    #single separation and full angle, contrast lists.
    chi2_diff = np.zeros((everything['nth'],everything['ncon'],everything['nsim']))
    kpo = everything['kpo']
    bsp = everything['bsp']
    # [AL, 2014.04.17] Restructured headers to make everything uniform in the loop
    heads=[]
    if kpo.nsets == 1:	   				
        heads.append({'tel' : kpo.hdr['tel'], 'filter' : kpo.hdr['filter']})
        count=[1]
    else :
        head_info=[]
        for head in kpo.hdr :
            head_info.append({'tel' : head['tel'],'filter' : head['filter']})
        heads = np.unique(np.array(head_info)) # [AL, 2014.04.17] removing calculating load by reducing a number of headers to process
        count = np.bincount(heads.searchsorted(head_info)) # [AL, 2014.04.17] removing calculation load by reducing a number of headers to process
    if bsp :
        rnd_err=kpo.bspe[:,np.newaxis]*everything['rands']	#bsp								
    else :
        rnd_err=kpo.kpe[:,np.newaxis]*everything['rands']	#kp										    						
    for j,th in enumerate(everything['ths']):
        for k,con in enumerate(everything['cons']):
            chi2_sngl = 0
            chi2_binr = 0												            
            num=0														
            for head in heads:
                if bsp :	
                    bin_kp=extract_bsp(cvis_binary(kpo.uv[:,0], kpo.uv[:,1], kpo.wavel,[everything['sep'],th,con]),uvrel=kpo.kpi.uvrel,rng=(0,len(kpo.bsp)),showMessages=False) #bsp
                else :																				
                    bin_kp = binary_model([everything['sep'],th,con],kpo.kpi, head) #kp
																
                #-----------------------
                # binary cp model
                # ----------------------
                rnd_kp = bin_kp[:,np.newaxis] + rnd_err # [AL, 2014.04.17] rnd_err is now generated outside of the loop

                if bsp :																
                    chi2_sngl += np.sum(((rnd_kp/ kpo.bspe[:,np.newaxis])**2),axis=0)*count[num] #bsp	
                else :																				
                    chi2_sngl += np.sum(((rnd_kp/ kpo.kpe[:,np.newaxis])**2),axis=0)*count[num] #kp																				
                #chi2_binr += np.sum((((rnd_kp-bin_kp[:,np.newaxis]) / kpo.kpe[:,np.newaxis])**2),axis=0)

                if bsp :																					
                    chi2_binr += np.sum(((rnd_err / kpo.bspe[:,np.newaxis])**2),axis=0)*count[num] # [AL, 2014.04.17] simplified #bsp	
                else :	
                    chi2_binr += np.sum(((rnd_err / kpo.kpe[:,np.newaxis])**2),axis=0)*count[num] # [AL, 2014.04.17] simplified #kp																				
                num+=1																
            chi2_diff[j,k] = chi2_binr-chi2_sngl # note not i,j,k - i is for seps

    if everything['ix'] % 8 ==0:
        print 'Done',everything['ix']
        if everything['ix'] != 0:
            remaining =  (time.time()-everything['tic'])*(everything['nsep']-everything['ix'])/float(everything['ix'])
            if remaining > 60:
                print('Estimated time remaining: %.2f mins' % (remaining/60.))
            else: 
                print('Estimated time remaining: %.2f seconds' % (remaining))

    return chi2_diff

# =========================================================================
# =========================================================================

''' [AL, 2014.03.10]'''
''' added draw, D and name parameter in order to prevent drawing or/and display separation in lambda/D units'''
# [AL, 2014.11.28] Added closure (bsp)/kernel phases flag for limits calculations
def detec_limits(kpo,nsim=10000,nsep=32,nth=20,ncon=32,smin='Default',smax='Default',
    cmin=1.0001,cmax=500.,addederror=0,threads=1,save=False, draw=True, D=0.0,name='',bsp=False):

    '''uses a Monte Carlo simulation to establish contrast-separation 
    detection limits given an array of standard deviations per closure phase.

    Because different separation-contrast grid points are entirely
    separate, this task is embarrassingly parallel. If you want to 
    speed up the calculation, use multiprocessing with a threads 
    argument equal to the number of available cores.

    Make nseps a multiple of threads! This uses the cores most efficiently.

    Hyperthreading (2x processes per core) in my experience gets a ~20%
    improvement in speed.

    Contrast detection limits for multiple kernel phase sets can have very
    (unphysically) high limits, so be warned.

    Written by F. Martinache and B. Pope.'''

    #------------------------
    # first, load your data!
    #------------------------

    u,v = kpo.uv[:,0],kpo.uv[:,1]

    wavel = kpo.wavel

    if bsp :
        ndata = len(kpo.bspe)
    else :								
        ndata = kpo.kpi.nkphi

    w = np.array(np.sqrt(u**2 + v**2))/np.max(wavel)

    if smin == 'Default':
        smin = rad2mas(1./4/np.max(w))

    if smax == 'Default':
        #smax = rad2mas(1./np.min(w))
        smax = rad2mas(1./2/np.min(w)) # [AL, 2013.03.18] Changed maximum separation to 0.5lambda/shortest baseline

    #------------------------
    # initialise Monte Carlo
    #------------------------

    seps = smin + (smax-smin) * np.linspace(0,1,nsep)
    ths  = 0.0 + 360.0  * np.linspace(0,1,nth)
    cons = cmin  + (cmax-cmin)  * np.linspace(0,1,ncon)

    rands = np.random.randn(ndata,nsim)

    #------------------------
    # Run Monte Carlo
    #------------------------

    tic = time.time() # start the clock

    if threads ==0:
        chi2_diff = np.zeros((nsep,nth,ncon,nsim))
        for i,sep in enumerate(seps):
            print("iteration # %3d: sep=%.2f" % (i, sep))            										
            chi2_diff[i,:,:,:]= detec_sim_loopfit(sep)
            toc = time.time()
            if i != 0:
                remaining =  (toc-tic)*(nsep-i)/float(i)
                if remaining > 60:
                    print('Estimated time remaining: %.2f mins' % (remaining/60.))
                else: 
                    print('Estimated time remaining: %.2f seconds' % (remaining))
    else:
        all_vars=[]
        for ix in range(nsep):
            everything={'sep':seps[ix],'cons':cons,'ths':ths, 'ix':ix,
                'nsep':nsep,'ncon':ncon,'nth':nth,'nsim':nsim,
                'rands':rands,'kpo':kpo, 'tic':tic, 'bsp':bsp}
            all_vars.append(everything)
        pool = multiprocessing.Pool(processes=threads)
        chi2_diff=pool.map(detec_sim_loopfit,all_vars)
        chi2_diff = np.array(chi2_diff)
    tf = time.time()
    if tf-tic > 60:
        print 'Total time elapsed:',(tf-tic)/60.,'mins'
    elif tf-tic <= 60:
        print 'Total time elapsed:',tf-tic,'seconds'

    ndetec = np.zeros((ncon, nsep))

    nc, ns = int(ncon), int(nsep)

    for k in range(nc):
        for i in range(ns):
            toto = (chi2_diff)[i,:,k,:]
            ndetec[k,i] = (toto < 0.0).sum()

    nbtot = nsim * nth

    ndetec /= float(nbtot)

    print 'ndetec',ndetec
		
    levels = [0.5,0.9, 0.99, 0.999]		
    data = {'levels': levels,
            'ndetec': ndetec,
            'seps'  : seps,
            'angles': ths,
            'cons'  : cons,
            'name'  : kpo.name}
	
    if save == True:
        if name=='' : 	
            file = 'limit_lowc'+kpo.name+'.pick'
        else :
	      file='limit_lowc_'+name+'.pick' # [AL, 2014.03.10 - output filename added]
        print file

        myf = open(file,'w')
        pickle.dump(data,myf)
        myf.close()

    # ---------------------------------------------------------------
    #                        contour plot!
    # ---------------------------------------------------------------
    lambdaD=0.0
    if D>0 :
        lambdaD=wavel/D   
    ''' [AL, 2014.03.10] Drawing is now separate function''' 	
    if draw :
        draw_limits(data, levels=levels, lambdaD=lambdaD)
    				
    return data

# =========================================================================
# =========================================================================

''' [AL, 2014.03.10]'''
''' added draw_limits function'''
''' if lambdaD is not zero then x axis is in lambda/D units'''
''' [AL, 2014.03.26]'''
''' added maximum separation parameter'''
def draw_limits(data, levels=[0.5,0.9, 0.99, 0.999], lambdaD=0.0, maxSep=0.0):
    mycols = ('k', 'k', 'k', 'k')
    seps=data['seps']
    if lambdaD>0 :
        seps=mas2rad(seps)/lambdaD
    if maxSep>0.0 :
        idx=0
        while idx<len(seps):
            if seps[idx]<=maxSep :
                idx+=1
            else :
                break
        seps=seps[0:idx]
        ndetec=data['ndetec'][:,0:idx]
    else :
        ndetec=data['ndetec']						
    plt.figure(0)			
    contours = plt.contour(ndetec, levels, colors=mycols, linewidth=2, 
                 extent=[seps[0], seps[len(seps)-1], data['cons'][0], data['cons'][len(data['cons'])-1]])
    plt.clabel(contours)
    plt.contourf(seps,data['cons'],ndetec,data['levels'],cmap=plt.cm.bone)
    plt.colorbar()
    if lambdaD>0 :
        plt.xlabel('Separation (lambda/D)') 
    else :	
        plt.xlabel('Separation (mas)')
    plt.ylabel('Contrast Ratio')
    plt.title('Contrast Detection Limits')
    plt.draw()
    plt.show()

# =========================================================================
# =========================================================================

def binary_fit(kpo, p0):
    '''Performs a best binary fit search for the dataset.
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    ------------------------------------------------------------- '''
    
    if np.all(kpo.kpe == 0.0):
        print("Closure phase object instance is not calibrated.\n")
        soluce = leastsq(kpo.bin_fit_residuals, p0, args=(kpo),
                     full_output=1)
    else:
        def lmcpmodel(index,params1,params2,params3):
            params = [params1,params2,params3]
            model = cp_model(params,kpo.u,kpo.v,kpo.wavel)
            return model[index]
        soluce = curve_fit(lmcpmodel,range(0,kpo.ndata),kpo.kpd,p0,sigma=kpo.kpe)
    kpo.covar = soluce[1]
    soluce[0][1] = np.mod(soluce[0][1],360.) # to get consistent position angle measurements
    return soluce

# =========================================================================
# =========================================================================

def bin_fit_residuals(params, kpo):
    '''Function for binary_fit without errorbars'''
    test = binary_model(params,kpo.kpi,kpo.hdr)
    err = (kpo.kpd - test)
    return err
    
# =========================================================================
# =========================================================================
    
def brute_force_chi2_grid(everything):
    '''Function for multiprocessing, does 3d chi2 fit, followed by
       Levenberg Marquadt, then returns best sep, PA, contrast ratio'''
    this_kpo=everything['sim_kpo']
    data_cp=this_kpo.kpd
    chi2=np.zeros((everything['nsep'],everything['nth'],everything['ncon']))
    for i,sep in enumerate(everything['seps']):
        for j,th in enumerate(everything['ths']):
            for k,con in enumerate(everything['cons']):
                mod_cps = cp_model([sep,th,con],everything['u'],everything['v'],everything['wavel'])
                chi2[i,j,k]=np.sum(((data_cp-mod_cps)/everything['error'])**2)
    b_params_ix=np.where(chi2==np.amin(chi2))
    b_params=[everything['seps'][b_params_ix[0][0]],everything['ths'][b_params_ix[1][0]],everything['cons'][b_params_ix[2][0]]]
    b_params=np.array(b_params)
    
    #now do L-M fitting. Sometimes it can't find the best position,
    #so take the coarse estimate instead
    try:
        [best_params,cov]=binary_fit(this_kpo,b_params)
    except:
        best_params=b_params
        print "couldn't find best params!"
    if everything['ix'] % 50 ==0:
        print 'Done',everything['ix']
    cov,chi2,b_params,this_kpo,data_cp=None,None,None,None,None
    return best_params

# =========================================================================
# =========================================================================

''' [AL, 2014.04.17]'''
''' added draw, D and name parameter in order to prevent drawing or/and display separation in lambda/D units'''
def brute_force_detec_limits(kpo,nsim=100,nsep=32,nth=20,ncon=32,smin='Default',smax='Default',
    cmin=10.,cmax=500.,addederror=0,threads=0,save=False, draw=True, D=0.0,name=''):

    '''uses a Monte Carlo simulation to establish contrast-separation 
    detection limits given an array of standard deviations per closure phase.

    Because different separation-contrast grid points are entirely
    separate, this task is embarrassingly parallel. If you want to 
    speed up the calculation, use multiprocessing with a threads 
    argument equal to the number of available cores.

    Make nseps a multiple of threads! This uses the cores most efficiently.

    Hyperthreading (2x processes per core) in my experience gets a ~20%
    improvement in speed.

    Written by F. Martinache and B. Pope.
    
    This version was modified by ACC to use a brute force
    chi2 grid instead of making any assumptions or approximations (and is
    obviously slower).'''

    #------------------------
    # first, load your data!
    #------------------------

    error = kpo.kpe + addederror
    #u,v = kpo.u,kpo.v
    u,v = kpo.uv[:,0],kpo.uv[:,1] # [AL, 2013.04.22] Fixed

    wavel = kpo.wavel

    # ndata = kpo.ndata
    ndata = kpo.kpi.nkphi # [AL, 2013.04.22] Fixed

    w = np.array(np.sqrt(u**2 + v**2))/wavel

    if smin == 'Default':
        smin = rad2mas(1./4/np.max(w))

    if smax == 'Default':
        #smax = rad2mas(1./np.min(w))
        smax = rad2mas(1./2/np.min(w)) # [AL, 2013.04.22] Fixed
    #------------------------
    # initialise Monte Carlo
    #------------------------

    seps = smin + (smax-smin) * np.linspace(0,1,nsep)
    ths  = 0.0 + 360.0  * np.linspace(0,1,nth)
    cons = cmin  + (cmax-cmin)  * np.linspace(0,1,ncon)

    rands = np.random.randn(ndata,nsim)
    sim_cps=rands*error[:,np.newaxis]
    sim_kpos=[]
    for ix in range(nsim):
        this_kpo=copy.deepcopy(kpo)
        this_kpo.kpd=sim_cps[:,ix]
        sim_kpos.append(this_kpo)
    
    #------------------------
    # Run Monte Carlo
    #------------------------

    tic = time.time() # start the clock
    best_params=[]
    if threads ==0:
        toc=time.time()
        best_params=np.zeros((nsim,3))
        for ix in range(nsim):
            best_params[ix,:]=brute_force_chi2_grid(ix)
            if (ix % 50) ==0:
                tc=time.time()
                print 'Done',ix,'. Time taken:',(tc-toc),'seconds'
                toc=tc  
    else:
        all_vars=[]
        for ix in range(nsim):
            everything={'seps':seps,'cons':cons,'ths':ths, 'ix':ix,
                'nsep':nsep,'ncon':ncon,'nth':nth,'u':u,'v':v,
                'sim_kpo':sim_kpos[ix],'error':error,'wavel':wavel}
            all_vars.append(everything)
        pool = multiprocessing.Pool(processes=threads)
        best_params=pool.map(brute_force_chi2_grid,all_vars)
    tf = time.time()
    if tf-tic > 60:
        print 'Total time elapsed:',(tf-tic)/60.,'mins'
    elif tf-tic <= 60:
        print 'Total time elapsed:',tf-tic,'seconds'

    ndetec = np.zeros((ncon, nsep))
    nc, ns = int(ncon), int(nsep)
    
    #collect them   
    for ix in range(nsim):
        sep=best_params[ix][0]
        con=best_params[ix][2]
        sep_ix=np.where(abs(seps-sep)==np.amin(abs(seps-sep)))
        con_ix=np.where(abs(cons-con)==np.amin(abs(cons-con)))
        ndetec[sep_ix,con_ix]+=1
    #Take the cumulative sum over contrast ratio at each sep
    cumsum_detec=ndetec.cumsum(axis=1)
    #turn into %
    maxdetec=np.amax(cumsum_detec,axis=1)
    ndetec=0*cumsum_detec
    for ix in range(nsep):
        if maxdetec[ix]==0:
            print 'No sims for sep '+str(seps[ix])+'mas'
        else:
            print str(maxdetec[ix])+" in "+str(seps[ix])+" bin."
            ndetec[ix,:]=cumsum_detec[ix,:]/maxdetec[ix]

    ndetec=1-ndetec
    #Axes were wrong way around (I blame IDL)
    ndetec=np.transpose(ndetec)

    # [AL, 2014.04.17 - saving added]
    if save == True:
        if name=='' : 	
            file = 'limit_lowc'+kpo.name+'.pick'
        else :
	      file='limit_lowc_'+name+'.pick' 
        print file

        myf = open(file,'w')
        pickle.dump(data,myf)
        myf.close()				
    # ---------------------------------------------------------------
    #                        contour plot!
    # ---------------------------------------------------------------
    lambdaD=0.0
    if D>0 :
        lambdaD=wavel/D   
    ''' [AL, 2014.04.17] Drawing is now separate function''' 	
    if draw :
        draw_limits(data, levels=levels, lambdaD=lambdaD)				
    
        
    return data
	

# [AL, 2014.03.26]			
# A function for calculating basic statistics of contrast for a given limits data
# NB: if all the values are over maximum contrast, it will be used as a value for all the data points
#     all the separation data points with no significant contrast are excluded from analysis
# data - input data (limits from detect_limits function)
# level - cutoff probability
# lambdaD  - [optional] lambda/D value in order to treat separations in lambda/D units. Used only if we define max and min
# maxSep - [optional] cutoff separation level either in mas or in lambda/D (if lambda/D>0)
# minSep - [optional] cutoff separation level either in mas or in lambda/D (if lambda/D>0)
def calc_contrast(data,level=0.999,lambdaD=0.0,maxSep=0.0,minSep=0.0) :
    '''	
    A function for calculating basic statistics of contrast for a given limits data
    NB: if all the values are over maximum contrast, it will be used as a value for all the data points
         all the separation data points with no significant contrast are excluded from analysis	
    '''
    seps=data['seps']
    if lambdaD>0 :
        seps=mas2rad(seps)/lambdaD
    if maxSep>0.0 or minSep>0.0:
        idx_st=0
        idx_end=0
        while idx_end<len(seps):
            if seps[idx_st]<minSep :
                idx_st+=1
            if seps[idx_end]<=maxSep :
                idx_end+=1
            else :
                break
        if idx_end>idx_st :
            seps=seps[idx_st:idx_end]
            ndetec=data['ndetec'][:,idx_st:idx_end]
        else :
            ndetec=data['ndetec']									
    else :
        ndetec=data['ndetec']	
    cons=-np.ones(np.shape(seps),dtype='float')	
    for i in range(np.shape(ndetec)[0]-1,-1,-1) :
        for j in range(0,np.shape(ndetec)[1]) :
            if ndetec[i,j]>=level and cons[j]<0 :
                cons[j]=data['cons'][i]
    cons_res=[]		
    num=0
    for i in range(0,len(cons)) :
        if cons[i]>=0 :
            cons_res.append(cons[i])
        else :
            num+=1
    if num>0 :
        print("Warning: %d data point(s) were excluded from analysis" % (num))
    if len(cons_res)>0 :
        res={'mean' : np.mean(cons_res), 
             'median' : np.median(cons_res),
             'std' : np.std(cons_res), 
             'max' : np.max(cons_res), 
             'min' : np.min(cons_res)
            }
    else :
        res={'mean' : 0, 
             'median' : 0,
             'std' : 0, 
             'max' : 0, 
             'min' : 0
            }					
    print("- Mean=%f" % (res['mean']))
    print("- Median=%f" % (res['median']))
    print("- Std=%f" % (res['std']))
    print("- Max=%f" % (res['max']))
    print("- Min=%f" % (res['min']))
    return res				
