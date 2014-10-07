import numpy as np
from scipy.optimize import leastsq
from core import *
from kpo import *
from kpi import *
import sys
from numpy.random import rand, randn
from random import choice, shuffle
import emcee
# import pymultinest ###### Removed temporary ! #######
import time

# =========================================================================
# =========================================================================
#[Al, 2014.05.12] Frantz's version	
def binary_model(params, kpi, hdr, vis2=False):
    ''' Creates a binary Kernel-phase model.
    
    ------------------------------------------------------------------ 
    uses a simple 5 parameter binary star model for the uv phases that
    should be observed with the provided geometry.
    
    Additional parameters are:
    - kpi, a kernel phase info structure
    - hdr, a header data information
    ------------------------------------------------------------------ '''
    
    params2 = np.copy(params)
    if 'Hale' in hdr['tel']: params2[1] += 220.0 + hdr['orient']
    if 'HST'  in hdr['tel']: params2[1] -= hdr['orient']
    else:                    params2[1] += 0.0

    wavel = hdr['filter']
    
    testPhi = phase_binary(kpi.uv[:,0], kpi.uv[:,1], wavel, params2)
    res = np.dot(kpi.KerPhi, testPhi)

    if vis2:
        res = vis2_binary(kpi.uv[:,0], kpi.uv[:,1], wavel, params2)
    return res
    
# =========================================================================
# =========================================================================
def binary_KPD_model(kpo, params):
    ''' Returns a 1D or 2D array of binary models.
    
    parameters are:
    - the kp structure (kpo) to be used as a template
    - the list of parameters of the binary model. '''

    models = np.zeros_like(kpo.kpd)

    nm = np.size(kpo.hdr) # number of kp realizations in kpo

    if nm == 1:
        models = binary_model(params, kpo.kpi, kpo.hdr)
    else:
        for i in xrange(nm):
            models[i] = binary_model(params, kpo.kpi, kpo.hdr[i])

    return models

# =========================================================================
# =========================================================================
def binary_KPD_fit_residuals(params, kpo):
    ''' Function to evaluate fit residuals, to be used in a leastsq
    fitting procedure. '''

    #test = binary_model(params, kpo.kpi, kpo.hdr)
    #test = binary_model(params, kpo.kpi, kpo.hdr[0])
    test = binary_KPD_model(kpo, params)
    err = kpo.kpd - test
    if kpo.kpe != None:
        err /= kpo.kpe
    return err

# =========================================================================
# =========================================================================
def binary_KPD_fit(kpo, p0):
    '''Performs a best binary fit search for the datasets.
    
    -------------------------------------------------------------
    p0 is the initial guess for the parameters 3 parameter vector
    typical example would be : [100.0, 0.0, 5.0].
    
    returns the full solution of the least square fit:
    - soluce[0] : best-fit parameters
    - soluce[1] : covariance matrix
    ------------------------------------------------------------- '''
    
    soluce = leastsq(binary_KPD_fit_residuals, 
                     p0, args=((kpo,)),
                     full_output=1)
    
    covar = soluce[1]
    return soluce

# =========================================================================
# =========================================================================
def correlation_plot(kpo, params=[250., 0., 5.],b=None, plot_error=True):
    '''Correlation plot between KP object and a KP binary model
    
    Parameters are:
    --------------
    - kpo: one instance of kernel-phase object
    - params: a 3-component array describing the binary (sep, PA and contrast)

    Option:
    - plot_error: boolean, errorbar or regular plot
    --------------------------------------------------------------------------
    '''
    params2 = np.copy(params)

    if 'Hale' in kpo.hdr['tel']: params2[1] -= 220.0 + kpo.hdr['orient']
    if 'HST'  in kpo.hdr['tel']: params2[1] += kpo.hdr['orient']
    else:                    params2[1] += 0.0
    if b == None:

        mm = np.round(np.max(np.abs(kpo.kpd)), -1)
        
        f1 = plt.figure()
        sp0 = f1.add_subplot(111)
        if plot_error:
            sp0.errorbar(binary_KPD_model(kpo, params2),  kpo.kpd,
                         yerr=kpo.kpe, linestyle='None')
        else:
            sp0.plot(binary_KPD_model(kpo, params2), kpo.kpd, 'bo')
        sp0.plot([-mm,mm],[-mm,mm], 'g')
        sp0.axis([-mm,mm,-mm,mm])

        rms = np.std(binary_KPD_fit_residuals(params2, kpo))
        msg  = "Model:\n sep = %6.2f mas" % (params[0],)
        msg += "\n   PA = %6.2f deg" % (params[1],)
        msg += "\n  con = %6.2f" % (params[2],)        
        msg += "\n(rms = %.2f deg)" % (rms,)
                
        plt.text(0.0*mm, -0.75*mm, msg, 
                 bbox=dict(facecolor='white'), fontsize=14)
                
        msg = "Target: %s\nTelescope: %s\nWavelength = %.2f um" % (
            kpo.kpi.name, kpo.hdr['tel'], kpo.hdr['filter']*1e6)
                
        plt.text(-0.75*mm, 0.5*mm, msg,
                  bbox=dict(facecolor='white'), fontsize=14)
        
        plt.ylabel('Data kernel-phase signal (deg)')
        plt.xlabel('Kernel-phase binary model (deg)')
        plt.draw()
        plt.show()
        return None

    else:
        plt.clf()
        mm = np.round(np.max(np.abs(kpo.kpd)), -1)
        plt.errorbar(b,kpo.kpd,yerr=kpo.kpe,fmt='b.')
        plt.plot([-mm,mm],[-mm,mm], 'g')
        plt.axis('tight',fontsize='large')
        plt.xlabel('Model Kernel Phases',fontsize='large')
        plt.ylabel('Kernel Phase Signal', fontsize='large')
        plt.title('Kernel Phase Correlation Diagram',
                  fontsize='large')
        plt.draw()
        plt.show()

# =========================================================================
# =========================================================================

def kp_chi2(params,kpd,kpe,kpi,hdr):
    '''Calculate chi2 for single band kernel phase data.
    Used both in the MultiNest and MCMC Hammer implementations.'''
    kps = binary_model(params,kpi,hdr)
    chi2 = np.sum(((kpd-kps)/kpe)**2)
    return chi2

# =========================================================================
# =========================================================================

def kp_loglikelihood(params,kpo):
    '''Calculate loglikelihood for kernel phase data.
    Used both in the MultiNest and MCMC Hammer implementations.'''
    if kpo.nsets == 1:
        params = [params[0],params[1],params[2]]
        chi2 = kp_chi2(params,kpo.kpd,kpo.kpe,kpo.kpi,kpo.hdr)
        loglike = -chi2/2.
    else:
        loglike = 0
        for j,band in enumerate(kpo.hdr):
            chi2 = kp_chi2([params[0],params[1],params[j+2]],kpo.kpd[j],kpo.kpe,kpo.kpi,band)
            loglike += -chi2/2.
    return loglike

# =========================================================================
# =========================================================================

def hammer(kpo,ivar=[131., 82., 27.],ndim=3,nwalkers=100,plot=False,burnin=100,nsteps=1000):

    '''Default implementation of emcee, the MCMC Hammer, for kernel phase
    fitting. Requires a kernel phase object kpo, and is best called with 
    ivar chosen to be near the peak - it can fail to converge otherwise.'''

    # make sure you're using the right number of parameters
    nbands = kpo.kpd.shape[0]
    if np.size(kpo.hdr) == 1:
        bands = str(round(1e6*kpo.hdr['filter'],3)) 
    else:
        bands = [str(round(1e6*hd['filter'],3)) for hd in kpo.hdr]

    ivar = np.array(ivar)  # initial parameters for model-fit

    p0 = np.array([ivar + 0.1*ivar*np.random.rand(ndim) for i in range(nwalkers)]) # initialise walkers in a ball

    print 'Running emcee now!'

    t0 = time.time()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, kp_loglikelihood, args=[kpo])

    # burn in
    pos,prob,state = sampler.run_mcmc(p0, burnin)
    sampler.reset()

    t1 = time.time()

    print 'Burnt in! Took %.3f seconds' %(t1-t0)

    # restart
    sampler.run_mcmc(pos,nsteps)

    tf = time.time()

    print 'Time elapsed = %.3f s' %(tf-t0)

    seps = sampler.flatchain[:,0]
    ths = sampler.flatchain[:,1]

    meansep = np.mean(seps)
    dsep = np.std(seps)

    meanth = np.mean(ths)
    dth = np.std(ths)

    print 'Separation %.3f pm %.3f mas' % (meansep,dsep)
    print 'Position angle %.3f pm %.3f deg' % (meanth,dth)

    if kpo.nsets ==1:
        cs = sampler.flatchain[:,2]
        bestcon = np.mean(cs)
        conerr = np.std(cs)
        print 'Contrast at',bands,'um %.3f pm %.3f' % (bestcon,conerr)

    else:
        for j, band in enumerate(bands):
            cs = sampler.flatchain[:,j+2]
            bestcon = np.mean(cs)
            conerr = np.std(cs)
            print 'Contrast at',band,'um %.3f pm %.3f' % (bestcon,conerr)

    if plot==True:

        plt.clf()

        paramnames = ['Separation','Position Angle'] + ['Contrast at '+band+' um' for band in bands]
        paramdims = ['(mas)', '(deg)'] + ['Ratio' for band in bands] 

        for i in range(ndim):
            plt.figure(i)
            plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
            plt.title(paramnames[i])
            plt.ylabel('Counts')
            plt.xlabel(paramnames[i]+paramdims[i])

        plt.show()


    return sampler.flatchain

# =========================================================================
# =========================================================================

def nest(kpo,paramlimits=[20.,250.,0.,360.,1.0001,10],ndim=3,resume=False,eff=0.3,multi=True,
    max_iter=0):

    '''Default implementation of a MultiNest fitting routine for kernel 
    phase data. Requires a kernel phase kpo object, parameter limits and 
    sensible keyword arguments for the multinest parameters. 

    This function does very naughty things creating functions inside this 
    function because PyMultiNest is very picky about how you pass it
    data.

    Optional parameter eff tunes sampling efficiency, and multi toggles multimodal 
    nested sampling on and off. Turning off multimodal sampling results in a speed 
    boost of ~ 20-30%. 

    '''

    # make sure you're using the right number of parameters
    nbands = kpo.kpd.shape[0]
    if np.size(kpo.hdr) == 1:
        bands = str(round(1e6*kpo.hdr['filter'],3))
        parameters = ['Separation','Position Angle','Contrast at ' + bands+' um']
    else:
        bands = [str(round(1e6*hd['filter'],3)) for hd in kpo.hdr]
        parameters = ['Separation','Position Angle'] + ['Contrast at ' + band + ' um' for band in bands]
    
    n_params = len(parameters)
    ndim = n_params

    def myprior(cube, ndim, n_params,paramlimits=paramlimits):
        cube[0] = (paramlimits[1] - paramlimits[0])*cube[0]+paramlimits[0]
        cube[1] = (paramlimits[3] - paramlimits[2])*cube[1]+paramlimits[2]
        for j in range(2,ndim):
            cube[j] = (paramlimits[5] - paramlimits[4])*cube[j]+paramlimits[4]

    def myloglike(cube,ndim,n_params):
        loglike = kp_loglikelihood(cube,kpo)
        return loglike

    tic = time.time() # start timing

    #---------------------------------
    # now run MultiNest!
    #---------------------------------

    pymultinest.run(myloglike, myprior, n_params, wrapped_params=[1], resume=resume, verbose=True, sampling_efficiency=eff, multimodal=multi, n_iter_before_update=1000,max_iter=max_iter)

    # let's analyse the results
    a = pymultinest.Analyzer(n_params = n_params)
    s = a.get_stats()

    toc = time.time()

    if toc-tic < 60.:
        print 'Time elapsed =',toc-tic,'s'
    else: 
        print 'Time elapsed =',(toc-tic)/60.,'mins'

    # json.dump(s, file('%s.json' % a.outputfiles_basename, 'w'), indent=2)
    print
    print "-" * 30, 'ANALYSIS', "-" * 30
    print "Global Evidence:\n\t%.15e +- %.15e" % ( s['global evidence'], s['global evidence error'] )
    params = s['marginals']

    bestsep = params[0]['median']
    seperr = params[0]['sigma']

    if 'Hale' in kpo.hdr['tel']: params[1]['median'] += 220.0 + kpo.hdr['orient']
    elif 'HST'  in kpo.hdr['tel']: params[1]['median'] -= kpo.hdr['orient']
    else:                    params[1]['median'] += 0.0

    params[1]['median'] = np.mod(params[1]['median'],360.)

    bestth = params[1]['median']
    therr = params[1]['sigma']

    print 'Separation: %.3f pm %.2f' % (bestsep,seperr)
    print 'Position angle: %.3f pm %.2f' %(bestth,therr)

    if kpo.nsets == 1:
        bestcon = params[2]['median']
        conerr = params[2]['sigma']
        print 'Contrast at',bands,'um: %.3f pm %.3f' % (bestcon,conerr)
    else:
        for j, band in enumerate(bands):

            bestcon = params[j+2]['median']
            conerr = params[j+2]['sigma']
        
            print 'Contrast at',band,'um: %.3f pm %.3f' % (bestcon,conerr)

    return params
				
# [AL, 2014.05.08] Correlation plot for bispectrum phases
# =========================================================================
# =========================================================================
def correlation_plot_bsp(kpo, params=[250., 0., 5.], plot_error=True,bsp_model=[]):
    '''Correlation plot between KP object and a KP binary model
    
    Parameters are:
    --------------
    - kpo: one instance of kernel-phase object
    - params: a 3-component array describing the binary (sep, PA and contrast)

    Option:
    - plot_error: boolean, errorbar or regular plot
    --------------------------------------------------------------------------
    '''
    params2 = np.copy(params)

    if 'Hale' in kpo.hdr['tel']: params2[1] -= 220.0 + kpo.hdr['orient']
    if 'HST'  in kpo.hdr['tel']: params2[1] += kpo.hdr['orient']
    else:                    params2[1] += 0.0
    if True:        
        if len(bsp_model)==0 :								
            bsp=extract_bsp(cvis_binary(kpo.uv[:,0], kpo.uv[:,1], kpo.wavel, params),uvrel=kpo.kpi.uvrel,rng=(0,len(kpo.bsp)))								
        else :								
            bsp=bsp_model
        mm_data = np.round(np.max(np.abs(kpo.bsp)), -1)
        mm_model = np.round(np.max(np.abs(bsp)), -1)
        mm=max(mm_data,mm_model)
        if mm==0 :
            mm=max(np.max(np.abs(kpo.bsp)),np.max(np.abs(bsp)))
        mm*=1.05 # adding 5% from both sides												
        f1 = plt.figure()
        sp0 = f1.add_subplot(111)
        if plot_error:
            sp0.errorbar(bsp,  kpo.bsp,
                         yerr=kpo.bspe, linestyle='None')
        else:
            sp0.plot(bsp, kpo.bsp, 'bo')
        sp0.plot([-mm,mm],[-mm,mm], 'g')
        sp0.axis([-mm,mm,-mm,mm])
        msg  = "Model:\n sep = %6.2f mas" % (params[0],)
        msg += "\n   PA = %6.2f deg" % (params[1],)
        msg += "\n  con = %6.2f" % (params[2],)                
        plt.text(0.0*mm, -0.75*mm, msg, 
                 bbox=dict(facecolor='white'), fontsize=14)                
        msg = "Target: %s\nTelescope: %s\nWavelength = %.2f um" % (
            kpo.kpi.name, kpo.hdr['tel'], kpo.hdr['filter']*1e6)               
        plt.text(-0.75*mm, 0.5*mm, msg,
                  bbox=dict(facecolor='white'), fontsize=14)        
        plt.ylabel('Data bispectal-phase signal (deg)')
        plt.xlabel('Bispectral-phase binary model (deg)')
        plt.draw()
        plt.show()
        return None

# =========================================================================
# =========================================================================				
