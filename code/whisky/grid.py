import numpy as np
from scipy.optimize import leastsq
from core import *
from fitting import *
from calibration import *
from kpo import *
from kpi import *
import sys
from numpy.random import rand, randn
from random import choice, shuffle

'''----------------------------------------------------------------------
Grid-fitting tools by Martinache & Pope for chi2 maps. 

Use these if nested sampling or MCMC are absolutely out of the question - 
they are very slow!
----------------------------------------------------------------------'''

# =========================================================================
# =========================================================================
def chi2map_sep_con(kpo, pa=0, reduced=False, cmap=None,
                    srng=[10.0, 200.0, 100], crng=[10.0, 50.0, 100]):

    ''' Draws a 2D chi2 map of the binary parameter space for a given PA
    ---------------------------------------
    PA: position angle in degrees.
    other options:
    - reduced: (boolean) reduced chi2 or not
    - cmap: (matplotlib color map) "jet" or "prism" are good here
    - srng: range of separations [min, max, # steps]
    - crng: range of contrasts   [min, max, # steps]
    --------------------------------------- '''
    nullvar = np.var(kpo.kpd)
    seps = srng[0] + (srng[1]-srng[0]) * np.arange(srng[2])/float(srng[2])
    cons = crng[0] + (crng[1]-crng[0]) * np.arange(crng[2])/float(crng[2])

    chi2map = np.zeros((srng[2], crng[2]))

    for i,sep in enumerate(seps):
        for j,con in enumerate(cons):
            chi2map[i,j] = np.sum(
                binary_KPD_fit_residuals([sep, pa, con], kpo)**2)
        sys.stdout.write("\r(%3d/%d): sep = %.2f mas" % (i+1, srng[2], sep))
        sys.stdout.flush()
    if reduced:
        chi2map /= kpo.kpd.size
        nullvar /= kpo.kpd.size
    plt.clf()

    asp = 1./np.abs((srng[1]-srng[0])/(crng[1]-crng[0]))
    plt.imshow(chi2map, aspect=asp, cmap=cmap,
               extent=[crng[0], crng[1], srng[0], srng[1]])
    plt.ylabel("ang. sep. (mas)")
    plt.xlabel("contrast prim/sec")

    print("\n----------------------------------------")
    print("Best "+reduced*"reduced "+"chi2 = %.3f" % (chi2map.min(),))
    print(reduced*"reduced "+"chi2 for null hypothesis: %.3f" % (nullvar,))
    arg = chi2map.argmin()
    print("Obtained for sep = %.1f mas" % (seps[arg // chi2map.shape[1]]))
    print("Obtained for con = %.1f"     % (cons[arg  % chi2map.shape[1]]))
    print("----------------------------------------")
    return chi2map
    
# =========================================================================
# =========================================================================
def chi2map_sep_pa(kpo, con=10., reduced=False, cmap=None,
                    srng=[10.0, 200.0, 100], arng=[0.0, 360.0, 60]):

    ''' Draws a 2D chi2 map of the binary parameter space for a given contrast
    ---------------------------------------
    con: contrast (primary/secondary)
    other options:
    - reduced: (boolean) reduced chi2 or not
    - cmap: (matplotlib color map) "jet" or "prism" are good here
    - srng: range of separations     [min, max, # steps]
    - arng: range of position angles [min, max, # steps]
    --------------------------------------- '''

    nullvar = np.var(kpo.kpd)
    seps = srng[0] + (srng[1]-srng[0]) * np.arange(srng[2])/float(srng[2])
    angs = arng[0] + (arng[1]-arng[0]) * np.arange(arng[2])/float(arng[2])
    nkphi = kpo.kpi.nkphi

    chi2map = np.zeros((srng[2], arng[2]))

    for i,sep in enumerate(seps):
        for j,th in enumerate(angs):
            chi2map[i,j] = np.sum(
                binary_KPD_fit_residuals([sep, th, con], kpo)**2)
        sys.stdout.write("\r(%3d/%d): sep = %.2f mas" % (i+1, srng[2], sep))
        sys.stdout.flush()

    if reduced:
        chi2map /= kpo.kpd.size
        nullvar /= kpo.kpd.size
    plt.clf()

    asp = 1./np.abs((srng[1]-srng[0])/(arng[1]-arng[0]))

    plt.imshow((chi2map), aspect=asp, cmap=cmap,
               extent=[arng[0], arng[1], srng[0], srng[1]])
    plt.xlabel("Position Angle (deg)")
    plt.ylabel("Separation (mas)")

    print("\n----------------------------------------")
    print("Best "+reduced*"reduced "+"chi2 = %.3f" % (chi2map.min()))
    print(reduced*"reduced "+"chi2 for null hypothesis: %.3f" % (nullvar,))
    arg = chi2map.argmin()
    print("Obtained for sep = %.1f mas" % (seps[arg // chi2map.shape[1]]))
    print("Obtained for ang = %.1f deg" % (angs[arg  % chi2map.shape[1]]))
    print("----------------------------------------")
    return chi2map
    

# =========================================================================
# =========================================================================
def chi2_volume(src, cal=None, regul="None"):
    ''' Proceed to an exhaustive search of the parameter space
    '''

    ns, s0, s1 = 50, 20.0, 200.0
    nc, c0, c1 = 50,  2.0, 100.0
    na, a0, a1 = 60,  0.0, 360.0
    
    seps = s0 + np.arange(ns) * (s1-s0) / ns
    angs = a0 + np.arange(na) * (a1-a0) / na
    cons = c0 + np.arange(nc) * (c1-c0) / nc

    chi2vol = np.zeros((ns,na,nc))

    sig = src.kpd
    if cal != None: sig -= cal.kpd

    for i,sep in enumerate(seps):
        for j, ang in enumerate(angs):
            for k, con in enumerate(cons):
                res = binary_multi_KPD_fit_residuals([sep, ang, con], src)
                chi2vol[i,j,k] = ((res)**2).sum()

    return chi2vol

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
    if b == None:

        mm = np.round(np.max(np.abs(kpo.kpd)), -1)
        
        f1 = plt.figure()
        sp0 = f1.add_subplot(111)
        if plot_error:
            sp0.errorbar(binary_KPD_model(kpo, params), kpo.kpd, 
                         yerr=kpo.kpe, linestyle='None')
        else:
            sp0.plot(binary_KPD_model(kpo, params), kpo.kpd, 'bo')
        sp0.plot([-mm,mm],[-mm,mm], 'g')
        sp0.axis([-mm,mm,-mm,mm])

        rms = np.std(binary_KPD_fit_residuals(params, kpo))
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
        
        plt.xlabel('Data kernel-phase signal (deg)')
        plt.ylabel('Kernel-phase binary model (deg)')
        plt.draw()
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