import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.signal import medfilt2d as medfilt
from scipy.io.idl import readsav
import pyfits as pf
import sys
import multiprocessing
from scipy.interpolate import griddata

''' ================================================================
    Tools and functions useful for manipulating kernel phase data.
    ================================================================ '''

shift = np.fft.fftshift
fft   = np.fft.fft2
ifft  = np.fft.ifft2

dtor = np.pi/180.0
m_zero=1e-10 # machine zero

# =========================================================================
# =========================================================================

def mas2rad(x):
    ''' Convenient little function to convert milliarcsec to radians '''
    return x*np.pi/(180*3600*1000)

# =========================================================================
# =========================================================================

def rad2mas(x):
    ''' Convenient little function to convert radians to milliarcseconds '''
    return x/np.pi*(180*3600*1000)

# =========================================================================
# =========================================================================

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

# =========================================================================
# =========================================================================
#[AL: 2014.05.15] Normalization was fixed
def cvis_binary(u, v, wavel, p, norm=False):
    ''' Calc. complex vis measured by an array for a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)
    
    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)

    - norm=None (required for vis2)
    ---------------------------------------------------------------- '''

    p = np.array(p)
    # relative locations
    th = (p[1] + 90.0) * np.pi / 180.0
    ddec =  mas2rad(p[0] * np.sin(th))
    dra  = -mas2rad(p[0] * np.cos(th))

    # baselines into number of wavelength
    x = np.sqrt(u*u+v*v)/wavel

    # decompose into two "luminosity"
    l2 = 1. / (p[2] + 1)
    l1 = 1 - l2
    
    # phase-factor
    phi = np.zeros(u.size, dtype=complex)
    phi.real = np.cos(-2*np.pi*(u*dra + v*ddec)/wavel)
    phi.imag = np.sin(-2*np.pi*(u*dra + v*ddec)/wavel)

    # optional effect of resolved individual sources
    if p.size == 5:
        th1, th2 = mas2rad(p[3]), mas2rad(p[4])
        v1 = 2*j1(np.pi*th1*x)/(np.pi*th1*x)
        v2 = 2*j1(np.pi*th2*x)/(np.pi*th2*x)
    else:
        v1 = np.ones(u.size)
        v2 = np.ones(u.size)

    cvis = l1 * v1 + l2 * v2 * phi
    if norm :
        cvis/=(l1+l2)					
    return cvis

# =========================================================================
# =========================================================================

def phase_binary(u, v, wavel, p):
    ''' Calculate the phases observed by an array on a binary star
    ----------------------------------------------------------------
    p: 3-component vector (+2 optional), the binary "parameters":
    - p[0] = sep (mas)
    - p[1] = PA (deg) E of N.
    - p[2] = contrast ratio (primary/secondary)
    
    optional:
    - p[3] = angular size of primary (mas)
    - p[4] = angular size of secondary (mas)

    - u,v: baseline coordinates (meters)
    - wavel: wavelength (meters)
    ---------------------------------------------------------------- '''
				
    #[AL: 2014.05.15] Duplicated code was cleaned			
    phase = np.angle(cvis_binary(u, v, wavel, p), deg=True)
    return np.mod(phase + 10980., 360.) - 180.0

# =========================================================================
# =========================================================================

def vis2_binary(u, v, wavel, p):
    ''' -------------------------------------------------------------------
      Calculate the vis-squareds observed by an array on a binary star
    p: 3-component vector (+2 optional), the binary "parameters":
      p[0] = sep (mas)
      p[1] = PA (deg) E of N.
      p[2] = contrast ratio (primary/secondary)

    optional:
      p[3] = angular size of primary (mas)
      p[4] = angular size of secondary (mas)

    u,v: baseline coordinates (meters)
    wavel: wavelength (meters)
    ---------------------------------------------------------------- '''
    #[AL: 2014.05.15] Duplicated code was cleaned	
    cvis=cvis_binary(u, v, wavel, p, norm=True)				
    vis2 = np.real(cvis*cvis.conjugate())
    
    return vis2

# =========================================================================
# =========================================================================

def super_gauss(xs, ys, x0, y0, w):
    ''' Returns an 2D super-Gaussian function
    ------------------------------------------
    Parameters:
    - (xs, ys) : array size
    - (x0, y0) : center of the Super-Gaussian
    - w        : width of the Super-Gaussian 
    ------------------------------------------ '''

    x = np.outer(np.arange(xs), np.ones(ys))-x0
    y = np.outer(np.ones(xs), np.arange(ys))-y0
    dist = np.sqrt(x**2 + y**2)

    gg = np.exp(-(dist/w)**4)
    return gg

# =========================================================================
# =========================================================================

def centroid(image, threshold=0, binarize=0):                        
    ''' ------------------------------------------------------
        simple determination of the centroid of a 2D array
    ------------------------------------------------------ '''

    signal = np.where(image > threshold)
    sy, sx = image.shape[0], image.shape[1] # size of "image"
    bkg_cnt = np.median(image)                                       

    temp = np.zeros((sy, sx))
    if (binarize == 1): temp[signal] = 1.0
    else:               temp[signal] = image[signal]

    profx = 1.0 * temp.sum(axis=0)
    profy = 1.0 * temp.sum(axis=1)
    profx -= np.min(profx)                                           
    profy -= np.min(profy)

    x0 = (profx*np.arange(sx)).sum() / profx.sum()
    y0 = (profy*np.arange(sy)).sum() / profy.sum()

    return (x0, y0)

# =========================================================================
# =========================================================================

def find_psf_center(img, verbose=True, nbit=10):                     
    ''' Name of function self explanatory: locate the center of a PSF.

    ------------------------------------------------------------------
    Uses an iterative method with a window of shrinking size to 
    minimize possible biases (non-uniform background, hot pixels, etc)

    Options:
    - nbit: number of iterations (default 10 is good for 512x512 imgs)
    - verbose: in case you are interested in the convergence
    ------------------------------------------------------------------ '''
    temp = img.copy()
    bckg = np.median(temp)   # background level
    temp -= bckg
    mfilt = medfilt(temp, 3) # median filtered, kernel size = 3
    (sy, sx) = mfilt.shape   # size of "image"
    xc, yc = sx/2, sy/2      # first estimate for psf center

    signal = np.zeros_like(img)
    if np.max(mfilt) >= 10:
        signal[mfilt > 10] = 1.0
    else: 
        signal[mfilt>0] = 1.0
        print 'Warning - weak signal! Centering may not work.'

    for it in xrange(nbit):
        sz = sx/2/(1.0+(0.1*sx/2*it/(4*nbit)))
        x0 = np.max([int(0.5 + xc - sz), 0])
        y0 = np.max([int(0.5 + yc - sz), 0])
        x1 = np.min([int(0.5 + xc + sz), sx])
        y1 = np.min([int(0.5 + yc + sz), sy])
                                                                     
        mask = np.zeros_like(img)
        mask[y0:y1, x0:x1] = 1.0
        
        #plt.clf()
        #plt.imshow((mfilt**0.2) * mask)
        #plt.draw()

        profx = (mfilt*mask*signal).sum(axis=0)
        profy = (mfilt*mask*signal).sum(axis=1)
        
        xc = (profx*np.arange(sx)).sum() / profx.sum()
        yc = (profy*np.arange(sy)).sum() / profy.sum()
                  
        #pdb.set_trace()
                                                   
        if verbose:
            print("it #%2d center = (%.2f, %.2f)" % (it+1, xc, yc))
            
    return (xc, yc)                                                  

# =========================================================================
# =========================================================================

# window image im0 by applying sg_rad supergaussian
def window_image(im0,sg_rad) :
    if len(im0.shape)==2 :  
        szh = im0.shape[1] # horiz
        szv = im0.shape[0] # vertic
    elif len(im0.shape)==3 :  
        szh = im0.shape[2] # horiz
        szv = im0.shape[1] # vertic		
    temp = max(szh,szv) # max dimension of image			
    for sz in [64, 128, 256, 512, 1024, 2048]:
        if sz >= temp: break
    dz = sz/2.           # image half-size
    orih, oriv = (sz-szh)/2, (sz-szv)/2
    sgmask = super_gauss(sz, sz, dz, dz, sg_rad)				
    if len(im0.shape)==2 :  
        im = np.zeros((sz, sz))
        im[oriv:oriv+szv,orih:orih+szh] = im0
        im -= np.median(im)
        im*=sgmask
    elif len(im0.shape)==3 :  
        im = np.zeros((im0.shape[0],sz, sz))
        im[:,oriv:oriv+szv,orih:orih+szh] = im0
        for i in range(im.shape[0]) : 
            im[i] -= np.median(im[i])
            im[i]*=sgmask  
    return im
    


# =========================================================================
# =========================================================================

# [AL, 2014.07.28] extended to 3-d case
def recenter(im0, sg_rad=25.0, verbose=True, nbit=10, manual = 0):
    ''' ------------------------------------------------------------
         The ultimate image centering algorithm... eventually...

        im0:    of course, the array to be analyzed (2d)
        sg_rad: super-Gaussian mask radius
        bflag:  if passed as an argument, a "bad" boolean is returned

        BP: I've added a manual flag to this function to help with
        crowded fields. This is by default zero, which turns off 
        manual input. Otherwise, set it to manual = [window]
        it displays the full FITS image and gives you the chance to 
        click on a source to window a square region of side length
        2*[window] around it and proceed with recentering.
        ------------------------------------------------------------ '''
    dim=len(im0.shape)
    if manual != 0 and dim==2:
        plt.imshow(im0)
        print 'Manually windowing'
        print 'Click on the pixel at the new window centre... '
        newcenter = np.round(plt.ginput()[0]) #make sure it's an integer!
        plt.close()
        plt.clf() 
        print 'Centre Pixel',newcenter
        im0 = im0[(newcenter[1]-manual):(newcenter[1]+manual),(newcenter[0]-manual):(newcenter[0]+manual)]

    if len(im0.shape)==2 :  
        szh = im0.shape[1] # horiz
        szv = im0.shape[0] # vertic
    elif len(im0.shape)==3 :  
        szh = im0.shape[2] # horiz
        szv = im0.shape[1] # vertic		
    temp = max(szh,szv) # max dimension of image			
    for sz in [64, 128, 256, 512, 1024, 2048]:
        if sz >= temp: break
    dz = sz/2.           # image half-size
    orih, oriv = (sz-szh)/2, (sz-szv)/2
    sgmask = super_gauss(sz, sz, dz, dz, sg_rad)	
    
    x,y = np.meshgrid(np.arange(sz)-dz, np.arange(sz)-dz)
    wedge_x, wedge_y = x*np.pi/dz, y*np.pi/dz
    offset = np.zeros((sz, sz), dtype=complex) # to Fourier-center array

    # insert image in zero-padded array (dim. power of two)
    if len(im0.shape)==2 :  
        im = np.zeros((sz, sz))
        im[oriv:oriv+szv,orih:orih+szh] = im0
        (x0, y0) = find_psf_center(im, verbose, nbit)
        im -= np.median(im)
        mynorm = (im * sgmask).sum()
        dx, dy = (x0-dz), (y0-dz)
        # test
        #dx=1.0
        #dy=1.0	
        #print(" Test only! dx=%f, dy=%f " % (dx,dy))					
        im = np.roll(np.roll(im, -int(dx), axis=1), -int(dy), axis=0)        
        dx -= np.int(dx)
        dy -= np.int(dy)					
        # array for Fourier-translation
        dummy = shift(-dx * wedge_x + dy * wedge_y)
        offset.real, offset.imag = np.cos(dummy), np.sin(dummy)
        im = np.abs(shift(ifft(offset * fft(shift(im*sgmask)))))*sgmask
        # image masking, and set integral to right value        
        im=im * mynorm / im.sum()
    elif len(im0.shape)==3 :  
        im = np.zeros((im0.shape[0],sz, sz))
        im[:,oriv:oriv+szv,orih:orih+szh] = im0
        for i in range(im.shape[0]) : 
            (x0, y0) = find_psf_center(im[i], verbose, nbit) 
            im[i] -= np.median(im[i])
            mynorm = (im[i] * sgmask).sum()
            dx, dy = (x0-dz), (y0-dz)					
            im[i] = np.roll(np.roll(im[i], -int(dx), axis=1), -int(dy), axis=0)
            dx -= np.int(dx)
            dy -= np.int(dy)
            # array for Fourier-translation
            dummy = shift(-dx * wedge_x + dy * wedge_y)
            offset.real, offset.imag = np.cos(dummy), np.sin(dummy)
            im[i] = np.abs(shift(ifft(offset * fft(shift(im[i]*sgmask)))))*sgmask
            # image masking, and set integral to right value
            im=im[i] * mynorm / im[i].sum()

    return im

# =========================================================================
# =========================================================================

def get_keck_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle NIRC2 data. '''
    data = {
        'tel'    : hdr['TELESCOP'],        # telescope
        'pscale' : 10.0,                   # NIRC2 narrow plate scale (mas)
        'fname'  : hdr['FILENAME'],        # original file name
        'odate'  : hdr['DATE-OBS'],        # UTC date of observation
        'otime'  : hdr['UTC'     ],        # UTC time of observation
        'tint'   : hdr['ITIME'   ],        # integration time (sec)
        'coadds' : hdr['COADDS'  ],        # number of coadds
        'RA'     : hdr['RA'      ],        # right ascension (deg)
        'DEC'    : hdr['DEC'     ],        # declination (deg)
        'filter' : hdr['CENWAVE' ] * 1e-6, # central wavelength (meters)
        # P.A. of the frame (deg) (formula from M. Ireland)
        'orient' : 360+hdr['PARANG']+hdr['ROTPOSN']-hdr['EL']-hdr['INSTANGL']
        }
    print "parang = %.2f, rotposn = %.2f, el=%.2f, instangl=%.2f" % \
        (hdr['PARANG'],hdr['ROTPOSN'],hdr['EL'],hdr['INSTANGL'])
    return data

# =========================================================================
# =========================================================================
def get_nicmos_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle NICMOS1 data. '''

    if hdr['CAMERA'] == 1:
        pscale = 43.1
    elif hdr['CAMERA'] == 2:
        pscale = 75.8667 # note - different by a couple of % between X and Y! - we ignore this here
    data = {
        'tel'    : hdr['TELESCOP'],         # telescope
        'pscale' : pscale,                    # HST NIC1 plate scale (mas)
        'fname'  : hdr['FILENAME'],         # original file name
        'odate'  : hdr['DATE-OBS'],         # UTC date of observation
        'otime'  : hdr['TIME-OBS'],         # UTC time of observation
        'tint'   : hdr['EXPTIME' ],         # integration time (sec)
        'coadds' : 1,                       # as far as I can tell...
        'RA'     : hdr['RA_TARG' ],         # right ascension (deg)
        'DEC'    : hdr['DEC_TARG'],         # declination (deg)
        'filter' : hdr['PHOTPLAM'] * 1e-10, # central wavelength (meters)
        'orient' : hdr['ORIENTAT'] # P.A. of image y axis (deg e. of n.)
        }
    return data

# =========================================================================
# =========================================================================
def get_idl_keywords(filename):
    '''Extract the relevant keyword information from an idlvar file.
    '''
    data = readsav(filename)
    wavel,bwidth = data['filter']
    data['filter'] = wavel
    data['bwidth'] = bwidth
    data['pscale'] = rad2mas(data.rad_pixel)
    data['fname'] = filename
    data['orient'] = 0
    data['tel'] = 'NIC1'

    return data


# =========================================================================
# =========================================================================
def get_pharo_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This version is adapted to handle PHARO data. '''

    data = {
        'tel'      : hdr['TELESCOP'],         # telescope
        'pscale'   : 25.2,                    # HST NIC1 plate scale (mas)
        'odate'    : hdr['DATE-OBS'],         # UTC date of observation
        'otime'    : hdr['TIME-OBS'],         # UTC time of observation
        'tint'     : hdr['T_INT' ],           # integration time (sec)
        'coadds'   : 1,                       # as far as I can tell...
        'RA'       : hdr['CRVAL1'],           # right ascension (deg)
        'DEC'      : hdr['CRVAL2'],           # declination (deg)
        'filter'   : np.nan, # place-holder   # central wavelength (meters)
        'filtname' : hdr['FILTER'],           # Filter name
        'grism'    : hdr['GRISM'],            # additional filter/nd
        'pupil'    : hdr['LYOT'],             # Lyot-pupil wheel position
        'orient'   : hdr['CR_ANGLE']          # Cassegrain ring angle
        }

    if 'H'       in data['filtname'] : data['filter'] = 1.635e-6
    if 'K'       in data['filtname'] : data['filter'] = 2.196e-6
    if 'CH4_S'   in data['filtname'] : data['filter'] = 1.570e-6
    if 'K_short' in data['filtname'] : data['filter'] = 2.145e-6
    if 'BrG'     in data['filtname'] : data['filter'] = 2.180e-6
    if 'FeII'     in data['grism']   : data['filter'] = 1.648e-6
    
    if np.isnan(data['filter']):
        print("Filter configuration un-recognized. Analysis will fail.")
    return data

# =========================================================================
# =========================================================================
def get_simu_keywords(hdr):
    '''Extract the relevant keyword information from a fits header.

    This is a special version for simulated data. '''
    	
    # [AL,21.02.2014] Header parser was modified to support wider range of simulations	
    keys = hdr.ascardlist().keys()		
    if 'PSCALE' in keys :
        pscale=hdr['PSCALE']
    else :
	  pscale=11.5
    if 'FNAME' in keys :
        fname=hdr['FNAME']
    else :
	  fname='simulation'
    if 'ODATE' in keys :
        odate=hdr['ODATE']
    else :
	  odate='Jan 1, 2000'
    if 'OTIME' in keys :
        otime=hdr['OTIME']
    else :
	  otime='0:00:00.00'	
    if 'TINT' in keys :
        tint=hdr['TINT']
    else :
	  tint=1.0		
    if 'COADDS' in keys :
        coadds=hdr['COADDS']
    else :
	  coadds=1		
    if 'RA' in keys :
        RA=hdr['RA']
    else :
	  RA=0.0
    if 'DEC' in keys :
        DEC=hdr['DEC']
    else :
	  DEC=0.0	
    if 'FILTER' in keys :
        filter=hdr['FILTER']
    else :
	  filter=1.6* 1e-6
    if 'ORIENT' in keys :
        orient=hdr['ORIENT']
    else :
	  orient=0.0				
    data = {
        'tel'    : hdr['TELESCOP'],        # telescope
        'pscale' : pscale,                 # simulation plate scale (mas)
        'fname'  : fname,           	   # original file name
        'odate'  : odate,          	   # UTC date of observation
        'otime'  : otime,                  # UTC time of observation
        'tint'   : tint,                   # integration time (sec)
        'coadds' : coadds,                 # number of coadds
        'RA'     : RA,                     # right ascension (deg)
        'DEC'    : DEC,                    # declination (deg)
        'filter' : filter,                 # central wavelength (meters)
        'orient' : orient                  # P.A. of the frame (deg)
        }
    return data

# =========================================================================
# =========================================================================
# [AL, 2014.04.16] Added sg_ld and D parameters - window size in lambda/D
#		       if D<=0 then use the shortest baseline instead
# [AL, 2014.04.16] windowing parameters added
# [AL, 2014.05.06] Bispectrum angle (bsp)
# [AL, 2014.05.28] Adjust sampling points in pupil plane to make coordinates in uv-plane integer (increases quality)
# [AL, 2014.05.28] Recentering and windowing parameters added
# [AL, 2014.05.28] The same definitions (except hdr) for extract_from_array and extract_from_fits_frame functions
# [AL, 2014.05.29] Description updated
# [AL, 2014.10.07] unwrap_kp flag added. Kernel phases unwrapping is off by default
def extract_from_array(array, hdr, kpi, save_im=False, wfs=False, plotim=False, manual=0,  wrad=25.0, sg_ld=1.0, D=0.0,re_center=True, window=True,  bsp=False, adjust_sampling=True, unwrap_kp=False):
    ''' Extract the Kernel-phase signal from a ndarray + header info.
    
    ----------------------------------------------------------------
    Assumes that the array has been cleaned.
    In order to be able to extract information at the right place,
    a header must be provided as additional argument. This function
    replaces extract_from_fits_frame() when working with a fits
    datacube (multiple frames, one single header).
    
    In addition to the actual Kernel-phase signal, some information
    is extracted from the fits header to help with the interpretation
    of the data to follow.
    
    Parameters are:
    - array: the frame to be examined
    - kpi: the k-phase info structure to decode the data
    - save_im: optional flag to set to False to forget the images
    and save some RAM space

    Options:
    - save_im: optional flag to set to False to forget the images
    and save some RAM space
    - wfs: wavefront sensing. Instead of kernel-phase, returns phases.
    - plotim: show image and uv plane on plot 
    - manual: manually select the object you want in an interactive 
    display.
    - wrad: window radius (default = 25 pixels). If both sg_ld and D are >0 then it is not used
    - sg_ld: window size in lambda/D 
    - D: Diameter of the pupil 
    - re_center : apply recentering algorithm
    - window : apply windowing function (if recentering is off)
    - bsp: extract bispectral phases
    - adjust_sampling: Adjust sampling points in pupil plane to make coordinates in uv-plane integer (increases quality but uses slightly different sampling points). kpi is also modified in this case

    The function returns a tuple:
    - (kpd_info, kpd_signal [, bsp_res])
    - (kpd_info, kpd_signal, im, ac [,bsp_res])
    - (kpd_info, kpd_phase [,bsp_res])

    ---------------------------------------------------------------- '''

    if 'Keck II' in hdr['TELESCOP']: kpd_info = get_keck_keywords(hdr)
    if 'HST'     in hdr['TELESCOP']: kpd_info = get_nicmos_keywords(hdr)
    if 'simu'    in hdr['TELESCOP']: kpd_info = get_simu_keywords(hdr)
    if 'Hale'    in hdr['TELESCOP']: kpd_info = get_pharo_keywords(hdr)
					
    rev = -1.0	
    if ('Hale' in hdr['TELESCOP']) or ('simu' in hdr['TELESCOP']): # P3K PA are clockwise
                                                                    # [AL, 04.07.2014] removed reverse from simulation		
        rev = 1.0								
    								
    # [AL, 2014.04.16] Added calculation of super gaussian radius in sg_ld*lambda/D
    if sg_ld*D>0 :							
        bl = D
        if D<=0 :
            bl=np.hypot(kpi.uv[:,0],kpi.uv[:,1]).min()
        wl=kpd_info['filter']
        pscale=kpd_info['pscale']
        sg_rad=int(rad2mas(wl/bl)/pscale)+1   
        sg_rad*=sg_ld	
        sg_rad+=(sg_rad%2)	
    elif wrad>0 :
        sg_rad=wrad
					
				
    # read and fine-center the frame
    # [AL, 2014.04.16] sg_rad=wrad changed
    if sg_rad<=0 : sg_rad=array.shape[1]
    if re_center:         
        im = recenter(array, sg_rad=sg_rad, verbose=False, nbit=20,manual=manual)
    elif window:         
        im = window_image(im0=array, sg_rad=sg_rad)
    else:
        im = array.copy()

    sz, dz = im.shape[0], im.shape[0]/2  # image is now square

    # meter to pixel conversion factor
    m2pix = mas2rad(kpd_info['pscale']) * sz / kpd_info['filter']

    # rotation of samples according to header info
    #th = 90.0 * np.pi/180.
    #rmat = np.matrix([[np.cos(th), np.sin(th)], [np.sin(th), -np.cos(th)]])
    #uv_rot = np.dot(rmat, kpi.uv.T).T

    uv_samp = kpi.uv * m2pix + dz # uv sample coordinates in pixels
    #uv_samp = uv_rot * m2pix + dz # uv sample coordinates in pixels
				
    if adjust_sampling: 
        uv_samp=adjust_samp(uv_samp,kpi,m2pix,sz) # rounding if not integer							
       
    # calculate and normalize Fourier Transform
    ac = shift(fft(shift(im)))
    ac /= (np.abs(ac)).max() / kpi.nbh

    # [AL, 2014.06.20] visibilities extraction
    uv_samp_rev=np.cast['int'](np.round(uv_samp))
    uv_samp_rev[:,0]*=rev
    data_cplx=ac[uv_samp_rev[:,1], uv_samp_rev[:,0]]				
				
    vis = np.real(ac*ac.conjugate())
    viscen = vis.shape[0]/2
    vis2 = np.real(data_cplx*data_cplx.conjugate())
    vis2 /= vis[viscen,viscen] #normalise to the origin

    # ---------------------------
    # calculate the Kernel-phases
    # ---------------------------

    #kpd_phase = kpi.RED * np.angle(data_cplx) # in radians for WFS # [Al, 2014.05.12] Replaced by Frantz's version
    #kpd_signal = np.dot(kpi.KerPhi, kpd_phase) / dtor # [Al, 2014.05.12] Replaced by Frantz's version
    #kpd_phase = np.angle(data_cplx) # uv-phase (in radians for WFS) #[Al, 2014.05.12] Frantz's version #[AL, 2014.07.30 Replaced]

    if not unwrap_kp :
        kpd_phase = np.angle(data_cplx) # uv-phase (in radians for WFS) #[Al, 2014.05.12] Frantz's version #[AL, 2014.07.30 Replaced]
    else :
        kpd_phase=unwrap_uv_phases(ac,uv_samp_rev,maxVar=np.pi) 
    
    
    kpd_signal = np.dot(kpi.KerPhi, kpd_phase) / dtor #[Al, 2014.05.12] Frantz's version
						
    # [AL, 2014.05.06] Bispectrum (bsp)
    if bsp :	   				
        bsp_res=extract_bsp(data_cplx,kpi.uvrel) # robust to phase wrapping
       # bsp_res=extract_bsp(kpd_phase,kpi.uvrel,rng=(0,50000)) # works if unwrapping algorithm is fine. Can be used to check it as it is a requirement for kpd extraction
								
    if bsp :   
        if (save_im): res = (kpd_info, kpd_signal,vis2, im, ac, bsp_res)
        else:         res = (kpd_info, kpd_signal,vis2, bsp_res)
        if (wfs):     res = (kpd_info, kpd_phase, bsp_res)					
    else :
        if (save_im): res = (kpd_info, kpd_signal,vis2, im, ac)
        else:         res = (kpd_info, kpd_signal,vis2)
        if (wfs):     res = (kpd_info, kpd_phase)

    
    if plotim:
        uvw = np.max(uv_samp)/2
        plt.clf()
        plt.figure(1, (15,5))
        f0 = plt.subplot(131)
        f0.imshow(im**0.5)
				
	#[AL, 2014.03.03 : Added power spectrum as well]				
        f1 = plt.subplot(132)       
        f1.imshow(np.abs(ac))								
        f1.plot(uv_samp_rev[:,0], uv_samp_rev[:,1], 'b.')
        f1.axis((dz-uvw, dz+uvw, dz-uvw, dz+uvw))	
								
        f2 = plt.subplot(133)
        f2.imshow(np.angle(ac)) 								
        f2.plot(uv_samp_rev[:,0], uv_samp_rev[:,1], 'b.')
        f2.axis((dz-uvw, dz+uvw, dz-uvw, dz+uvw))								
        plt.draw()
        plt.show()

    return res

# =========================================================================
# =========================================================================
# [AL, 2014.03.10] Added plotim parameter
# [AL, 2014.03.18] Added sg_ld and D parameters - window size in lambda/D
#		       if D<=0 then use the shortest baseline instead
# [AL, 2014.03.20] Recentering and windowing parameters added
# [AL, 2014.05.06] Bispectrum phases (bsp) 
# [AL, 2014.05.20] Interpolation for image grid added. grid_size - size of the grid of the image to calculate interpolation on
# [AL, 2014.07.23] inter_type - method for interpolation (linear, nearest, cubic)
# [AL, 2014.05.28] Adjust sampling points in pupil plane to make coordinates in uv-plane integer (increases quality)
# [AL, 2014.05.29] The same definitions (except hdr) for extract_from_array and extract_from_fits_frame functions. extract_from_fits_frame is now defined via extract_from_array function
# [AL, 2014.05.29] Description updated
# [AL, 2014.10.07] unwrap_kp flag added. Kernel phases unwrapping is off by default
def extract_from_fits_frame(fname, kpi, save_im=True, wfs=False, plotim=True, manual=0, wrad=25.0, sg_ld=1.0, D=0.0, re_center=True, window=True, bsp=False, adjust_sampling=True, unwrap_kp=False):

    ''' Extract the Kernel-phase signal from a single fits frame.
    
    ----------------------------------------------------------------
    Assumes that the fits file has been somewhat massaged, and is
    pretty much ready to go: no pairwise subtraction necessary, 
    etc...
    
    In addition to the actual Kernel-phase signal, some information
    is extracted from the fits header to help with the interpretation
    of the data to follow (think orientation of the telescope!).

    Parameters are:
    - fname: the frame to be examined
    - kpi: the k-phase info structure to decode the data

    Options:
    - save_im: optional flag to set to False to forget the images
    and save some RAM space
    - wfs: wavefront sensing. Instead of kernel-phase, returns phases.
    - plotim: show image and uv plane on plot 
    - manual: manually select the object you want in an interactive 
    display.
    - wrad: window radius (default = 25 pixels). If both sg_ld and D are >0 then it is not used
    - sg_ld: window size in lambda/D 
    - D: Diameter of the pupil 
    - re_center : apply recentering algorithm
    - window : apply windowing function (if recentering is off)
    - bsp: extract bispectral phases
    - adjust_sampling: Adjust sampling points in pupil plane to make coordinates in uv-plane integer (increases quality but uses slightly different sampling points). kpi is also modified in this case

    The function returns a tuple:
    - (kpd_info, kpd_signal)
    - (kpd_info, kpd_signal, im, ac)
    - (kpd_info, kpd_phase)
    ----------------------------------------------------------------  '''
    im0=pf.getdata(fname)
    hdr = pf.getheader(fname)
    return extract_from_array(im0, hdr, kpi, save_im=save_im, wfs=wfs, plotim=plotim, manual=manual,  wrad=wrad, sg_ld=sg_ld, D=D,re_center=re_center, window=window,  bsp=bsp, adjust_sampling=adjust_sampling,unwrap_kp=unwrap_kp)


# [AL, 2014.07.25]
# this functions alters sampling points in uv plane in order to get integer sampling in uv plane
# uv_samp - intitial sampling in pixels
# kpi - kpi object with uvrel matrix defined (kpi is modified as well)
# m2pix - conversion factor meters to pixels
# sz - image size
# save_kpi - save changes in the initial kpi
def adjust_samp(uv_samp,kpi,m2pix,sz,save_kpi=True) :
    dz=sz/2.
    idx=-np.ones((kpi.nbh-1),dtype='int') # indices of uv-points related to sampling point 0
    for i in range(1,kpi.nbh) :
        if kpi.uvrel[0,i]>0 : idx[i-1]=kpi.uvrel[0,i]
        else : idx[i-1]=-kpi.uvrel[i,0]

    eps=(uv_samp[np.abs(idx)]-np.minimum(np.round(uv_samp[np.abs(idx)]),sz-1))/m2pix # calculate shifts of sampling points in pupil plane
    #correcting for the sign if required
    eps[:,0]*=np.sign(idx)
    eps[:,1]*=np.sign(idx)

    if save_kpi: 
        mask=kpi.mask
        uv=kpi.uv
    else: 
        mask=np.copy(kpi.mask)
        uv=np.copy(kpi.uv)
    mask[1:kpi.nbh,:]+=eps # sampling points are updated

    # converting back sampling points to uv-points
    for i in range(0,kpi.nbh-1) :
        for j in range(i+1,kpi.nbh) :
            if kpi.uvrel[i,j]>=0 : uv[kpi.uvrel[i,j]]=mask[i,:]-mask[j,:]
            else : uv[kpi.uvrel[j,i]]=mask[j,:]-mask[i,:]

    # converting to pixels once again
    return uv * m2pix + dz

# [AL, 2014.07.29]				
# ---- this function unwraps phases for a given set of uv points by goint to each uv point from across a line from the center ------
#data # initial array with phases or complex visibilities (square image)
#uv0    # array with sampling points
#maxVar # maximum variation between positive and negative phases
#return_image # return image with unwrapped phases
# output :
# - unwrapped phases for sampling at given uv-points
def unwrap_uv_phases(data,uv0,maxVar=np.pi,return_image=False) :
    two_pi=np.pi*2
    sz=data.shape[0] # square image size
    dz=sz//2
    #shifting initial array
    uv=np.array(np.round(uv0-dz),dtype=int)
    if data.dtype=='complex' :				
        phases=np.angle(data)
    else : 	phases=np.copy(data)							
    res=np.empty((uv0.shape[0]),dtype=phases.dtype)
    # determining the line function between each of the sampling points and image center y=ax
    for i in range(uv.shape[0]) : 
        xShift=True
        if uv[i,1]==0 and uv[i,0]==0:
            continue # no unwrapping in case of image center		
        elif uv[i,1]==0 :           # going vertically along y axis
            dx=0.
            dy=np.sign(uv[i,0])
            xShift=False
        elif uv[i,0]==0 :          # going horizontally along x axis
            dx=np.sign(uv[i,1])
            dy=0.																		
        else :			   # determining x and y shifts
            dy=np.abs(1.*uv[i,0]/uv[i,1])*np.sign(uv[i,0])
            if np.abs(dy)<=1 :
                dx=np.sign(uv[i,1])
            else :
                dx=np.abs(1.*uv[i,1]/uv[i,0])*np.sign(uv[i,1])
                dy=np.sign(uv[i,0])
                xShift=False								
        if xShift:				
            x=dz
            y1=dz
            y2=dz				
            while (np.abs(uv[i,1])>np.abs(x-dz)) :
                x+=dx
                y=np.int(np.abs(x-dz)*dy+dz)
                if (np.abs(phases[x-dx,y1]-phases[x,y])>maxVar) or \
		         (np.abs(phases[x-dx,y2]-phases[x,y])>maxVar) :
                    dist=np.abs(phases[x-dx,y1]-phases[x,y])%two_pi
                    if (dist%two_pi<=np.pi) :
                        phases[x,y]=phases[x-dx,y1]-np.sign(phases[x-dx,y1])*dist
                    else :
                        phases[x,y]=phases[x-dx,y1]+np.sign(phases[x-dx,y1])*(two_pi-dist)													                    
                    if (np.abs(phases[x,y]-phases[x,y+1])>maxVar)	:
                        dist=np.abs(phases[x,y]-phases[x,y+1])%two_pi
                        if (dist%two_pi<=np.pi) :
                            phases[x,y+1]=phases[x,y]-dist*np.sign(phases[x,y])
                        else :
                            phases[x,y+1]=phases[x,y]+np.sign(phases[x,y])*(two_pi-dist)
                    #print("1) f(%d,%d)=%f | f(%d,%d)=%f \n   f(%d,%d)=%f | f(%d,%d)=%f" % (x,y,phases[x,y],x,y+1,phases[x,y+1],x-dx,y1,phases[x-dx,y1],x-dx,y2,phases[x-dx,y2]))																								
                if (dy!=0) and \
                    ((np.abs(phases[x-dx,y1]-phases[x,y+1])>maxVar) or \
                    (np.abs(phases[x-dx,y2]-phases[x,y+1])>maxVar)) :
                    dist=np.abs(phases[x-dx,y1]-phases[x,y+1])%two_pi
                    if (dist%two_pi<=np.pi) :
                        phases[x,y+1]=phases[x-dx,y1]-np.sign(phases[x-dx,y1])*dist
                    else :
                        phases[x,y+1]=phases[x-dx,y1]+np.sign(phases[x-dx,y1])*(two_pi-dist)
                    if (np.abs(phases[x,y]-phases[x,y+1])>maxVar)	:
                        dist=np.abs(phases[x,y]-phases[x,y+1])%two_pi
                        if (dist%two_pi<=np.pi) :
                            phases[x,y]=phases[x,y+1]-dist*np.sign(phases[x,y+1])
                        else :
                            phases[x,y]=phases[x,y+1]+np.sign(phases[x,y+1])*(two_pi-dist)																		
                    #print("2) f(%d,%d)=%f | f(%d,%d)=%f | f(%d,%d)=%f" % (x,y+1,phases[x,y+1],x-dx,y1,phases[x-dx,y1],x-dx,y2,phases[x-dx,y2]))																					
                y1=y        												
                y2=y+1   
        else :				
            y=dz
            x1=dz
            x2=dz				
            while (np.abs(uv[i,0])>np.abs(y-dz)) :
                y+=dy
                x=np.int(np.abs(y-dz)*dx+dz)
                if (np.abs(phases[x1,y-dy]-phases[x,y])>maxVar) or \
                   (np.abs(phases[x2,y-dy]-phases[x,y])>maxVar) :																				
                    dist=np.abs(phases[x1,y-dy]-phases[x,y])%two_pi
                    if (dist%two_pi<=np.pi) :
                        phases[x,y]=phases[x1,y-dy]-dist*np.sign(phases[x1,y-dy])
                    else :
                        phases[x,y]=phases[x1,y-dy]+np.sign(phases[x1,y-dy])*(two_pi-dist)													                    
                    if (np.abs(phases[x,y]-phases[x+1,y])>maxVar)	:
                        dist=np.abs(phases[x,y]-phases[x+1,y])%two_pi
                        if (dist%two_pi<=np.pi) :
                            phases[x+1,y]=phases[x,y]-dist*np.sign(phases[x,y])
                        else :
                            phases[x+1,y]=phases[x,y]+np.sign(phases[x,y])*(two_pi-dist)																												
                if (dx!=0) and \
                   ((np.abs(phases[x2,y-dy]-phases[x+1,y])>maxVar) or \
                    (np.abs(phases[x2,y-dy]-phases[x+1,y])>maxVar)):																				
                    dist=np.abs(phases[x1,y-dy]-phases[x+1,y])%two_pi
                    if (dist%two_pi<=np.pi) :
                        phases[x+1,y]=phases[x1,y-dy]-dist*np.sign(phases[x1,y-dy])
                    else :
                        phases[x+1,y]=phases[x1,y-dy]+np.sign(phases[x1,y-dy])*(two_pi-dist)	
                    if (np.abs(phases[x,y]-phases[x+1,y])>maxVar)	:
                        dist=np.abs(phases[x,y]-phases[x+1,y])%two_pi
                        if (dist%two_pi<=np.pi) :
                            phases[x,y]=phases[x+1,y]-dist*np.sign(phases[x+1,y])
                        else :
                            phases[x,y]=phases[x+1,y]+np.sign(phases[x+1,y])*(two_pi-dist)																									
                x1=x        												
                x2=x+1																
        res[i]=phases[uv[i,1]+dz,uv[i,0]+dz]																															
    if not return_image : 
        return res  
    else :
        return res,phases						

# =========================================================================
# =========================================================================

#[AL, 07.05.2014] Extract bispectral phases for a given set of visibilities
# vis - input visibilities or phases (in radians!!!)
# uvrel - relations matrix between uv points and sampling points 
# deg - return result in degrees 
# rng - upper and lower bounds for bsp to be extracted. Used to reduce the resources usage
# nonred - extract non-redundant Bsp only. Much slower
def extract_bsp(vis,uvrel,deg=True,rng=(0,50000),nonred=False):
    # determining number of sampling points
    nsp=uvrel.shape[0]
    # creating a relationship								
    # maximum number of triangles								
    n=nsp*(nsp-1)*(nsp-2)/6	
    # number of uv points				
    nuv=uvrel.max()+1
    # a matrix to avoid redundancy
    if nonred : red=np.zeros((nuv**3,),dtype='bool')				
    # determining lower limit				
    if rng[0]>=0 and rng[0]<n:
        l=rng[0]
    elif rng[0]<0 :
        l=0
    else:
        l=n
    # determining upper limit								
    if rng[1]>l and rng[1]<=n:
        u=rng[1]
    elif rng[1]<l :
        u=l
    else:
        u=n							
    bsp_res=[]
    # visibilities or phases?
    isComplex=True				
    if nsp>0 :
        if vis[0].dtype!='complex' :
            isComplex=False									
    if u-l>0 :				
        total=0							
        for i in range(0,nsp) :									
            if total>=u :
                break									
            sys.stdout.write("\r                                     | Extracting bsp from img %d of %d" % (total+1-l,u))																													
            for j in range(i+1,nsp) :													
                if total>=u :
                    break														
                for k in range(j+1,nsp) :																	
                    if total>=l:
                        uv1=max(uvrel[i,j],uvrel[j,i])																						
                        uv2=max(uvrel[j,k],uvrel[k,j])																						
                        uv3=max(uvrel[i,k],uvrel[k,i])
                        isRed=False																								
                        if nonred : 
                            redidx=uv1*nuv*nuv+uv2*nuv+uv3
                            isRed=red[redidx]																												
                        if uv1>-1 and uv2>-1 and uv3>-1 and (not isRed):		
                            if nonred : red[redidx]=True
                            # Dealing with baseline directions for each of the three visibilities																					
                            if uvrel[i,j]>=0 :																											
                                v1=vis[uv1]																												
                            else :																							
                                if isComplex : v1=np.conjugate(vis[uv1])
                                else : v1=-vis[uv1]
                            if uvrel[j,k]>=0 :																									
                                v2=vis[uv2]
                            else :																								
                                if isComplex : v2=np.conjugate(vis[uv2])
                                else : v2=-vis[uv2] 
                            # the 3rd visibility is reverted to form a triangle																																
                            if uvrel[i,k]>=0 :
                                if isComplex : v3=np.conjugate(vis[uv3])
                                else : v3=-vis[uv3]																														
                            else :																									
                                v3=vis[uv3]																							
                            if isComplex :																					
                                bsp_res.append(np.angle(v1*v2*v3))
                            else :
                                bsp_res.append(v1+v2+v3)																												
                            total+=1																											
                    if total>=u :
                        break
    res=np.asarray(bsp_res)
    if deg :																							
        return res/dtor
    else :
        return res			

# =========================================================================
# =========================================================================
		
def clip_signal(signal,threshold=3):
    '''Cut out bad points'''

    newsignal = signal
    sigma = np.std(newsignal)
    j = 0

    while np.max(np.abs(newsignal-np.median(newsignal)))>=threshold*sigma and j<10:
        sigma = np.std(newsignal)
        newsignal = newsignal[(np.abs(newsignal-np.median(newsignal)))<=threshold*sigma]
        j+=1
        if j >6: 
            print 'warning, many iterations!'

    return newsignal

# =========================================================================
# =========================================================================

def reject_outliers(data, m=2):
    return data[abs(data - np.median(data)) < m * np.std(data)]