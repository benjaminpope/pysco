'''Frebin and pixel rescaling function
Written by Simon Zieleniewski (frebin copied from IDL)

Last modified 27-09-13
'''

import numpy as n
import pyfits as p
import scipy as s

def rebin(a, new_shape):
    """
    Resizes a 2d array by averaging or repeating elements, 
    new dimensions must be integral factors of original dimensions
 
    Parameters
    ----------
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array (y, x)
 
    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated
 
    See Also
    --------
    resize : Return a new array with the specified shape.
 
    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
 
    >>> c = rebin(b, (2, 3)) #downsize
    >>> c
    array([[ 0. ,  0.5,  1. ],
           [ 2. ,  2.5,  3. ]])
 
    """
    M, N = a.shape
    m, nn = new_shape
    if m<M:
        return a.reshape((m,M/m,nn,N/nn)).mean(3).mean(1)
    else:
        return n.repeat(n.repeat(a, m/M, axis=0), nn/N, axis=1)


    

def frebin(array, shape, total=True):
    '''Function that performs flux-conservative
    rebinning of an array.

    Inputs:
        array: numpy array to be rebinned
        shape: tuple (x,y) of new array size
	total: Boolean, when True flux is conserved

    Outputs:
	new_array: new rebinned array with dimensions: shape
    '''

    #Determine size of input image
    y, x = array.shape

    y1 = y-1
    x1 = x-1

    xbox = x/float(shape[0])
    ybox = y/float(shape[1])

    #Determine if integral contraction so we can use rebin
    if (x == int(x)) and (y == int(y)):
        if (x % shape[0] == 0) and (y % shape[1] == 0):
            return rebin(array, (shape[1], shape[0]))*xbox*ybox

    #Otherwise if not integral contraction
    #First bin in y dimension
    temp = n.zeros((shape[1], x),dtype=float)
    #Loop on output image lines
    for i in xrange(0, int(shape[1]), 1):
        rstart = i*ybox
        istart = int(rstart)
        rstop = rstart + ybox
        istop = int(rstop)
        if istop > y1:
            istop = y1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
        
    #Add pixel values from istart to istop an subtract
    #fracion pixel from istart to rstart and fraction
    #fraction pixel from rstop to istop.
        if istart == istop:
            temp[i,:] = (1.0 - frac1 - frac2)*array[istart,:]
        else:
            temp[i,:] = n.sum(array[istart:istop+1,:], axis=0)\
                        - frac1*array[istart,:]\
                        - frac2*array[istop,:]
            
    temp = n.transpose(temp)

    #Bin in x dimension
    result = n.zeros((shape[0], shape[1]), dtype=float)
    #Loop on output image samples
    for i in xrange(0, int(shape[0]), 1):
        rstart = i*xbox
        istart = int(rstart)
        rstop = rstart + xbox
        istop = int(rstop)
        if istop > x1:
            istop = x1
        frac1 = rstart - istart
        frac2 = 1.0 - (rstop - istop)
    #Add pixel values from istart to istop an subtract
    #fracion pixel from istart to rstart and fraction
    #fraction pixel from rstop to istop.
        if istart == istop:
            result[i,:] = (1.-frac1-frac2)*temp[istart,:]
        else:
            result[i,:] = n.sum(temp[istart:istop+1,:], axis=0)\
                          - frac1*temp[istart,:]\
                          - frac2*temp[istop,:]

    if total:
        return n.transpose(result)
    elif not total:
        return n.transpose(result)/float(xbox*ybox)


#Datacube Spaxel rescaling
def fits_spaxel_scale(image, pscale, spaxel, new_file):
    '''Function that takes image and rebins it to chosen
    spaxel scale. Reads spaxel scale of input image
    from header keywords CDELT1/2 - UNITS MUST BE 'MAS'. 

    Inputs:
        image: science image FITS file
        spaxel: spaxel scale mas (x, y)
        new_file: new FITS filename and path

    Outputs:
        new_cube: datacube rebinned to chosen spaxel scale
        new_head: updated header file
    '''

    print 'Spaxel scale'

    #image, head = p.getdata(fits_image, 0, header=True)
    hdu = p.PrimaryHDU()
    head = hdu.header
    head["CDELT1"] = spaxel
    head["CDELT2"] = spaxel

    y, x = image.shape
    cdelt1 = pscale
    cdelt2 = pscale

    #total field of view in mas
    x_field = cdelt1*x
    y_field = cdelt2*y

    x_newsize = x_field/float(spaxel)
    y_newsize = y_field/float(spaxel)

    new_im = n.zeros((y_newsize, x_newsize), dtype=float)

    print (x_newsize, y_newsize)

    new_im[:,:] = frebin(image, (x_newsize, y_newsize), total=True)

    head.update('CDELT1', spaxel, "mas")
    head.update('CDELT2', spaxel, "mas")

    print 'Spaxel scale - done!'

    p.writeto(str(new_file), new_im, head) 
    
 #   return new_im, head

#Datacube Spaxel rescaling
def spaxel_scale(image, pscale, spaxel,verbose=False):
    '''Function that takes image and rebins it to chosen
    spaxel scale. Reads spaxel scale of input image
    from header keywords CDELT1/2 - UNITS MUST BE 'MAS'. 

    Inputs:
        image: science image FITS file
        spaxel: spaxel scale mas (x, y)
        new_file: new FITS filename and path

    Outputs:
        new_cube: datacube rebinned to chosen spaxel scale
        new_head: updated header file

    This is Ben's hack to get it to take ordinary arrays
    '''

    #print 'Spaxel scale'

    y, x = image.shape
    cdelt1 = pscale #hacky hack hack
    cdelt2 = pscale

    #total field of view in mas
    x_field = cdelt1*x
    y_field = cdelt2*y

    x_newsize = x_field/float(spaxel)
    y_newsize = y_field/float(spaxel)

    new_im = n.zeros((y_newsize, x_newsize), dtype=float)

    # print (x_newsize, y_newsize)

    new_im[:,:] = frebin(image, (x_newsize, y_newsize), total=True)

    if verbose:
        print 'Spaxel scale - done!'

    # p.writeto(str(new_file), new_im, head) 
    
    return new_im
