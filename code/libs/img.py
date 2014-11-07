# -*- coding: utf-8 -*-
# Written by Guillaume, 2014

import numpy as np
import cairo
from scipy.ndimage import gaussian_filter as _blur_image

import libs.funcs as funcs



def azimuthal(image, center=None, zone_radius=None, bin_size=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).

    TODO : implémenter bin_size
    
    """
    # Calculate the indices from the image
    if len(image.shape)>2: raise Exception, "can only take 2D images"
    y, x = np.indices(image.shape)

    if center is None:
        center = np.asarray(image.shape)/2.

    if zone_radius is None:
        zone_radius = 1e99

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]<zone_radius
    i_sorted = image.flat[ind][r_sorted]
    r_sorted = r.flat[ind][r_sorted]

    if bin_size is None: return r_sorted, i_sorted


    # Get the integer part of the radii (bin size = 1)
    r_int = np.round(r_sorted).astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof



def add_poisson_noise(data):
    """
    Adds poisson noise to a N-dim array. Replaces each value of the array with a random new value following poisson distribution whose mean is the initial value of the given array.

    :param data: N-dim array
    :type data: real
    :returns: the same N-dim array where each value has been replaced with a random poisson distribution value with same mean (integer values)

    >>> import misc.funcs as funcs
    >>> funcs.add_poisson_noise([[4.9,5.3],[5.1,5.2]])
    array([[ 3.,  4.],
           [ 6.,  7.]])
    """
    # poisson function doesn't work for too large numbers of lam (~1.e9). This is fixed with a dummy reduction factor
    data = np.asarray(data).copy()
    # applies the poisson random process
    rnd = funcs.gen_generator()
    filt = data<1e9
    data[filt] = rnd.poisson(lam=data[filt], size=np.size(data[filt]))
    return data



def find_boundaries(center, subframesize, framesize=None, mode="fit", integer=True, fuckingvariablewhichnobodyknowswhatisit="hui"):
    """
    Returns the coordinates of a subframe in a frame, given its center and shape.

    :param center: the N-dim array of the subframe center's coordinates in the frame
    :type center: real or array of real
    :param subframesize: the N-dim array of the subframe shape
    :type subframesize: int or array of int
    :param framesize: the N-dim array of the frame shape, None means "infinite" which is equivalent to ``mode``=``simple``
    :type framesize: int or array of int
    :param mode:
        * *simple*: no checks on the frame borders are performed
        * *fit*: in the case of a subframe centered too close to a frame border, the subframe center is shifted so the subframe fits inside the frame with the requested sub-frame size
        * *cut*: in the case of a subframe centered too close to a frame border, the subframe is truncated so it doesn't go outside the frame. The sub-frame size is not conserved
    :type mode: keyword
    :param integer: forces the output to be integers
    :type integer: bool
    :returns: a N-tuple of 2-element arrays ((xmin, xmax), (ymin, ymax), ...) of the subframe coordinates in the frame
    :raises: Exception, if the subframe is smaller than the frame in at least 1 dimension

    >>> import misc.funcs as funcs
    >>> funcs.find_boundaries([5.32, 16.76], subframesize=14, framesize=20,
        mode="fit")
    (array([ 0, 14]), array([ 6, 20]))
    """
    if fuckingvariablewhichnobodyknowswhatisit!="hui": print "hui..."
    dim = np.asarray(center).flatten().size # takes dim of center and assumes it is the dim of data
    subframesize = np.asarray(subframesize) # transforms to np
    subframesize = np.tile(subframesize, dim//subframesize.size+1)[:dim] # if dim(subframesize)!=dim(center), repeats or crops
    if framesize is None: mode="simple"
    framesize = np.asarray(framesize) # transforms to np
    framesize = np.tile(framesize, dim//framesize.size+1)[:dim] # if dim(framesize)!=dim(center), repeats or crops
    if (subframesize>framesize).any(): raise Exception, "Subframe smaller than frame"
    subframesize_i = subframesize/2.
    if integer:
        subframesize = np.round(subframesize).astype(int)
        framesize = np.round(framesize).astype(int)
        subframesize_i = subframesize_i.astype(int)
    if mode.upper()=="SIMPLE": # don't care about checking borders
        if integer:
            mins = np.round(center - subframesize//2).astype(int)
            return tuple(np.c_[mins, mins + subframesize])
        else:
            return tuple(np.c_[center - subframesize/2., center + subframesize/2.])
    # defines the clipmin of subframe if we care about checking borders of frame
    elif mode.upper()=="CUT":
        clipmin = -1e99 # if truncate, set ridiculous low inferior border to calculate true upper bound from it
    else:
        clipmin = 0 # if check of borders
    mins = np.clip(center - subframesize_i, clipmin, framesize) # defines inferior limit
    maxs = np.clip(mins + subframesize, clipmin, framesize) # from inferior limit, calculates max limit
    if mode.upper()=="CUT":
        mins = np.clip(mins, 0, framesize) # truncates the subframe
    else:
        mins = maxs - subframesize # from checked max limit, redefines inferior limit if shift subwindow is allowed
    if integer:
        return tuple(np.round(np.c_[mins,maxs]).astype(int))
    else:
        return tuple(np.c_[mins,maxs]) # returns xmin, xmax, ymin, ymax, zmin, zmax, ...



def subframe(data, center, subframesize, mode="fit", fill_value=0., by_ref=False):
    """
    Selects a N-dim subframe out of a N-dim frame, given its center and shape.

    :param data: the N-dim array from which the subframe will be taken
    :param center: the N-dim array of the subframe center's coordinates in the frame
    :type center: real or array of real
    :param subframesize: the N-dim array of the subframe shape
    :type subframesize: real or array of real
    :param mode:
        * *fill*: in case the subframe goes over the edge of the frame, conserves the center location in the frame and the subfram size by filling the output with value ``fill_value``
        * *fit*: in the case of a subframe centered too close to a frame border, the subframe is shifted so the subframe fits inside the frame with the requested sub-frame shape
        * *cut*: in the case of a subframe centered too close to a frame border, the subframe is truncated so it doesn't go outside the frame. The sub-frame shape is not conserved
    :type mode: keyword
    :param fill_value: the fill value that is used if ``mode`` is set to fill
    :type fill_value: real or int
    :param by_ref: if True, the function will return a subframe of the frame that points to the corresponding slice of the frame. If False, the function returns a copy of it
    :type by_ref: bool
    :returns: a N-dim array of shape ``subframesize``
    :raises:
        * Exception, if the subframe selection is done by reference with ``mode`` set to "fill"
        * Exception, if the subframe selection is done by reference but data is not a nparray

    >>> import misc.funcs as funcs
    >>> funcs.subframe(np.array(np.matrix(np.arange(10)).T*np.arange(10)),
        center=[3.4, 1.7], subframesize=3, mode="fit")
    [[ 2  4  6]
     [ 3  6  9]
     [ 4  8 12]]

    .. note::
        This function calls :func:`find_boundaries`.
    """
    if not isinstance(data, np.ndarray) and by_ref: raise Exception, "Can only perform in place modifications if data is type numpy.ndarray"
    if mode.upper()=="FIT" or mode.upper()=="CUT":
        coord = find_boundaries(center, subframesize=subframesize, framesize=np.shape(data), mode=mode, integer=True)
        myslice = [slice(mind, maxd) for mind, maxd in coord]
    elif mode.upper()=="FILL":
        if by_ref: raise Exception, "Can't perform in place modifications for fill mode"
        dim = len(np.shape(data))
        subframesize = np.round(np.tile(subframesize, dim//np.size(subframesize)+1)[:dim]).astype(int) # if dim(subframesize)!=dim(center), repeats or crops
        ret = np.ones(subframesize)*fill_value
        coord = find_boundaries(center, subframesize=subframesize, framesize=np.shape(data), mode="cut", integer=True)
        myslice = [slice(mind, maxd) for mind, maxd in coord] # in the frame
        framebit = np.copy(np.asarray(data)[myslice])
        coord2 = find_boundaries(center, subframesize=subframesize, framesize=np.shape(data), mode="fit", integer=True)
        offsets = [coord2[i][1]-coord[i][1] for i in range(len(coord))]
        mysliceret = [slice(offsets[i], offsets[i]+framebit.shape[i]) for i in range(len(offsets))] # in the sub-frame
        ret[mysliceret] = framebit
        return ret
    else:
        raise Exception, "Keyword for mode unknown"
    if by_ref:
        return data[myslice]
    else:
        return np.copy(np.asarray(data)[myslice])


def blur_image(input, sigma, order=0, output=None, mode='reflect', cval=0.0):
    """
    Multi-dimensional Gaussian filter.

    :param input: input array to filter
    :param sigma: standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.
    :type sigma: scalar or sequence of scalars
    :param order: The order of the filter along each axis is given as a sequence of integers, or as a single number. An order of 0 corresponds to convolution with a Gaussian kernel. An order of 1, 2, or 3 corresponds to convolution with the first, second or third derivatives of a Gaussian. Higher order derivatives are not implemented
    :type order: 0, 1, 2, 3 or sequence from same set
    :param output: The output parameter passes an array in which to store the filter output
    :type output: array
    :param mode: The mode parameter determines how the array borders are handled, where ``cval`` is the value when ``mode`` is equal to 'constant'. Default is 'reflect'
    :type mode: 'reflect','constant','nearest','mirror', 'wrap'
    :param cval: Value to fill past edges of input if ``mode`` is 'constant'
    :type cval: scalar

    .. note::
        
        The multi-dimensional filter is implemented as a sequence of one-dimensional convolution filters. The intermediate arrays are stored in the same data type as the output. Therefore, for output types with a limited precision, the results may be imprecise because intermediate results may be stored with insufficient precision.
        
        This function is an alias of scipy.ndimage.gaussian_filter
    """
    return _blur_image(input=input, sigma=sigma, order=order, output=output, mode=mode, cval=cval)



def get_blob_centers(im, n=2, blob_radius=10, sigma_blurring=None, return_sigma=False, stop_if_error=False, silent=False):
    """Returns the coordinates of the n blobs of the image im after having performed a blur if specified

    TODO:
    - blob mode sub ou flat rond ou carré,
    - auto find size of blob from 3 px then sigma (or 9 px etc)
    - silent option
    - opt return cleaned frame
    - auto blur sub pixel"""
    coord = np.zeros([n,2])
    sigma = np.zeros(n)
    max_i = np.zeros(n)
    image = np.copy(im)
    replace_val = image.min()
    if sigma_blurring is not None: image = blur_image(image, sigma=sigma_blurring)
    for i in range(n):
        y_max = np.argmax(image)%image.shape[1]
        x_max = (np.argmax(image)-y_max)//image.shape[1]
        (xmin,xmax),(ymin,ymax) = find_boundaries((x_max, y_max), blob_radius*2, image.shape, mode="fit")
        if stop_if_error:
            x_max, x_sigma = funcs.gauss_fit(np.linspace(xmin,xmax,xmax-xmin+1)[:-1],np.sum(image[xmin:xmax,ymin:ymax],axis=1))[0][1:3]
            y_max, y_sigma = funcs.gauss_fit(np.linspace(ymin,ymax,ymax-ymin+1)[:-1],np.sum(image[xmin:xmax,ymin:ymax],axis=0))[0][1:3]
        else:
            try:
                x_max, x_sigma =  funcs.gauss_fit(np.linspace(xmin,xmax,xmax-xmin+1)[:-1],np.sum(image[xmin:xmax,ymin:ymax],axis=1))[0][1:3]
            except:
                print "could not find x-center/x-fwhm, set to 0"
                x_max, x_sigma=0.,0.
            try:
                y_max, y_sigma =  funcs.gauss_fit(np.linspace(ymin,ymax,ymax-ymin+1)[:-1],np.sum(image[xmin:xmax,ymin:ymax],axis=0))[0][1:3]
            except:
                print "could not find y-center/y-fwhm, set to 0"
                y_max, y_sigma=0.,0.
        sigma[i] = np.sqrt(x_sigma*y_sigma)
        coord[i,:] = np.array([x_max,y_max])
        if i==n-1: break
        subframe(image, center=(x_max, y_max), subframesize=blob_radius*2+1, mode="cut", by_ref=True)[:] = replace_val
    return coord
    if return_sigma:
        return coord[np.argsort(coord[:,0],axis=0)], sigma[np.argsort(coord[:,0],axis=0)] #returns coords of all blobs, with increasing x    
    else:
        return coord[np.argsort(coord[:,0],axis=0)] #returns coords of all blobs with increasing x



def disk_add2array(arr, x0, y0, radius, max_amplitude=1.):
    """
    Draws and adds a disk patterns to an input array.

    :param arr: The 2D array to which the disk will be added. If dim(``arr``)>2, only the 2 last dimensions will be considered
    :param x0-y0: The pixel coordinates of the disk center
    :type x0-y0: real, in pixels
    :param radius: The radius of the disk
    :type radius: real, in pixels
    :param max_amplitude: The maximum amplitude of the disk
    :type max_amplitude: real
    :returns: 2D numpy array -- the input array to which the disk was added. Values are real between 0 and ``max_amplitude``.

    .. note::

       The addition of the disk pattern to the input array is genuinely an addition of the disk pattern values, not a replacement. Overlapping patterns will give values greater than ``max_amplitude``
      
       If the radius is lower than 0.5 px, the function will just add max_amplitude at ``x0`` and ``y0`` coordinates

    >>> import misc.funcs as funcs
    >>> import matplotlib.pyplot as plt
    >>> myarray = funcs.disk_add2array(np.zeros([512,512]), 80, 250, 30)
    >>> plt.matshow(myarray)
    """
    height, width = arr.shape[:2] # only 2 last dimensions to init a 2D image
    if radius<0.5: # if radius tiny, just add a dot to the center pixel
        arr[int(round(x0)),int(round(y0))]+=max_amplitude
        return arr

    data = np.zeros((height, width, 4), dtype=np.uint8)

    surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, height, width)

    ctx = cairo.Context(surface)
    # draw circle + fill
    # 2*pi = 6.283185307179586
    ctx.arc(x0+0.5, y0+0.5, radius, 0, 6.283185307179586) # centers on the pixel coordinates x0, y0 and not the inter-pixel
    ctx.set_line_width(0)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.fill()
    ctx.stroke()
    if data.max()==0: raise Exception, "Nothing plotted (probably radius too small)"
    return arr+np.array(np.mean(data,axis=2)).astype(float)*max_amplitude/data.max()

