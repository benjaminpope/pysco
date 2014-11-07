# -*- coding: utf-8 -*-
# Written by Guillaume, 2014

import re as _re
import numpy as _np
from matplotlib.pyplot import semilogy as _matplotlibpyplotsemilogy
from matplotlib.pyplot import plot as _matplotlibpyplotplot
from matplotlib.pyplot import figure as _matplotlibpyplotfigure
import matplotlib.cm as _cm
from matplotlib.colors import LinearSegmentedColormap as _matplotlibcolorsLinearSegmentedColormap
from matplotlib.pyplot import Normalize as _matplotlibpyplotNormalize
import time as _time
from multiprocessing import current_process as _multiprocessingcurrent_process
from sys import maxint as _sysmaxint
from scipy.interpolate import interp1d as _scipyinterpolateinterp1d
from scipy.io import readsav as _scipyioreadsav
from scipy.optimize import curve_fit as _scipyoptimizecurve_fit
from scipy.integrate import cumtrapz as _scipyintegratecumtrapz
from multiprocessing import Pool as _multiprocessingPool
from subprocess import Popen as _subprocessPopen
from subprocess import PIPE as _subprocessPIPE
from subprocess import STDOUT as _subprocessSTDOUT



#def var_exist(var):
#    return var in globals() # doit faire le test dans l'espace de la fonction d'appel, mettre en parametre le niveau de l'appel



#def var_test(var, cond):
#    if not var_exist(var): return False
#    if cond is None: return globals()[var] is None # doit chercher la valeur dans l'espace de la fonction d'appel
#    return globals()[var]==cond # doit chercher la valeur dans l'espace de la fonction d'appel



_colormap_looping = {  'red'  :  (  (0., 0., 0.),
                                    (1/6., 1., 1.),
                                    (2/6., 1., 1.),
                                    (3/6., 0., 0.),
                                    (4/6., 0., 0.),
                                    (5/6., 0., 0.),
                                    (1., 0., 0.)),
                      'green':  (   (0., 0., 0.),
                                    (1/6., 0., 0.),
                                    (2/6., 1., 1.),
                                    (3/6., 1., 1.),
                                    (4/6., 1., 1.),
                                    (5/6., 0., 0.),
                                    (1., 0., 0.)),
                      'blue' :  (   (0., 0., 0.),
                                    (1/6., 0., 0.),
                                    (2/6., 0., 0.),
                                    (3/6., 0., 0.),
                                    (4/6., 1., 1.),
                                    (5/6., 1., 1.),
                                    (1., 0., 0.))}
cm_looping = _matplotlibcolorsLinearSegmentedColormap('looping', _colormap_looping, 1024)
_additional_cm={'looping':cm_looping}

_colormap_bwhite = { 'red'  :  (   (0., 0., 0.9),
                        (0.6, 0.45, 0.45),
                        (1., 0., 0.)),
          'green':  (   (0., 0., 0.9),
                        (0.6, 0.6, 0.6),
                        (1., 0., 0.)),
          'blue' :  (   (0., 0., 0.9),
                        (0.6, 0.8, 0.8),
                        (1., 0.05, 0.))}
cm_bwhite = _matplotlibcolorsLinearSegmentedColormap('bwhite', _colormap_bwhite, 1024)
_additional_cm.update({'cm_bwhite':cm_bwhite})



def colorbar(cmap="jet", cm_min=0, cm_max=1):
    if cmap in _additional_cm.keys():
        cmap = _additional_cm[cmap]
    elif isinstance(cmap, str):
        cmap = _cm.get_cmap(cmap)
    norm = _matplotlibpyplotNormalize(cm_min, cm_max)
    mappable = _cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable._A = []
    return cmap, norm, mappable


def setval(cond, iftrue, iffalse, toobject=None, attr=""):
    """
    Returns (or adds) a value (or an attribute to an object) depending on a condition
    
    :param cond: the condition. If ``cond`` is not already a boolean, then ``cond`` is checked to be None: ``cond = cond is not None``
    :type cond: bool or None
    :param iftrue: the value that will be returned or added to the object if ``cond`` is ``True``
    :type iftrue: any
    :param iffalse: the value that will be returned or added to the object if ``cond`` is ``False``
    :type iffalse: any
    :param toobject: if not None, setval will not return a value, but add it to this object
    :type toobject: non-write protected object
    :param attr: the name of the attribute under which the value will be added to ``toobject``. Ignored if ``toobject`` is None
    :type attr: string
    :returns: ``iftrue`` value if ``cond`` is ``True`` or not ``None``, ``iffalse`` value if ``cond`` is ``False`` or ``None``, nothing if the value is added to ``toobject``

    >>> import misc.funcs as funcs
    >>> a = 12
    >>> print funcs.setval(a>10, "Hello", "Goodbye")
    Hello
    """
    if not isinstance(cond, bool): cond = cond is not None
    if toobject is None:
        if cond:
            return iftrue
        else:
            return iffalse
    else:
        if attr=="": raise Exception, "Attribute name cannot be null"
        if cond:
            setattr(toobject, str(attr), iftrue)
        else:
            setattr(toobject, str(attr), iffalse)


def replace_multi(text, reps, ignore_case=False):
    """
    Return the string obtained by replacing the leftmost non-overlapping occurrences of pattern in string by the replacement repl.
    """
    if ignore_case:
        rep = dict((_re.escape(k).lower(), v) for k, v in reps.iteritems())
        pattern = _re.compile("|".join(rep.keys()), _re.IGNORECASE)
        return pattern.sub(lambda m: rep[_re.escape(m.group(0)).lower()], text)
    else:
        rep = dict((_re.escape(k), v) for k, v in reps.iteritems())
        pattern = _re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[_re.escape(m.group(0))], text)



def rebin_2D_smart(x,y,data, bin_x, bin_y):
    x1, y1, w1 = rebin_2D(x, y, data, bin_x, bin_y, bin_x/3., bin_y/3.)
    x2, y2, w2 = rebin_2D(x, y, data, bin_x, bin_y, -bin_x/3., bin_y/3.)
    x3, y3, w3 = rebin_2D(x, y, data, bin_x, bin_y, -bin_x/3., -bin_y/3.)
    x4, y4, w4 = rebin_2D(x, y, data, bin_x, bin_y, bin_x/3., -bin_y/3.)
    x=_np.concatenate((x1,x2,x3,x4))
    y=_np.concatenate((y1,y2,y3,y4))
    w=_np.concatenate((w1,w2,w3,w4))
    return rebin_2D(x, y, w, bin_x, bin_y)



def rebin_2D(x, y, data, bin_x, bin_y, offset_x=0., offset_y=0., halfway=False):
    xx = _np.round((x+offset_x) / bin_x).astype(int)
    yy = _np.round((y+offset_y) / bin_y).astype(int)
    shape = [xx.max()+1, yy.max()+1]
    bins = _np.ravel_multi_index([xx, yy], shape)
    bincts = _np.bincount(bins)
    bincts[bincts==0] = -1
    x_mean = _np.bincount(bins, x) / bincts
    y_mean = _np.bincount(bins, y) / bincts
    if halfway: return x_mean[bincts!=-1], y_mean[bincts!=-1], bins, bincts
    data_mean = _np.bincount(bins, data) / bincts
    return x_mean[bincts!=-1], y_mean[bincts!=-1], data_mean[bincts!=-1]



def cmd_line(command, ret=True, clean=True):
    """
    Executes a command line in shell environment
    
    :param command: the command to execute
    :type command: string
    :param ret: if ``True``, the function will wait until ``command`` is executed. Else, the python script will go on while ``command`` is executed in the background
    :type ret: bool.
    :param clean: if ``True``, the 'empty lines' and the 'end of line characters' will be deleted from the output list
    :type clean: bool.
    :returns: ``None`` if ``ret`` is ``False``, or if it is ``True`` a list of the standard output lines given back by ``command``

    >>> import misc.funcs as funcs
    >>> print funcs.cmd_line('sleep 1')
    [] (after 1 sec wait)
    >>> print funcs.cmd_line('sleep 1', False)
    will output nothing and jump to the next python instruction
    >>> print funcs.cmd_line('sleep 1;ls *.dum', True, False)
    ['dummy.dum\\n', 'dummy2.dum\\n'] (after 1 sec wait)
    """
    res = _subprocessPopen(command, shell=True, stderr=_subprocessSTDOUT, stdout=_subprocessPIPE)
    if ret is not True: return None
    res = res.stdout.readlines()
    if clean is not True: return res
    return [i.strip() for i in res if i!=""]



def check_str(text, rem_char=None, auth_char=None, rem_words=None, auth_words=None, ignore_case=False):
    if rem_words is not None:
        if isinstance(rem_words, str): rem_words = list([rem_words])
        if not isinstance(rem_words, list): raise Exception, "rem_words argument must be a list"
        reps={}
        for elmt in rem_words:
            reps.update({elmt:""})
        text = replace_multi(text, reps, ignore_case=ignore_case)
    if rem_char is not None:
        if not isinstance(rem_char, str): raise Exception, "rem_char argument must be a string"
        reps={}
        for elmt in rem_char:
            reps.update({elmt:""})
        text = replace_multi(text, reps, ignore_case=ignore_case)
    return text
    #if authchar is not None: authchar = authchar.lower()
    #if rem_char is not None: rem_char = rem_char.lower()



def timestamp_name(filename, ext=None, gm=True, fmt="%Y%m%dT%H%M%S", rem_char="sp", auth_char=None):
    if ext is not None: # if ext provided
        ext = str(ext)
        if ext[0] != ".": ext = "." + ext
        if filename.find('.') != -1: filename = filename[:filename.rfind('.')]  # if point in filename remove extension
    else: #
        ext = ""
        if filename.find('.') != -1:
            ext = filename[filename.rfind('.'):]  # if point in filename copies extension including point
            filename = filename[:filename.rfind('.')]
    return filename + "_" + timestamp(gm=gm, fmt=fmt) + ext



def timestamp(gm=True, fmt="%Y%m%dT%H%M%S"):
    if gm:
        return _time.strftime(fmt, _time.gmtime())
    else:
        return _time.strftime(fmt, _time.localtime())



def unique_timeid(unicity=None, freq=10, len_output=None):
    """
    Generates a short unique ID based on the UT timestamp
    
    :param unicity: the duration over which the unique ID will be unique. Leave ``None`` for truly unique ID
    :type unicity: real, in years or None
    :param freq: the unique ID update frequency. Systems usually provide timestamps down to the micro-second (freq=1e6)
    :type freq: real, in Hz
    :param len_output: the length of the string output, which will be padded with 0 if necessary. Leave ``None`` for no padding.
    :type len_output: int or None
    :returns: a string representing the unique ID based on the UT timestamp

    >>> import misc.funcs as funcs
    >>> print funcs.unique_timeid(10, 1)
    1zxtk

    .. note::
        
        The alphabet used for the string generation is ``abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789``.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    lenalphabet = len(alphabet)
    yeartosec = 31556736.
    if unicity_years is None: unicity_years=10000
    num = int(((_time.time()/yeartosec)%unicity_years)*yeartosec*freq_sec)
    ret=''
    for i in range(int(np.log(num)/np.log(lenalphabet))+1):
        ret += alphabet[num%lenalphabet]
        num = num//lenalphabet
    if len_output is None: return ret[::-1]
    return ret[::-1].zfill(int(len_output))



def cat_cond(cond, iftrue="", iffalse=""):
    """
    Returns iftrue as string is cond is True or not None, or iffalse as string if cond if false or None
    """
    return setval(cond=cond, iftrue=str(iftrue), iffalse=str(iffalse))



class idlvar_read():
    """
    Reads an IDLVAR file and converts it to a python object
    
    :param filename: (path+)filename to the IDLVAR file
    :type filename: string
    :returns: an object whose attributes are the different variables found in the IDLVAR file

    >>> import misc.funcs as funcs
    >>> myvar = funcs.idlvar('/folder/folder/file.idlvar')
    >>> myvar. <tab>
    This will display all variables stored in the IDLVAR file
    """
    def __init__(self, filename):
        dic = _scipyioreadsav(filename)
        for i in dic:
            setattr(self, i, dic[i])



def aslist(data, numpy=False, integer=False):
    """
    Transforms data into a 1 dimensional list

    :param data: the data to transform into a list, can be any dimension and size
    :param numpy: returns the list as nparray
    :type numpy: bool
    :param integer: returns the list as integer
    :type integer: bool
    :returns: a 1 dimensional list of ``data``

    >>> import misc.funcs as funcs
    >>> print funcs.aslist(1)
    [1]
    >>> print funcs.aslist([(1.0,4.0),(2.0,3.0)], numpy=True, integer=True)
    [1, 4, 2, 3]
    """
    if _np.iterable(data)==False:
        if integer:
            ret = _np.asarray([int(data)])
        else: ret = _np.asarray([data])
    else:
        if integer:
            ret = _np.asarray(data).flatten().astype(int)
        else: ret = _np.asarray(data).flatten()
    if not numpy:
        return list(ret)
    return ret



def pad_array(data, fill_value=0, axes=None, newshape=None):
    """
    Pads data to the double of its size (unless specified) in each specified axis

    :param data: the N-dim array to pad
    :param fill_value: the value to fill the padding with
    :type fill_value: real
    :param axes: the axes along which to perform anti-padding, default is all axes
    :type axes: int or tuple of int
    :param newshape:
        * if set to "fft", the data will be padded to the next power of 2 along each specified axis
        * if array.shape-like, ``axes`` is ignored and data returned as the center of a an array with shape ``newshape``
    :type newshape: "fft", int or tuple of int
    :returns: padded data as ndarray
    :raises: Exception, if dimension of ``newshape`` does not agree with data.shape

    >>> import misc.funcs as funcs
    >>> print funcs.pad_array([1,1])
    [ 0.   1.   1.   0. ]
    >>> print funcs.pad_array([[1,1],[2,2]],3, axes=-1)
    [[ 3.   1.   1.   3. ]
     [ 3.   2.   2.   3. ]]

    .. note::
        
        If ``newshape`` is set to "fft", all specified axes will be returned with the same size, which will be determined by the greatest next power of 2
        
        If ``newshape`` is not specified, the size of each specified axis will be multiplied by 2

    >>> print funcs.pad_array([[1,2]], 0, axes=(1), newshape="fft")
    [[ 0.  1.  2.  0.]]
    >>> print funcs.pad_array([[1,2]], 0, axes=(0,1), newshape="fft")
    [[ 0.  0.  0.  0.]
     [ 0.  1.  2.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    """
    data = _np.asarray(data)
    if axes is None: axes = _np.arange(len(data.shape)).astype(int)
    make_fft = False
    if str(newshape).upper().find("FFT")!=-1: make_fft = True
    if newshape is not None and make_fft is False:
        if _np.size(newshape) != _np.size(data.shape): raise Exception, "newshape must agree with data shape"
        newshape = _np.asarray(newshape)
    else:
        newshape = _np.asarray(data.shape)
        if make_fft is True:
            indexes = aslist(axes, numpy=True, integer=True)
            newshape[indexes] = nextp2(newshape[indexes]+1).max()
        else:
            for i in aslist(axes):
                newshape[i] *= 2
    currsize = _np.asarray(data.shape)
    startind = (newshape - currsize) // 2
    endind = startind + currsize
    new_im = _np.ones(newshape)*fill_value
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    new_im[tuple(myslice)] = data
    return new_im



def padrev_array(data, axes=None, newshape=None):
    """
    Anti-pads data to the half of its size in each specified axis, or to newshape is specified

    :param data: the N-dim array to unpad
    :param axes: the axes along which to perform anti-padding, default is all axes
    :type axes: int or tuple of int
    :param newshape: if not None, ``axes`` is ignored and the inner region of data is returned, with shape ``newshape``
    :type newshape: int or tuple of int
    :returns: unpadded data as ndarray

    >>> import misc.funcs as funcs
    >>> print funcs.padrev_data([0,1,1,0])
    [ 1.   1. ]
    >>> print funcs.padrev_data([[0,1,1,0],[0,2,2,0]], axes=1)
    [[1 1]
     [2 2]]
    >>> print funcs.padrev_data([[0,1,1,0],[0,2,2,0]], newshape=(2,2))
    [[1 1]
     [2 2]]

    .. note::
        
        This function calls :func:`center_array`.
        
        If ``newshape`` is not specified, the size of each specified axis will be divided by 2
    """
    if axes is None: axes = _np.arange(len(data.shape)).astype(int)
    if newshape is None:
        newshape = _np.asarray(_np.shape(data))
        for i in aslist(axes):
            newshape[i] = newshape[i]//2
    return center_array(data, newshape)



def center_array(data, newshape):
    """
    Extracts the central part of an array

    :param data: the N-dim array to process
    :param newshape: the shape of the final subarray to extract
    :type newshape: int or tuple of int
    :returns: central portion of the array, with shape ``newshape``

    >>> import misc.funcs as funcs
    >>> print funcs.center_array([0,1,1,0], 2)
    [1 1]
    >>> print funcs.center_array([[0,1,1,0],[0,2,2,0],[0,3,3,0]], (1,2))
    [[2 2]]
    """
    data = _np.asarray(data)
    newshape = _np.asarray(newshape)
    currsize = _np.array(data.shape)
    startind = (currsize - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return data[tuple(myslice)]



def bin_array(data, binning, mode=_np.sum):
    """
    Groups together adjacent cells of an array

    :param data: the N-dim array to process
    :param binning: the shape of the N-dim cude that defines adjacent cells
    :type binning: int or tuple of int
    :param mode: the function to apply on the adjacent cells
    :type mode: callable function
    :returns: the data, where adjacent cells in a cube of shape ``binning`` got collapsed together following ``mode`` method. The shape of the new data is _np.asarray(data.shape)/_np.asarray(binning)
    :raises: Exception, if binning ``data``.shape is not a multiple of ``binning``, or if their size disagree

    >>> import misc.funcs as funcs
    >>> vector = _np.array(_np.matrix(_np.arange(4)).T*_np.arange(6))
    >>> data=_np.array([vector, vector+1])
    >>> print data
    [[[ 0  0  0  0  0  0]
      [ 0  1  2  3  4  5]
      [ 0  2  4  6  8 10]
      [ 0  3  6  9 12 15]]
     [[ 1  1  1  1  1  1]
      [ 1  2  3  4  5  6]
      [ 1  3  5  7  9 11]
      [ 1  4  7 10 13 16]]]
    >>> print data.shape
    (2, 4, 6)
    >>> binned_data = funcs.bin_array(data, (2, 2, 1))
    >>> print binned_data
    [[[ 2  4  6  8 10 12]
      [ 2 12 22 32 42 52]]]
    >>> print binned_data.shape
    (1, 2, 6)

    .. note::
        
        ``mode`` must be a callable function that collapses 2D data along its axis 1 to a 1D data.
        
        This callable function must take:
            * some 2D data as first parameter
            * a parameter ``axis`` that will be set to 1
        
        You may use _np.sum, _np.mean, _np.median, _np.max, _np.min, ... functions.
    """
    data = _np.asarray(data)
    datashape = _np.asarray(data.shape).astype(int)
    binning = aslist(binning, numpy=True, integer=True)
    if (datashape%binning).sum()!=0 or binning.size!=datashape.size: raise Exception, "Can't bin, data shape disagrees with binning parameter"
    for i in range(len(datashape)):
        datashape[i] = datashape[i] / binning[i]
        temp_size = datashape.copy()
        temp_size[i] = datashape[-1]
        temp_size[-1] = datashape[i]
        data = _np.reshape(mode(_np.reshape(data.swapaxes(i,-1), (-1, binning[i])), axis=1), temp_size).swapaxes(i,-1)
    return data


def euler_rot(x, y, z, z1, x2, z3, degrees=False):
    """
    Calculates the Euler rotation (z1, x2, z3) of the input coordinates

    :param x-y-z: the cooridnates to rotate
    :type x-y-z: N-dim array
    :param z1-x2-z3: the Euler angles 
    :type z1-x2-z3: real, radian
    :param degrees: if True, the Euler angles ``z1``, ``x2``, ``z3`` will be expected in degrees
    :type degrees: bool
    :returns: (x', y', z'), the Euler angle rotated values of ``(x, y, z)``

    >>> import misc.funcs as funcs
    >>> print funcs.euler_rot(1, 2, 3, 0.1, 0.2, -0.6)
    (1.898096374053891, 0.66116903171326336, 3.1559603397867377)
    """
    x = _np.asarray(x)
    y = _np.asarray(y)
    z = _np.asarray(z)
    if degrees is True:
        conv = _np.pi/180.
    else: conv = 1.
    ca1 = _np.cos(z1*conv)
    ca2 = _np.cos(x2*conv)
    ca3 = _np.cos(z3*conv)
    sa1 = _np.sin(z1*conv)
    sa2 = _np.sin(x2*conv)
    sa3 = _np.sin(z3*conv)
    rot = _np.array([[ca1*ca3-ca2*sa1*sa3, ca3*sa1+ca1*ca2*sa3, sa2*sa3],
                     [-ca1*sa3-ca2*ca3*sa1, ca1*ca2*ca3-sa1*sa3, ca3*sa2],
                     [sa1*sa2, -ca1*sa2, ca2]])
    return x * rot[0,0] + y * rot[1,0] + z * rot[2,0], x * rot[0,1] + y * rot[1,1] + z * rot[2,1], x * rot[0,2] + y * rot[1,2] + z * rot[2,2]



def gen_seed():
    """
    Generates a seed based on the micro second time and the process id. This makes it multi-threading safe.

    :returns: an integer seed

    >>> import misc.funcs as funcs
    >>> print funcs.gen_seeed()
    1825296981
    """
    return int(long(str((_time.time()%3600)/3600)[2:])*_multiprocessingcurrent_process().pid%_sysmaxint)



def gen_generator(seed=None):
    """
    Generates a random generator with a seed based on the micro second time and the process id. This makes it multi-threading safe.

    :returns: a random number generator object

    >>> import misc.funcs as funcs
    >>> rnd = funcs.gen_generator()
    >>> print rnd.uniform()
    0.472645

    .. note::
        This function calls :func:`gen_seed` if seed is not given.
    """
    if seed is None:
        return _np.random.RandomState(gen_seed())
    else:
        return _np.random.RandomState(seed)



def random_custom_pdf(x, pdf, size=1, renorm=False):
    """
    Generates random numbers (or the generating function) following a custom pdf

    :param x: the x data relative to the pdf
    :type x: array
    :param pdf: the pdf, must be the same size as x
    :type pdf: array
    :param size: the size of the output:
        * if False, this fuction returns a function
        * if int, this function returns an array of size ``size``
        * if array, this function returns an array whose shape is ``size``
    :type size: bool, int or tuple of int
    :param renorm: forces first and last element of the cdf (obtained from the pdf) to be 0 and 1
    :type renorm: bool
    :returns: randomly generated numbers (or the generating function) following the pdf distribution
    :raises: Exception, if any value of the pdf is negative

    >>> import misc.funcs as funcs
    >>> x = _np.r_[_np.linspace(0,0.5,100), _np.linspace(0.5,1,101)[1:]]
    >>> y = _np.r_[_np.linspace(0,0.5,100), 0.5-_np.linspace(0,0.5,100)]
    >>> print funcs.random_custom_pdf(x, y, size=(3,2), renorm=True)
    [[ 0.48273149  0.76731952]
     [ 0.8438356   0.32353295]
     [ 0.44305302  0.5361907 ]]

    .. note::
        This function calls :func:`random_custom_cdf`, which calls :func:`gen_generator`.
    """
    if (_np.asarray(pdf)<0).any(): raise Exception, "Can't compute pdf with negative values" # if any negative value in pdf, raises error
    cdf = integrate_array(x, pdf)
    return random_custom_cdf(x, cdf, size=size, renorm=renorm)



def random_custom_cdf(x, cdf, size=1, renorm=False):
    """
    Generates random numbers (or the generating function) following a custom cdf

    :param x: the x data relative to the pdf
    :type x: array
    :param cdf: the cdf, must be the same size as x
    :type cdf: array
    :param size: the size of the output:
        * if False, this fuction returns a function
        * if int, this function returns an array of size ``size``
        * if array, this function returns an array whose shape is ``size``
    :type size: bool, int, tupple
    :param renorm: forces first and last element of the cdf (obtained from the pdf) to be 0 and 1
    :type renorm: bool
    :returns: randomly generated numbers (or the generating function) following the pdf distribution
    :raises:
        * Exception, if cdf is not monotonic increasing
        * Exception, if first and last element of cdf are not 0 and 1 (and renorm is False)

    >>> import misc.funcs as funcs
    >>> x = _np.r_[_np.linspace(0,0.5,100), _np.linspace(0.5,1,101)[1:]]
    >>> y = _np.r_[_np.linspace(0,0.5,100), 0.5-_np.linspace(0,0.5,100)]
    >>> y = integrate_array(x, y)
    >>> print func.random_custom_cdf(x, y, size=(3,2), renorm=True)
    [[ 0.34601846  0.57638203]
     [ 0.46625474  0.52558282]
     [ 0.28020879  0.14224155]]

    .. note::
        This function calls :func:`gen_generator`.
    """
    cdf = _np.array(cdf)
    if (_np.diff(cdf)<0).any(): raise Exception, "cdf must be monotonic increasing"
    if not renorm:
        if _np.round(cdf[-1],10)!=1 or _np.round(cdf[0],10)!=0: raise Exception, "Wrong cdf distribution (first element not 0 or last element not 1)" # if first or last element not close enough from 0 and 1, rejects
    cdf = (cdf-cdf[0])/(cdf[-1]-cdf[0]) # forces first and last elements to be 0 and 1 exactly
    rnd = gen_generator()
    if size is False: return _scipyinterpolateinterp1d(cdf, x, kind='linear')
    if size==1: return float(_scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=1)))
    if _np.size(size)==1: return _scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=size))
    return _scipyinterpolateinterp1d(cdf, x, kind='linear')(rnd.uniform(size=_np.prod(size)).reshape(size))



def array_to_binsize(arr):
    """
    TBD

    Give the bin size between elements of arr so that the whole range from min(arr) to max(arr) is filled.
    the i-th binsize = (arr[i+1]-arr[i-1])/2

    >>> import misc.funcs as funcs
    >>> print funcs.array_to_binsize([1,4,5,7])
    [ 3.   2.   1.5  2. ]
    """
    arr = _np.array(arr)
    if _np.size(arr)==1: raise Exception, "Size must be greater than one"
    if _np.size(_np.shape(arr))>1: raise Exception, "Dimension must be one"
    return _np.r_[arr[1]-arr[0], (arr[2:]-arr[:-2])/2., arr[-1]-arr[-2]]



def array_to_bins(arr):
    """
    TBD

    :param data: the data to pad, can be 1 to 3 dimensions
    :type data: array
    :returns: anti-padded data as ndarray

    >>> import misc.funcs as funcs
    >>> print func.array_to_bins([1,4,5,7])
    [-0.5  2.5  4.5  6.   8. ]
    """
    binsize = array_to_binsize(arr)
    return _np.r_[1-binsize[0]/2.,1-binsize[0]/2.+_np.cumsum(binsize)]



def percentile_view(data, bins=None, log=False, plot=None):
    """
    Calculates the percentile of a data and plots the result if required

    :param data: the data, any dimension possible as it will be flattened
    :type data: bool
    :param bins: the number of percentile values to be calculated, they are linearly spaced between 0 to 100 (included). If ``bins`` is an array, the percentile values are calculated for each of these values
    :type bins: int or array
    :param log: if true, shows the plot with log scale on y
    :type log: bool
    :param plot: the number of the figure on which the curve is (over)plotted; if False, the function returns a vector; else or None, a new figure is created
    :type plot: int or False
    :returns: a vector of the calculated percentile if ``plot`` is False, otherwise None

    >>> import misc.funcs as funcs
    >>> print funcs.percentile_view(_np.random.random(1000),[25,50,75])
    [0.23487384236763059, 0.48188344917304932, 0.76106477406241557]
    """
    if bins is None: bins = _np.r_[_np.linspace(0,3,10)[:-1],_np.linspace(3,97,50)[:-1],_np.linspace(97,100,10)[1:]]
    if _np.size(bins)==1: bins = _np.linspace(0,100,bins+1)
    data = _np.percentile(data.flatten(), list(bins))
    if plot is None:
        f=_matplotlibpyplotfigure()
    elif isinstance(plot,int) and plot is not False:
        f=_matplotlibpyplotfigure(plot)
    else:
        return data
    _matplotlibpyplotplot(bins, data)
    if log: _matplotlibpyplotsemilogy()
    f.gca().set_xlim([min(bins)-1, max(bins)+1])



def integrate_array(x, y, axis=0):
    """
    Calculates the trapezoidal integral Y(x) of y(x) such as Y[i] = integral of y from x[0] to x[i], along one specific axis of y if it is multi-dimensional

    :param x-y: arrays that define a function f such as y=f(x). x and y along ``axis`` axis must have the same dimension
    :type x-y: real
    :param axis: if y is multi-dimensional, it tells the axis along which the integral must be carried
    :type axis: int
    :returns: Y(x), a numpy array having the same dimention as y

    >>> import misc.funcs as funcs
    >>> y = _np.tile(_np.array(_np.matrix(_np.ones(3)).T*_np.arange(4)), (2,1,1))
    >>> print y
    [[[ 0.  1.  2.  3.]
      [ 0.  1.  2.  3.]
      [ 0.  1.  2.  3.]]
     [[ 0.  1.  2.  3.]
      [ 0.  1.  2.  3.]
      [ 0.  1.  2.  3.]]]
    >>> print y.shape
    (2, 3, 4)
    >>> print funcs.integrate_array(_np.arange(4), y, axis=2)
    [[[ 0.   0.5  2.   4.5]
      [ 0.   0.5  2.   4.5]
      [ 0.   0.5  2.   4.5]]
     [[ 0.   0.5  2.   4.5]
      [ 0.   0.5  2.   4.5]
      [ 0.   0.5  2.   4.5]]]
    """
    y = _np.array(y).swapaxes(axis,-1)
    inty = _np.zeros(y.shape)
    myslice = [slice(0,i) for i in aslist(y.shape)[:-1]]+[slice(1,y.shape[-1])]
    inty[myslice] = _scipyintegratecumtrapz(y, x)
    return _np.array(inty).swapaxes(axis,-1)



def apodizer(sizeframe, no_power_radius, full_power_radius=None, order=2., no_power_limit=0.01, full_power_limit=0.99):
    if full_power_radius is None:
        t=_np.array([gauss_sup(sizeframe/2+no_power_radius,1.,sizeframe/2.,sigma,0,order) for sigma in _np.arange(1,sizeframe/2.,1)])
        sigma=_np.argmax(t[t<no_power_limit])+1
    else:
        order=_np.log(_np.log(no_power_limit)/_np.log(full_power_limit))*1./_np.log(1.*no_power_radius/full_power_radius)
        sigma=no_power_radius*1./_np.sqrt(2)*1./(-_np.log(no_power_limit))**(1./order)
    return gauss2D_sup(_np.arange(sizeframe),_np.arange(sizeframe),1.,sizeframe/2.,sizeframe/2.,sigma,0.,order)



def gauss2D_sup(x, y, a=1., x0=0., y0=0., sigma=1., foot=0., n=2):
    Y,X=_np.meshgrid(x*1.,y*1.)
    return a*_np.exp(-((X-x0)**2+(Y-y0)**2)**(n/2.)/(_np.sqrt(2)*sigma)**n)+foot



def gauss2D(x, y, a=1., x0=0., y0=0., sigma=1., foot=0.):
    return gauss2D_sup(x, y, a=a, x0=x0, y0=y0, sigma=sigma, foot=foot, n=2)



def gauss_sup(x, a=1., x0=0., sigma=1., foot=0., n=2):
    return foot+a*_np.exp(-(_np.abs(x*1.-x0)/(_np.sqrt(2)*sigma))**n)



def gauss(x, a=1., x0=0., sigma=1., foot=0.):
    return gauss_sup(x, a=a, x0=x0, sigma=sigma, foot=foot, n=2)



def gauss_fit(x, y, a=None, x0=None, sigma=None, foot=None):
    if foot is None: foot=_np.min(y)
    if a is None: a=_np.max(y)-foot
    if x0 is None: x0 = _np.sum(x*y)/_np.sum(y)
    if sigma is None: sigma = _np.std(x-x0)
    popt,pcov = _scipyoptimizecurve_fit(gauss,x,y,p0=[a,x0,sigma,foot])
    return popt,pcov



def shiftmicro_1d(data, delta):
    data = _np.asarray(data)
    if len(data.shape)>1:
        return _np.array([_np.interp(_np.arange(mpi.shape[0])-delta, _np.arange(mpi.shape[0]), mpi) for mpi in data])
    else:
        return _np.interp(_np.arange(data.shape[0])-delta, _np.arange(data.shape[0]), data)



def shiftmicro_2d(data, dx, dy):
    return shiftmicro_1d((shiftmicro_1d(data, dx)).T, dy).T


def resample_1d(data, pts, norm=True):
    data = _np.asarray(data)
    if _np.size(pts)==1: pts = _np.linspace(0, data.T.shape[0]-1, pts)
    if len(data.shape)>1:
        res = _np.array([_np.interp(pts, _np.arange(mpi.shape[0]), mpi) for mpi in data])
    else:
        res = _np.interp(pts, _np.arange(data.shape[0]), data)
    if norm:
        return res/res.sum()*data.sum()#(data-data.min()).sum())+data.min()
    else:
        return res



def resample_2d(data, shape, norm=True):
    return resample_1d(resample_1d(data, shape[0], norm).T, shape[1], norm).T



def nextp2(data):
    return 2**_np.ceil(_np.log2(data))



def nextpn(data, power):
    if power==1 or power<=0: raise Exception, "Invalid power"
    return power**_np.ceil(_np.log(data)/_np.log(power))



