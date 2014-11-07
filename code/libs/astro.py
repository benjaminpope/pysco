# -*- coding: utf-8 -*-
# Written by Guillaume, 2014

import numpy as np
import libs.funcs as funcs
import time
import calendar
import angle
import ephem as E

# CONSTANTS
PC2M = 3.08567758e16 # multiply parsec with that and you get meter
M2PC = 1/PC2M
AU2M = 149597871000. # multiply AU with that and you get meter
M2AU = 1/AU2M
AU2PC = AU2M/PC2M # multiply AU with that and you get parsec
PC2AU = 1/AU2PC
JUPRADIUS = 69911000. # multiply Jupiter radius with that and you get meter [m]


def next_date(initdate, period, date_after=False):
    "ads 'period' to 'initdate' as many times as required to have the new date right before the current gm time (or right after if 'date_after' is True)"
    if np.size(initdate)==1: initdate = julian_to_calendar(initdate)
    if np.size(initdate)!=6: raise Exception, "initdate not understood, must be Julian date of [Y,M,D,H,M,S]"
    return cal2JD(time.gmtime(calendar.timegm(initdate)+int(int(date_after)+(time.time()-calendar.timegm(initdate))/(period*86400))*(period*86400))[:6])



def orbital_pos(nu, e, i, w, a=AU2M, o=0., degrees=False):
    """
    Calculates the orbital positions of a body with Keplerian orbit

    :param nu: true anomaly of the body (position on its orbit)
    :type nu: N-dim array
    :param e: eccentricity of the orbit, must be [0,1)
    :type e: real
    :param i: inclination of the orbit, must be [0,180). i>90 means retrograde body
    :type i: real, radian
    :param w: argument at periapsis of the orbit, should be [0,360)
    :type w: real, radian
    :param a: semi-major axis of the orbit
    :type a: real
    :param o: longitude of the ascending node of the orbit, should be [0,360)
    :type o: real, radian
    :param degrees: if True, ``i``, ``w``, ``o`` will be expected in degrees
    :type degrees: bool
    :returns: (radius, projected radius, north angle, phase angle), each having the same shape as ``nu``. radius and projected radius have the same unit as ``a``; north angle and phase angle are radians unless ``degrees`` is ``True``.
    :raises:
        * Exception, if e is outside [0,1)
        * Exception, if i is outside [0,180]

    >>> import misc.funcs as funcs
    >>> print funcs.orbital_pos([10,20,30], 0.1, 89.6, 56, 1, 90, degrees=True)
    (array([ 0.90124472,  0.90496144,  0.91109671]),
     array([ 0.36661431,  0.21901579,  0.06387085]),
     array([ 90.89833379,  91.60388027,  95.70132773]),
     array([ 24.00313585,  14.00559899,   4.01991791]))
    """
    if e>=1 or e<0: raise Exception, "Eccentricity must be [0,1)"
    if i>180 or i<0: raise Exception, "Inclination must be [0,180]"
    if degrees is True:
        conv = angle.DEG2RAD
    else: conv = 1.
    nu = np.asarray(nu)*conv
    radius = a*(1-e**2)/(1+e*np.cos(nu))
    x, y, z = funcs.euler_rot(radius*np.cos(nu), radius*np.sin(nu), np.zeros(nu.shape), o, i, w, degrees=degrees)
    radius_proj = np.sqrt(x**2+y**2)
    north_angle = np.arctan2(y, x)/conv
    return radius, radius_proj, north_angle, np.arccos(np.sin(nu+w*conv)*np.sin(i*conv))/conv



# converts true anomaly angles [deg] to orbital period fraction [% of period]
def nu_to_fr(nu, eccentricity, degrees=False):
    """
    input: nu N-dim, eccentricity [0-1); output: fr [% of period] (same shape as nu)
    """
    if degrees is True:
        conv = angle.DEG2RAD
    else: conv = 1.
    E = 2*np.arctan(np.sqrt((1-eccentricity)/(1+eccentricity))*np.tan(np.radians(nu*conv)/2.))
    return ((E-eccentricity*np.sin(E))/(2*np.pi)) #fraction de T ecoulee depuis le dernier periastre



# converts orbital period fraction [% of period] to true anomaly angles [deg], with a given precision
def fr_to_nu(fr, eccentricity, prec=1.e-10, degrees=False): #fr [0-1]
    """
    input: fr [% of period] (can be vector), eccentricity [0-1]; output: nu [deg]  (same vector-size as fr)
    """
    if degrees is True:
        conv = angle.DEG2RAD
    else: conv = 1.
    M = fr*2*np.pi
    E = M+eccentricity*np.sin(M)
    for i in range(100): # max loop of 100 in case it doesn't converge
        Ebis = M+eccentricity*np.sin(E)
        if np.max(np.abs(Ebis-E))<np.max(prec*E):
            break
        else: E = Ebis
    return (2*np.arctan(np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(Ebis/2.)))%(2*np.pi)/conv


def now():
    """
    :returns: the current UT time
    """
    return E.now()


def julian_to_calendar(julian):
    """
    Calculates the calendar date from a julian date.

    :param julian: the data to pad, can be 1 to 3 dimensions
    :type julian: real
    :returns: anti-padded data as ndarray

    >>> dvsef
    """
    if np.size(julian)>1: raise Exception, "julian nate not udnerstood, must be a float"
    jf=np.int(julian+0.5)+1401
    jf=jf+(((4*np.int(julian+0.5)+274277)/146097)*3)/4-38
    je=4*jf+3
    jg=(je%1461)/4
    jh=5*jg+2
    jm=((jh/153+2)%12)+1
    jt=(julian+0.5)%1
    return [je/1461-4716+(14-jm)/12,jm,(jh%153)/5+1,np.int(jt*24),np.int(jt*1440%60),np.int(jt*86400%60)]



def cal2JD(time=None):
    """

    """
    if time is None: return E.julian_date(now())
    if np.size(time)>1: return E.julian_date(tuple(time))
    return E.julian_date(time)
