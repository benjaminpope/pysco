# -*- coding: utf-8 -*-
# Written by Guillaume, 2014
import numpy as np
import libs.funcs as funcs
import scipy.interpolate
import os


class c_filter:
    def __init__(self, band, data):
        self.band=band
        self.mean_wl=data[0]
        self.delta_wl=data[1]
        self.flux_jansky=data[2]
        self.start_wl=data[0]-data[1]/2.
        self.end_wl=data[0]+data[1]/2
        self.flux_wm2m=data[3]
    def __repr__(self):
        output='\nBand -> '+str(self.__dict__['band'])+"\n"
        for i in self.__dict__:
            if i!='band': output += "%s -> %e\n" % (i, self.__dict__[i])
        return output+'\n'
    def __str__(self):
        output='\nBand -> '+str(self.__dict__['band'])+"\n"
        for i in self.__dict__:
            if i!='band': output += "%s -> %e\n" % (i, self.__dict__[i])
        return output+'\n'


def import_bands(filename='data/UBVRIJHK.txt', skiprows=2):
    """
    Reads filename skiping the skiprows first rows
    Returns a dictionnary of filters

    EXAMPLE:
    filters=import_bands('UBVRIJHK.txt', 2)
    filters['V'] <enter> shows all properties
    """
    if filename=='data/UBVRIJHK.txt':
        this_dir, this_filename = os.path.split(__file__)
        filename = os.path.join(this_dir, "data", "UBVRIJHK.txt")
    try:
        bands = np.loadtxt(filename, usecols=(0,), dtype=str)
        data = np.loadtxt(filename, usecols=(1,2,3,4))
    except:
        raise Exception, "Could not find/parse the filter file properly: %s" % filename
    filters={}
    for i in range(len(bands)):
        filters.update({bands[i]:c_filter(bands[i], data[i])})
    return filters


def blackbody_spectral_irr(teff, wl):
    """
    Give an effective temerature
    Returns the emitted (at surface!) flux in W/m2/m assuming black body behaviour

    UNITS:
    teff: K
    wl: m - can be vector
    output: W/m2/m

    EXAMPLE:
    >>> blackbody_spectral_irr(5778, np.linspace(550-44, 550+44)*e-9)
    """
    # pi*2*h*c**2 = 3.741771524664128e-16
    # h*c/kb = 0.014387769599838155
    return 3.741771524664128e-16/wl**5*1/(np.exp(0.014387769599838155/(wl*teff))-1)



def magref2bbflux(ref_mag, ref_band, band_or_wl, teff, photometry_file='data/UBVRIJHK.txt', unit='W'):
    """
    Give a reference magnitude, a reference band and wavelength or band for which
    you want to know the flux given an effective temperature of the star and the
    black body assumption.
    Returns a magnitude in the band_or_wl band, or a flux in W/m2/m for each of
    the wavelength in band_or_wl.

    UNITS:
    ref_mag/ref_band: U, B, V, R, I, J, H, K
    band_or_wl: band or meter - can be a vector
    teff: K
    output: depending on unit
    'Phm': Ph/s/m2/m (m being the walength grain)
    'Wm': W/m2/m (m being the walength grain)
    'Ph': Ph/s/m2
    'W': W/m2
    if band_or_wl is a photometric band, unit is disregarded and a mag. is returned


    EXAMPLE:
    >>> magref2bbflux(4.83, 'V', 'R', 5778)
    
    http://www.stsci.edu/hst/nicmos/tools/conversion_form.html
    """
    wl_bin=100 # 100 points on which to calculate the magnitude
    filters = import_bands(photometry_file) # imports filters from standard file
    if not filters.has_key(ref_band): raise Exception, "Unknown magnitude band"
    # calculates the black body reference flux for the ref magnitude, average over the photometric band
    flux_ref_surface = np.sum(blackbody_spectral_irr(teff, np.logspace(np.log10(filters[ref_band].start_wl), np.log10(filters[ref_band].end_wl), wl_bin)))/wl_bin
    if isinstance(band_or_wl, str): # if user gave a band as output format
        if not filters.has_key(band_or_wl): raise Exception, "Unknown magnitude band"
        # calculates the black body flux for the output mag
        flux_surface = np.sum(blackbody_spectral_irr(teff, np.logspace(np.log10(filters[band_or_wl].start_wl), np.log10(filters[band_or_wl].end_wl), wl_bin)))/wl_bin
        # calculates the output mag
        return -2.5*np.log10(flux_surface*filters[ref_band].flux_wm2m/filters[band_or_wl].flux_wm2m/flux_ref_surface)+ref_mag
    else: # if output is flux in W/m2/m
        # calculates the black body flux for the output mag
        flux_surface = blackbody_spectral_irr(teff, band_or_wl) # W/m2/m
        # gets units right
        conversion_factor = 1.
        if unit.upper().find('M')==-1: # if user wants W/m2 or Ph/s/m2 (not a flux per wl bin)
            if np.size(band_or_wl)==1: raise Exception, "You can't get W/m2 or Ph/s/m2 with only 1 input wavelength. Wavelength bins cannot be determined."
            conversion_factor = conversion_factor*funcs.array_to_binsize(band_or_wl) # wl_grain computed here
        if unit.upper().find('PH')!=-1: conversion_factor = conversion_factor*band_or_wl/1.9864456832693028e-25 # 1.9864456832693028e-25 = h * c
        # calculates the output flux
        return flux_surface * filters[ref_band].flux_wm2m / flux_ref_surface * 10**(-0.4 * ref_mag)*conversion_factor

