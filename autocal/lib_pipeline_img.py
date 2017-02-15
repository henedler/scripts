#!/usr/bin/python

import os, sys
import logging
import numpy as np


def angsep(ra1deg, dec1deg, ra2deg, dec2deg):
    """Returns angular separation between two coordinates (all in degrees)"""
    import math

    if ra1deg == ra2deg and dec1deg == dec2deg: return 0

    ra1rad=ra1deg*math.pi/180.0
    dec1rad=dec1deg*math.pi/180.0
    ra2rad=ra2deg*math.pi/180.0
    dec2rad=dec2deg*math.pi/180.0

    # calculate scalar product for determination
    # of angular separation
    x=math.cos(ra1rad)*math.cos(dec1rad)*math.cos(ra2rad)*math.cos(dec2rad)
    y=math.sin(ra1rad)*math.cos(dec1rad)*math.sin(ra2rad)*math.cos(dec2rad)
    z=math.sin(dec1rad)*math.sin(dec2rad)

    if x+y+z >= 1: rad = 0
    else: rad=math.acos(x+y+z)

    # Angular separation
    deg=rad*180/math.pi
    return deg


def flatten(f, channel=0, freqaxis=0):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """
    from astropy import wcs

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
    if naxis==2:
        return f[0].header,f[0].data

    w = wcs.WCS(f[0].header)
    wn=wcs.WCS(naxis=2)

    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"]=2
    copy=('EQUINOX','EPOCH')
    for k in copy:
        r=f[0].header.get(k)
        if r:
            header[k]=r

    slice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            slice.append(np.s_[:],)
        elif i==freqaxis:
            slice.append(channel)
        else:
            slice.append(0)

    # slice=(0,)*(naxis-2)+(np.s_[:],)*2
    return header,f[0].data[slice]


def size_from_reg(filename, region, coord, pixscale, pad=1.2):
    """
    find the minimum image size in pixels to cover a certain region given an img center
    coord : coordinate of the image center
    pad : multiplicative factor on the final size
    pixscale : pixel scale in arcsec
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    import pyregion

    # open fits
    fits = pyfits.open(filename)
    header, data = flatten(fits)
    w = pywcs.WCS(header)

    # extract mask
    r = pyregion.open(region)
    mask = r.get_mask(header=header, shape=data.shape)
    y,x = mask.nonzero()
    x_min = np.min(x)
    y_min = np.min(y)
    x_max = np.max(x)
    y_max = np.max(y)

    # find max dist in pixel on reference image
    x_c, y_c = w.all_world2pix(coord[0], coord[1], 0, ra_dec_order=True)
    max_x_size = 2*max([abs(x_c - x_min), abs(x_c - x_max)])
    max_y_size = 2*max([abs(y_c - y_min), abs(y_c - y_max)])

    # in case ref image and pixscale are different
    pix_factor = np.abs(header['CDELT1']*3600/pixscale)
    return int(pad*max([max_x_size, max_y_size])*pix_factor)

 
def scale_from_ms(ms):
    """
    Get the pixel scale in arcsec for a full-res image.
    It is 1/4 of the max resolution assuming zenit observation.
    Completely flagged lines are removed
    """
    from pyrap.tables import table
    import numpy as np

    c = 299792458.

    t = table(ms, ack=False).query('not all(FLAG)')
    col = t.getcol('UVW')
    maxdist = 0

    t = table(ms+'/SPECTRAL_WINDOW', ack=False)
    wavelenght = c/t.getcol('REF_FREQUENCY')[0]
    #print 'Wavelenght:', wavelenght,'m (Freq: '+str(t.getcol('REF_FREQUENCY')[0]/1.e6)+' MHz)'

    maxdist = np.max( np.sqrt(col[:,0]**2 + col[:,1]**2) )

    return int(round(wavelenght/maxdist*(180/np.pi)*3600/3.)) # arcsec

def blank_image_fits(filename, maskname, outfile=None, inverse=False, blankval=0.):
    """
    Set to "blankval" all the pixels inside the given region
    if inverse=True, set to "blankval" pixels outside region.

    filename: fits file
    region: ds9 region
    outfile: output name
    inverse: reverse region mask
    blankval: pixel value to set
    """
    import astropy.io.fits as pyfits
    import pyregion

    if outfile == None: outfile = filename

    with pyfits.open(maskname) as fits:
        mask = fits[0].data
    
    if inverse: mask = ~(mask.astype(bool))

    with pyfits.open(filename) as fits:
        data = fits[0].data

        assert mask.shape == data.shape # mask and data should be same shape

        sum_before = np.sum(data)
        data[mask] = blankval
        print "Sum of values: %f -> %f" % (sum_before, np.sum(data))
        fits.writeto(outfile, clobber=True)

 
def blank_image_reg(filename, region, outfile=None, inverse=False, blankval=0.):
    """
    Set to "blankval" all the pixels inside the given region
    if inverse=True, set to "blankval" pixels outside region.

    filename: fits file
    region: ds9 region
    outfile: output name
    inverse: reverse region mask
    blankval: pixel value to set
    """
    import astropy.io.fits as pyfits
    import pyregion

    if outfile == None: outfile = filename

    # open fits
    with pyfits.open(filename) as fits:
        origshape = fits[0].data.shape
        header, data = flatten(fits)
        # extract mask
        r = pyregion.open(region)
        mask = r.get_mask(header=header, shape=data.shape)
        if inverse: mask = ~mask
        data[mask] = blankval
        # save fits
        fits[0].data = data.reshape(origshape)
        fits.writeto(outfile, clobber=True)


def get_nose_img(filename):
    """
    Return the rms of all the pixels in an image
    """
    import astropy.io.fits as pyfits
    with pyfits.open(filename) as fits:
        return np.sqrt(np.mean((fits[0].data)**2))


def get_coord_centroid(filename, region):
    """
    Get centroid coordinates from an image and a region
    filename: fits file
    region: ds9 region
    """
    import astropy.io.fits as pyfits
    import astropy.wcs as pywcs
    import pyregion
    from scipy.ndimage.measurements import center_of_mass

    fits = pyfits.open(filename)
    header, data = flatten(fits)

    # extract mask and find center of mass
    r = pyregion.open(region)
    mask = r.get_mask(header=header, shape=data.shape)
    dec_pix, ra_pix = center_of_mass(mask)
    
    # convert to ra/dec in angle
    w = pywcs.WCS(fits[0].header)
    #w = w.celestial # needs newer astropy
    ra, dec = w.all_pix2world(ra_pix, dec_pix, 0, 0, 0, ra_dec_order=True)

    fits.close()
    return float(ra), float(dec)



