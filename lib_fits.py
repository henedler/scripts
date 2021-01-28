#!/usr/bin/env python
#i -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - Francesco de Gasperin
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

from astropy.wcs import WCS as pywcs
from astropy.io import fits as pyfits
from astropy.cosmology import FlatLambdaCDM

import numpy as np
import os, sys, logging, re

def flatten(filename, channel=0, freqaxis=0):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """

    f = pyfits.open(filename)

    naxis=f[0].header['NAXIS']
    if naxis<2:
        raise RadioError('Can\'t make map from this')
    if naxis==2:
        #pass
        return f[0].header,f[0].data

    w = pywcs(f[0].header)
    wn = pywcs(naxis=2)

    wn.wcs.crpix[0]=w.wcs.crpix[0]
    wn.wcs.crpix[1]=w.wcs.crpix[1]
    wn.wcs.cdelt=w.wcs.cdelt[0:2]
    wn.wcs.crval=w.wcs.crval[0:2]
    wn.wcs.ctype[0]=w.wcs.ctype[0]
    wn.wcs.ctype[1]=w.wcs.ctype[1]

    header = wn.to_header()
    header["NAXIS"]=2
    header["NAXIS1"]=f[0].header['NAXIS1']
    header["NAXIS2"]=f[0].header['NAXIS2']
    copy=('EQUINOX','EPOCH')
    for k in copy:
        r=f[0].header.get(k)
        if r:
            header[k]=r

    dataslice=[]
    for i in range(naxis,0,-1):
        if i<=2:
            dataslice.append(np.s_[:],)
        elif i==freqaxis:
            dataslice.append(channel)
        else:
            dataslice.append(0)

    # add freq
    header["FREQ"] = find_freq(f[0].header)

    # add beam if present
    try:
        header["BMAJ"]=f[0].header['BMAJ']
        header["BMIN"]=f[0].header['BMIN']
        header["BPA"]=f[0].header['BPA']
    except:
        pass

    # slice=(0,)*(naxis-2)+(np.s_[:],)*2
    return header, f[0].data[tuple(dataslice)]



def correct_beam_header(header):
    """ 
    Find the primary beam headers following AIPS convenction
    """
    if ('BMAJ' in header) and ('BMIN' in header) and ('PA' in header): return header
    elif 'HISTORY' in header:
        for hist in header['HISTORY']:
            if 'AIPS   CLEAN BMAJ' in hist:
                # remove every letter from the string
                bmaj, bmin, pa = re.sub(' +', ' ', re.sub('[A-Z ]*=','',hist)).strip().split(' ')
                header['BMAJ'] = float(bmaj)
                header['BMIN'] = float(bmin)
                header['BPA'] = float(pa)
    return header

def find_freq(header):
    """
    Find frequency value in most common places of a fits header
    """
    if not header.get('RESTFRQ') is None and not header.get('RESTFRQ') == 0:
        return header.get('RESTFRQ')
    elif not header.get('FREQ') is None and not header.get('FREQ') == 0:
        return header.get('FREQ')
    else:
        for i in range(5):
            type_s = header.get('CTYPE%i' % i)
            if type_s is not None and type_s[0:4] == 'FREQ':
                return header.get('CRVAL%i' % i)

    return None # no freq information found
 

class Image(object):

    def __init__(self, imagefile):
        """
        imagefile: name of the fits file
        """

        self.imagefile = imagefile
        header = pyfits.open(imagefile)[0].header
        header = correct_beam_header(header)
        self.img_hdr_orig = header

        try:
            beam = [header['BMAJ'], header['BMIN'], header['BPA']]
        except:
            logging.warning('%s: No beam information found.' % self.imagefile)
            sys.exit(1)
        logging.debug('%s: Beam: %.1f" %.1f" (pa %.1f deg)' % \
                (self.imagefile, beam[0]*3600., beam[1]*3600., beam[2]))

        self.freq = find_freq(header)
        if self.freq is None:
            logging.warning('%s: No frequency information found.' % self.imagefile)
            # sys.exit(1)
        else:
            logging.debug('%s: Frequency: %.0f MHz' % (self.imagefile, self.freq/1e6))

        self.noise = None
        self.img_hdr, self.img_data = flatten(self.imagefile)
        self.img_hdu = pyfits.ImageHDU(data=self.img_data, header=self.img_hdr)
        self.set_beam(beam)
        self.set_freq(self.freq)
        self.ra = self.img_hdr['CRVAL1']
        self.dec = self.img_hdr['CRVAL2']

    def write(self, filename=None, inflate=False):
        if filename is None:
            filename = self.imagefile
        if inflate:
            """ Inflate a fits file so that it becomes a 4D image. Return new header and data """
            hdr = self.img_hdr
            hdr_inf = self.img_hdr_orig
            hdr_inf['CRVAL1'] = hdr['CRVAL1']
            hdr_inf['CRVAL2'] = hdr['CRVAL2']
            # hdr_string = f"SIMPLE  =                    T / file does conform to FITS standard             BITPIX  =                  -32 / number of bits per data pixel                  NAXIS   =                    4 / number of data axes                            NAXIS1  =                 {hdr['NAXIS1']} / length of data axis 1                          NAXIS2  =                 {hdr['NAXIS2']} / length of data axis 2                          NAXIS3  =                    1 / length of data axis 3                          NAXIS4  =                    1 / length of data axis 4                          EXTEND  =                    T / FITS dataset may contain extensions            COMMENT   FITS (Flexible Image Transport System) format is defined in 'AstronomyCOMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H BSCALE  =                   1.                                                  BZERO   =                   0.                                                  BUNIT   = 'JY/BEAM '           / Units are in Jansky per beam                   BMAJ    =  {hdr['BMAJ']}                                                  BMIN    =  {hdr['BMAJ']}                                                  BPA     =     {hdr['BPA']}                                                  EQUINOX =                2000. / J2000                                          BTYPE   = 'Intensity'                                                           TELESCOP= 'LOFAR   '                                                            OBSERVER= 'unknown '                                                            OBJECT  = 'BEAM_0  '                                                            ORIGIN  = 'WSClean '           / W-stacking imager written by Andre Offringa    CTYPE1  = 'RA---SIN'           / Right ascension angle cosine                   CRPIX1  =                 {hdr['CRPIX1']}                                                 CRVAL1  =          {hdr['CRVAL1']:f}                                                   CDELT1  =             {hdr['CDELT1']:f}                                                 CUNIT1  = 'deg     '                                                            CTYPE2  = 'DEC--SIN'           / Declination angle cosine                       CRPIX2  =                 {hdr['CRPIX2']}                                                 CRVAL2  =     {hdr['CRVAL2']}                                                   CDELT2  =             {hdr['CDELT2']}                                                  CUNIT2  = 'deg     '                                                            CTYPE3  = 'FREQ    '           / Central frequency                              CRPIX3  =                   1.                                                  CRVAL3  =     {find_freq(hdr)}                                                  CDELT3  =            46875000.                                                  CUNIT3  = 'Hz      '                                                            CTYPE4  = 'STOKES  '                                                            CRPIX4  =                   1.                                                  CRVAL4  =                   1.                                                  CDELT4  =                   1.                                                  CUNIT4  = '        '                                                            SPECSYS = 'TOPOCENT' "
            # hdr_inflated = Header.fromstring(hdr_string)
            pyfits.writeto(filename, self.img_data[np.newaxis,np.newaxis], hdr_inf, overwrite=True)
        else:
            pyfits.writeto(filename, self.img_data, self.img_hdr, overwrite=True)

    def set_beam(self, beam):
        self.img_hdr['BMAJ'] = beam[0]
        self.img_hdr['BMIN'] = beam[1]
        self.img_hdr['BPA'] = beam[2]

    def set_freq(self, freq):
        self.img_hdr['RESTFREQ'] = freq
        self.img_hdr['FREQ'] = freq

    def get_beam(self):
        return [self.img_hdr['BMAJ'], self.img_hdr['BMIN'], self.img_hdr['BPA']]

    def get_wcs(self):
        return pywcs(self.img_hdr)

    def apply_region(self, regionfile, blankvalue=np.nan, invert=False):
        """
        Blank inside mask
        invert: blank outside region
        """
        import pyregion
        if not os.path.exists(regionfile):
            logging.error('%s: Region file not found.' % regionfile)
            sys.exit(1)

        logging.debug('%s: Apply region %s' % (self.imagefile, regionfile))
        r = pyregion.open(regionfile)
        mask = r.get_mask(header=self.img_hdr, shape=self.img_data.shape)
        if invert: self.img_data[~mask] = blankvalue
        else: self.img_data[mask] = blankvalue


    def apply_mask(self, mask, blankvalue=np.nan, invert=False):
        """
        Blank inside mask
        invert: blank outside mask
        """
        logging.debug('%s: Apply mask' % self.imagefile)
        if invert: self.img_data[~mask] = blankvalue
        else: self.img_data[mask] = blankvalue


    def calc_noise(self, niter=1000, eps=None, sigma=5, bg_reg=None):
        """
        Return the rms of all the pixels in an image
        niter : robust rms estimation
        eps : convergency criterion, if None is 1% of initial rms
        bg_reg : If ds9 region file provided, use this as background region
        """
        if bg_reg:
            import pyregion
            if not os.path.exists(bg_reg):
                logging.error('%s: Region file not found.' % bg_reg)
                sys.exit(1)

            logging.debug('%s: Apply background region %s' % (self.imagefile, bg_reg))
            r = pyregion.open(bg_reg)
            mask = r.get_mask(header=self.img_hdr, shape=self.img_data.shape)
            print('STD:', np.nanstd(self.img_data[mask]),np.nanstd(self.img_data[~mask]))
            self.noise = np.nanstd(self.img_data[mask])
            logging.debug('%s: Noise: %.3f mJy/b' % (self.imagefile, self.noise * 1e3))
        else:
            if eps == None: eps = np.nanstd(self.img_data)*1e-3
            data = self.img_data[ ~np.isnan(self.img_data) ] # remove nans
            if len(data) == 0: return 0
            oldrms = 1.
            for i in range(niter):
                rms = np.nanstd(data)
                if np.abs(oldrms-rms)/rms < eps:
                    self.noise = rms
                    #print('%s: Noise: %.3f mJy/b' % (self.imagefile, self.noise*1e3))
                    logging.debug('%s: Noise: %.3f mJy/b' % (self.imagefile, self.noise*1e3))
                    return rms

                data = data[np.abs(data)<sigma*rms]
                oldrms = rms
            raise Exception('Noise estimation failed to converge.')

    def convolve(self, target_beam):
        """
        Convolve *to* this rsolution
        beam = [bmaj, bmin, bpa]
        """
        from lib_beamdeconv import deconvolve_ell, EllipticalGaussian2DKernel
        from astropy import convolution

        # if difference between beam is negligible <1%, skip - it mostly happens when beams are exactly the same
        beam = self.get_beam()
        if (np.abs((target_beam[0]/beam[0])-1) < 1e-2) and (np.abs((target_beam[1]/beam[1])-1) < 1e-2) and (np.abs(target_beam[2] - beam[2]) < 1):
            logging.debug('%s: do not convolve. Same beam.' % self.imagefile)
            return
        # first find beam to convolve with
        convolve_beam = deconvolve_ell(target_beam[0], target_beam[1], target_beam[2], beam[0], beam[1], beam[2])
        if convolve_beam[0] is None:
            logging.error('Cannot deconvolve this beam.')
            sys.exit(1)
        logging.debug('%s: Convolve beam: %.3f" %.3f" (pa %.1f deg)' \
                % (self.imagefile, convolve_beam[0]*3600, convolve_beam[1]*3600, convolve_beam[2]))
        # do convolution on data
        bmaj, bmin, bpa = convolve_beam
        assert abs(self.img_hdr['CDELT1']) == abs(self.img_hdr['CDELT2'])
        pixsize = abs(self.img_hdr['CDELT1'])
        fwhm2sigma = 1./np.sqrt(8.*np.log(2.))
        gauss_kern = EllipticalGaussian2DKernel((bmaj*fwhm2sigma)/pixsize, (bmin*fwhm2sigma)/pixsize, (90+bpa)*np.pi/180.) # bmaj and bmin are in pixels
        self.img_data = convolution.convolve(self.img_data, gauss_kern, boundary=None)
        self.img_data *= (target_beam[0]*target_beam[1])/(beam[0]*beam[1]) # since we are in Jt/b we need to renormalise
        self.set_beam(target_beam) # update beam

    def apply_shift(self, dra, ddec):
        """
        Shift header by dra/ddec
        dra, ddec in degree
        """
        # correct the dra shift for np.cos(DEC*np.pi/180.) -- only in the log as the reference val doesn't need it!
        logging.info('%s: Shift %.2f %.2f (arcsec)' % (self.imagefile, dra*3600*np.cos(self.dec*np.pi/180.), ddec*3600))
        dec = self.img_hdr['CRVAL2']
        self.img_hdr['CRVAL1'] += dra
        self.img_hdr['CRVAL2'] += ddec

    def degperpixel(self):
        """
        Return the number of degrees per image pixel. This assumes SQUARE pixels!
        Returns
        -------
        degperpixel: float
        """
        wcs = self.get_wcs()
        #TODO crosscheck this method
        return np.abs(wcs.all_pix2world(0, 0, 0)[1] - wcs.all_pix2world(0, 1, 0)[1])

    def degperkpc(self, z):
        """
        How many degrees are there per kpc? Assume H0=70km/S/Mpcm O_m = 0.3

        Parameters
        ----------
        z: Source redshift

        Returns
        -------
        degperkpc: float
        """
        wcs = self.get_wcs()
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        return cosmo.arcsec_per_kpc_proper(z).value / 3600.


    def pixelperkpc(self, z):
        """
        Return the number of pixel per kpc. This assumes SQUARE pixels!
        Returns
        -------
        pixelperkpc: float
        """
        return self.degperkpc(z) / self.degperpixel()

