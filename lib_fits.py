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

import numpy as np
import os, sys, logging, re

from astropy.wcs import WCS as pywcs
from astropy.io import fits as pyfits
from astropy.cosmology import FlatLambdaCDM
from astropy.nddata import Cutout2D
import astropy.units as u

from reproject import reproject_interp, reproject_exact
reproj = reproject_exact

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
 
class AllImages():

    def __init__(self, filenames):
        if len(filenames) == 0:
            logging.error('Cannot find images!')
            raise ValueError()

        self.filenames = filenames
        self.images = []
        for filename in filenames:
            self.images.append(Image(filename))


    def __len__(self):
        return len(self.images)


    def center_at(self, ra, dec):
        """
        Re-align all images to a common center
        Parameters
        ----------
        ra: float, Right ascension in deg
        dec: float, declination in deg
        """
        for image in self.images:
            image.apply_recenter_cutout(ra, dec)

    def convolve_to(self, beam=None, circbeam=False):
        """
        Convolve all images to a common beam. By default, convolve to smalles common beam.

        Parameters
        ----------
        beam: list, optional. Default = None
            Beam parameters [b_major, b_minor, b_pa] in deg. None: find smalles common beam
        circbeam: bool, optional. Default = False
            Force circular beam
        """

        if beam is None:
            if circbeam:
                maxmaj = np.max([image.get_beam()[0] for image in self.images])
                target_beam = [maxmaj * 1.01, maxmaj * 1.01, 0.]  # add 1% to prevent crash in convolution
            else:
                from radio_beam import Beams
                my_beams = Beams([image.get_beam()[0] for image in self.images] * u.deg,
                                 [image.get_beam()[1] for image in self.images] * u.deg,
                                 [image.get_beam()[2] for image in self.images] * u.deg)
                common_beam = my_beams.common_beam()
                target_beam = [common_beam.major.value, common_beam.minor.value, common_beam.pa.value]
        else:
            target_beam = [beam[0] / 3600., beam[1] / 3600., beam[2]]
        logging.info('Final beam: %.1f" %.1f" (pa %.1f deg)' \
                     % (target_beam[0] * 3600., target_beam[1] * 3600., target_beam[2]))

        for image in self.images:
            image.convolve(target_beam)

    def regrid_common(self, size=None, pixscale=None, square=True):

        rwcs = pywcs(naxis=2)
        rwcs.wcs.ctype = self.images[0].get_wcs().wcs.ctype
        if pixscale:
            cdelt = pixscale
        else:
            cdelt = self.images[0].get_beam()[1] / 5.  # 1/5 of minor axes (deg)
        logging.info('Pixel scale: %f"' % (cdelt * 3600.))
        rwcs.wcs.cdelt = [-cdelt, cdelt]
        mra = self.images[0].img_hdr['CRVAL1']
        mdec = self.images[0].img_hdr['CRVAL2']
        rwcs.wcs.crval = [mra, mdec]

        # Calculate sizes of all images to find smalles size that fits all images
        sizes = np.empty((len(self.images,2)))
        for i, image in enumerate(self.images):
            sizes[i] = np.array(image.img_data.shape) * image.degperpixel()
        if size:
            size = np.array(size)
            if np.any(np.min(sizes, axis=1) < size):
                logging.warning(f'Requested size {size} is larger than smallest image size {np.min(sizes, axis=1)} in at least one dimension. This will result in NaN values in the regridded images.')
        else:
            size = np.min(sizes, axis=1)
            if square:
                size = np.min(size)

        xsize = int(np.rint(size[0] / cdelt))
        ysize = int(np.rint(size[-1] / cdelt))
        if xsize % 2 != 0: xsize += 1
        if ysize % 2 != 0: ysize += 1
        rwcs.wcs.crpix = [xsize / 2, ysize / 2]

        regrid_hdr = rwcs.to_header()
        regrid_hdr['NAXIS'] = 2
        regrid_hdr['NAXIS1'] = xsize
        regrid_hdr['NAXIS2'] = ysize

        logging.info('Image size: %f deg (%i %i pixels)' % (size, xsize, ysize))
        for image in self.images:
            this_regrid_hdr = regrid_hdr.copy()
            this_regrid_hdr['BMAJ'], this_regrid_hdr['BMIN'], this_regrid_hdr['BPA'] = image.get_beam()
            image.regrid(this_regrid_hdr)

    def write(self, suffix):
        """ Write all (changed) images to imagename-suffix.fits"""
        for image in self.images:
            image.write(image.imagefile.replace('.fits', f'-{suffix}.fits'))


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
        """
        Write to fits-file
        Parameters
        ----------
        filename: str, filename
        inflate: bool, optional. Default=False
                If False, write as flat 2D-fits file. If true, inflate to 4D.
        """
        if filename is None:
            filename = self.imagefile
        if inflate:
            # Inflate a fits file so that it becomes a 4D image. Return new header and data
            hdr_inf = pyfits.Header()
            hdr_inf['SIMPLE'  ] = self.img_hdr_orig['SIMPLE']
            hdr_inf['BITPIX'  ] = self.img_hdr_orig['BITPIX']
            hdr_inf['NAXIS'   ] = 4
            hdr_inf['NAXIS1'  ] = self.img_hdr['NAXIS1']
            hdr_inf['NAXIS2'  ] = self.img_hdr['NAXIS2']
            hdr_inf['NAXIS3'  ] = 1
            hdr_inf['NAXIS4'  ] = 1
            hdr_inf['EXTEND'  ] = 'T'
            hdr_inf['BUNIT'   ] = 'JY/BEAM'
            hdr_inf['RADESYS' ] = 'FK5'
            hdr_inf['EQUINOX' ] = 2000.
            hdr_inf['BMAJ'    ] = self.img_hdr['BMAJ']
            hdr_inf['BMIN'    ] = self.img_hdr['BMIN']
            hdr_inf['BPA'     ] = self.img_hdr['BPA']
            hdr_inf['EQUINOX' ] = self.img_hdr_orig['EQUINOX']
            hdr_inf['BTYPE'   ] = 'INTENSITY'
            hdr_inf['TELESCOP'] = self.img_hdr_orig['TELESCOP']
            hdr_inf['OBJECT'  ] = self.img_hdr_orig['OBJECT']
            hdr_inf['CTYPE1'  ] = self.img_hdr['CTYPE1']
            hdr_inf['CRPIX1'  ] = self.img_hdr['CRPIX1']
            hdr_inf['CRVAL1'  ] = self.img_hdr['CRVAL1']
            hdr_inf['CDELT1'  ] = self.img_hdr['CDELT1']
            hdr_inf['CUNIT1'  ] = self.img_hdr['CUNIT1']
            hdr_inf['CTYPE2'  ] = self.img_hdr['CTYPE2']
            hdr_inf['CRPIX2'  ] = self.img_hdr['CRPIX2']
            hdr_inf['CRVAL2'  ] = self.img_hdr['CRVAL2']
            hdr_inf['CDELT2'  ] = self.img_hdr['CDELT2']
            hdr_inf['CUNIT2'  ] = self.img_hdr['CUNIT2']
            hdr_inf['CTYPE3'  ] = 'FREQ'
            hdr_inf['CRPIX3'  ] = 1.
            hdr_inf['CRVAL3'  ] = self.freq
            hdr_inf['CDELT3'  ] = 10000000.
            hdr_inf['CUNIT3'  ] = 'Hz'
            hdr_inf['CTYPE4'  ] = 'STOKES'
            hdr_inf['CRPIX4'  ] = 1.
            hdr_inf['CRVAL4'  ] = 1.
            hdr_inf['CDELT4'  ] = 1.
            hdr_inf['CUNIT4'  ] = ' '
            pyfits.writeto(filename, self.img_data[np.newaxis,np.newaxis], hdr_inf, overwrite=True, output_verify='fix')
        else:
            pyfits.writeto(filename, self.img_data, self.img_hdr, overwrite=True, output_verify='fix')

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

    def convolve(self, target_beam, stokes=True):
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
        self.img_data = convolution.convolve(self.img_data, gauss_kern, boundary=None, preserve_nan=True)
        if stokes: # if not stokes image (e.g. spectral index, do not renormalize)
            self.img_data *= (target_beam[0]*target_beam[1])/(beam[0]*beam[1]) # since we are in Jt/b we need to renormalise

        self.set_beam(target_beam) # update beam

    def regrid(self, regrid_hdr):
        """ Regrid image to new header """
        logging.debug('%s: regridding' % (self.imagefile))
        self.img_data, __footprint = reproj((self.img_data, self.img_hdr), regrid_hdr, parallel=True)
        beam = self.get_beam()
        freq = find_freq(self.img_hdr)
        self.img_hdr = regrid_hdr
        self.img_hdr['FREQ'] = freq
        self.set_beam(beam) # retain beam info if not present in regrd_hdr

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

    def apply_recenter_cutout(self, ra, dec):
        """
        Center the image at ra, dec. The image will be cut accordingly.

        Parameters
        ----------
        ra: float, RA in deg.
        dec: float, Dec in deg.
        """
        new_center_pix = self.get_wcs().all_world2pix([[ra, dec]], 0)[0]
        shape = np.array(np.shape(self.img_data))
        new_shape = [np.min([shape[0] - new_center_pix[0], new_center_pix[0]]),
        np.min([shape[1] - new_center_pix[1], new_center_pix[1]])]
        print(self.get_wcs(), ra, dec)
        cutout = Cutout2D(self.img_data, new_center_pix, new_shape, wcs = self.get_wcs())
        hdr = cutout.wcs.to_header()
        hdr['BMAJ'], hdr['BMIN'], hdr['BPA'] = self.get_beam()
        self.img_hdr = hdr
        self.img_data = cutout.data
        logging.info(f'{self.imagefile}: recenter, size ({shape[0]}, {shape[1]})' \
                     f' -->  ({new_shape[0]}, {new_shape[1]})')

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

