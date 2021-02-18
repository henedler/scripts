#!/usr/bin/env python
#
# Script to trace the flux-density evolution of a path in one or more fits-files.
# Required input: a ds9 region file which contains an ordered sequence of points, fits image(s).

import os, sys, argparse
import logging as log
import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord, Angle
# from astropy.convolution import Gaussian2DKernel
# from radio_beam import Beams
# from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate#, integrate, optimize
import pandas as pd
import pyregion

import lib_fits
from lib_linearfit import fit_path_to_regions
log.root.setLevel(log.INFO)


def beam_ellipse(ra, dec, image):
    """
    Return pyregion ellpse regon coresponding to the image beam at ra, dec
    Parameters
    ----------
    ra: float, ra in degrees
    dec: float, dec in degrees
    image: obj, lib_fits.Image

    Returns
    -------
    beam_ellipse: obj, pyregion region
    """
    b = image.get_beam()
    ra = Angle(str(ra)+'d', unit=units.deg).to_string(sep = ':', unit=units.hour)
    dec = Angle(dec, unit=units.deg).to_string(sep = ':')
    fact = 1/np.sqrt(4*np.log(2)) # this factor is needed to make the pixels in the beam area match the beam_area from the lib_fits function. Not sure why...
    ell_str = f"ellipse({ra}, {dec}, {b[0]*3600*fact}\", {b[1]*3600*fact}\", {b[2]})"
    return pyregion.parse(ell_str)



def interpolate_path(region, image, n, z, flux_scale_err = 0.0):
    """
    Interpolate a path defined by ordered ds9 points and calculate 'n' evenly spaced points on this path.
    Slide a psf-sized region along this path and calculate the mean image value along the path.
    Parameters
    ----------
    region: string
        ds9 region filename - MUST be ordered from start to end.
    image: obj, lib_fits.Image
    n: int, number of points to space on the path
    z: float, redshift

    Returns
    -------
    trace_data: array, shape(n,)
        Values of sliding psf region means.
    xy: array, shape (n,2)
        Evenly spaced points on the path
    path_length: array, shape (n,)
        Length of the path from first point to point n in kpc
    flux_scale_err: float, optional. Default = 0.0
        Relative error of the flux scale.
    """

    df = pd.DataFrame()
    xy, l = fit_path_to_regions(region, image, z, n)
    df['l'] = l
    radec = image.get_wcs().all_pix2world(xy,0) #TODO check origin
    df['ra'], df['dec'] = radec.T

    # data = image.img_data
    # data_ma = np.ma.masked_invalid(data)
    # interp_data = interpolate.RectBivariateSpline(np.arange(data.shape[1]), np.arange(data.shape[0]), data_ma.T)
    # trace_data_px = interp_data(xy[:,0], xy[:,1], grid=False)
    # df[f'F_{image.freq:.2e}'] = trace_data_px
    # df[f'F_err_{image.freq:.2e}'] = np.sqrt((image.noise**2) + (trace_data_px * flux_scale_err)**2)

    # get covariances
    # C = np.empty((len(xy), len(xy)))
    # for i, p1 in enumerate(xy):
    #     for j, p2 in enumerate(xy[i:]):
    #         image.pixel_covariance(p1, p2)
    path_data_psf = np.zeros(len(xy))
    path_data_error = np.zeros(len(xy))
    for i,p in enumerate(radec):
        beam = beam_ellipse(p[0], p[1], image).as_imagecoord(image.img_hdr)
        beam_mask = beam.get_mask(hdu=image.img_hdu, header=image.img_hdr, shape=image.img_data.shape)
        # b = lib_fits.Image(args.image).get_beam()
        # where0, where1 = np.argwhere(np.any(beam_mask, axis=0)), np.argwhere(np.any(beam_mask, axis=1))
        # n0 = (where0[-1] - where0[0])[0]
        # n1 = (where1[-1] - where1[0])[0]
        # fwhm2sig = 1. / np.sqrt(8. * np.log(2.))
        # psf_weight = Gaussian2DKernel(fwhm2sig*b[0]/degperpixel, fwhm2sig*b[1]/degperpixel, theta=np.deg2rad(b[2]), x_size=n0, y_size=n1).array
        # psf_weight/np.max(psf_weight)
        # print(psf_weight)
        # print(psf_weight.shape, data[beam_mask].shape)
        # print(data[beam_mask])
        npix = np.sum(beam_mask)
        data = image.img_data[beam_mask]
        nndata = data[~np.isnan(data)]
        print(npix, image.get_beam_area(unit='pixel'))
        path_data_psf[i] = np.sum(nndata) / image.get_beam_area(unit='pixel')
        path_data_error[i] = image.noise * np.sqrt(npix / image.get_beam_area(unit='pixel'))
        print(f'flux: {path_data_psf[i]:.4f} +/- {path_data_error[i]:.4f}')

    df[f'F_{image.freq/1e6:.0f}'] = path_data_psf
    df[f'F_err_{image.freq/1e6:.0f}'] = path_data_error

    # fwhm2sigma = 1. / np.sqrt(8. * np.log(2.))
    # a_beam =  2*np.pi*image.get_beam()[0]*image.get_beam()[1]*fwhm2sigma**2
    # df[f'F_err_{image.freq:.2e}'] = np.sqrt((image.noise**2) + (trace_data_px * flux_scale_err)**2)
    # df[f'F_err_{image.freq:.2e}'] = np.sqrt((image.noise**2)/a_beam + (trace_data_px * flux_scale_err)**2)
    return df


# def get_spidx(f1, ferr1, f2, ferr2, freq1, freq2):


parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('region', help='ds9 point regions defining the path, must be ordered from start to end!')
parser.add_argument('bg', help='Path to ds9 region for background estimation.')
parser.add_argument('stokesi', nargs='+', default=[], help='List of fits images of Stokes I.')
parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
parser.add_argument('-b', '--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
parser.add_argument('-n', '--n', default=100, type=int, help='Number of points to sample.')
parser.add_argument('-o', '--out', default='trace_path', type=str, help='Name of the output image and csv file.')
parser.add_argument('--align', action='store_true', help='Align the images.')
parser.add_argument('--reuse-shift', action='store_true', help='Resue catalogue shifted images if available.')
parser.add_argument('--reuse-regrid', action='store_true', help='Resue regrid images if availalbe.')
parser.add_argument('--reuse-df', action='store_true', help='Resue data frame if availalbe.')
parser.add_argument('-d', '--debug', action='store_true', help='Debug output.')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity.')
args = parser.parse_args()

if args.verbose:
    log.root.setLevel(log.INFO)

stokesi = []
all_images = lib_fits.AllImages(args.stokesi)

# find+apply shift w.r.t. first image
if args.align:
    if args.reuse_shift and np.all([os.path.exists(name.replace('.fits', '-shifted.fits')) for name in args.stokesi]):
        log.info('Reuse cat shifted images.')
        all_images = lib_fits.AllImages(
            [name.replace('.fits', '-recenter-convolve-regrid.fits') for name in args.stokesi])
    else:
        log.info('Align images to catalogue matches')
        all_images.align_catalogue()
        if args.debug: all_images.write('shifted')

# convolve images to the same beam (for now force circ beam)
if args.reuse_regrid and np.all([os.path.exists(name.replace('.fits', '-recenter-convolve-regrid.fits')) for name in args.stokesi]):
    log.info('Reuse prepared images.')
    all_images = lib_fits.AllImages([name.replace('.fits', '-recenter-convolve-regrid.fits') for name in args.stokesi])
else:
    log.info('Recenter, convolve and regrid all images')
    # if args.radec:
    #     all_images.center_at(*args.radec)
    # else: # recenter at first image
        # all_images.center_at(all_images[0].img_hdr['CRVAL1'], all_images[0].img_hdr['CRVAL2'])
    # if args.debug: all_images.write('recenter')
    all_images.convolve_to(circbeam=True) # elliptical beam seems buggy in some cases. Also, circ beam is nice to treat covariance matrix of pixels
    if args.debug: all_images.write('recenter-convolve')
    all_images.regrid_common()
    all_images.write('recenter-convolve-regrid')

for image in all_images:
    image.calc_noise() # update noise in all images TODO: which is best way? BG region??

if args.reuse_df and os.path.exists(f'{args.out}.csv'):
    df = pd.read_csv(f'{args.out}.csv')
else:
    df_list = []
    for image in all_images:
        df_list.append(interpolate_path(args.region, image, args.n, args.z))

    df = pd.concat(df_list, axis=1, join_axes=[df_list[0].index])
    df = df.loc[:,~df.columns.duplicated()]

    log.info(f'Save DataFrame to {args.out}.csv...')
    df.to_csv(f'{args.out}.csv')

# do the plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
fig, ax = plt.subplots(constrained_layout=True)
ax.set_xlabel('distance [kpc]')

for i, im_obj in enumerate(all_images):
    im = im_obj.imagefile
    # scale data to 54MHz using beta=-0.6
    scale = (144e6/im_obj.freq)**(-0.6)

    ax.plot(df['l'], df[f'F_{im_obj.freq/1e6:.0f}']*scale, color=f'C{i}', label=r'$\nu = $' + f'{im_obj.freq/1e6:.0f} MHz')
    ax.hlines(y=0.0, xmin=df['l'].min(), xmax=df['l'].max(), linewidth=1, color='grey', ls='dotted', alpha=0.5)
    ax.fill_between(df['l'], scale*(df[f'F_{im_obj.freq/1e6:.0f}']-df[f'F_err_{im_obj.freq/1e6:.0f}']), scale*(df[f'F_{im_obj.freq/1e6:.0f}']+df[f'F_err_{im_obj.freq/1e6:.0f}']), color=f'C{i}', alpha=0.3)
    ax.set_ylabel(r'Flux density at 144MHz assuming $\alpha = -0.6$ [Jy]')

ax.legend(loc='best')
ax.set_xlim([0,df['l'].max()])
# ax.set_ylim(bottom = 0)

log.info(f'Save plot to {args.out}.pdf...')
plt.savefig(args.out+'.pdf')