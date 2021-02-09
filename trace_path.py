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
    ell_str = f"ellipse({ra}, {dec}, {b[0]*3600}\", {b[1]*3600}\", {b[2]})"
    return pyregion.parse(ell_str)

def interpolate_path(region, image, n, z):
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
    """
    approxres = 1000
    df = pd.DataFrame()
    # Load region, region must be sorted!
    trace = pyregion.open(region)
    trace = np.array([p.coord_list for p in trace.as_imagecoord(image.img_hdr)])

    # Linear interpolation
    distance_lin = np.cumsum(np.linalg.norm(np.diff(trace, axis=0), axis=1))
    distance_lin = np.insert(distance_lin,0,0)
    tck, u = interpolate.splprep([trace[:,0], trace[:,1]], u=distance_lin, s=0)

    # Cubic spline interpolation of linear interpolated data to get correct distances
    # Calculate a lot of points on the spline and then use the accumulated distance from points n to point n+1 as the integrated path
    xy = interpolate.splev(np.linspace(0,u[-1],approxres), tck, ext=2)
    distance_cube = np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=0))
    distance_cube = np.insert(distance_cube,0,0)
    tck, u = interpolate.splprep([xy[0], xy[1]], s=0)
    length = np.linspace(0, 1, n) # length at point i in fraction
    xy = np.array(interpolate.splev(length, tck, ext=2)).T # points where we sample in image coords.
    length = length*distance_cube[-1]/image.pixelperkpc(z) # length at point i in kpc
    df['l'] = length
    log.info(f"Trace consists of {len(trace)} points. Linear interpolation length: {distance_lin[-1]/image.pixelperkpc(z):.2f} kpc,  cubic interpolation length: {distance_cube[-1]/image.pixelperkpc(z):.2f} kpc")
    if not np.all(np.isclose([trace[0], trace[-1]], [xy[0], xy[-1]])):  # Assert
        raise ValueError(f'Check points: First diff - {trace[0] - xy[0]}; last diff = {trace[-1] - xy[-1]}')
    data = image.img_data
    # data_ma = np.ma.masked_invalid(data)
    # mask = np.isnan(data)
    # interp_data = interpolate.RectBivariateSpline(np.arange(data.shape[1]), np.arange(data.shape[0]), data_ma.T)
    # trace_data_px = interp_data(xy[:,0], xy[:,1], grid=False)
    trace_data_psf = np.zeros(len(xy))
    # trace_data_psf_err = np.zeros_like(trace_data_px)
    # TODO: weight data?
    radec = image.get_wcs().all_pix2world(xy,0) #TODO check origin
    df['ra'], df['dec'] = radec.T
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
        trace_data_psf[i] = np.nanmean(data[beam_mask])
    df[f'F_{image.freq:.2e}'] = trace_data_psf
    df[f'F_err_{image.freq:.2e}'] = image.calc_noise() #TODO bg reg
    return df

parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('region', help='ds9 point regions defining the path, must be ordered from start to end!')
parser.add_argument('bg', help='Path to ds9 region for background estimation.')
parser.add_argument('stokesi', nargs='+', default=[], help='List of fits images of Stokes I.')
parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
parser.add_argument('-b', '--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
parser.add_argument('-n', '--n', default=100, type=int, help='Number of points to sample.')
parser.add_argument('-o', '--out', default='trace_path', type=str, help='Name of the output image and csv file.')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity.')
args = parser.parse_args()

if args.verbose:
    log.root.setLevel(log.INFO)

df_list = []
stokesi = []
all_beams = []
all_images = []

for imname in args.stokesi:
    all_images.append(lib_fits.Image(imname))
    # my_beams = Beams([b[0] for b in all_beams] * units.deg, [b[1] for b in all_beams] * units.deg,
    #                  [b[2] for b in all_beams] * units.deg)
    # common_beam = my_beams.common_beam()
    # target_beam = [common_beam.major.value, common_beam.minor.value, common_beam.pa.value]
maxmaj = np.max([b[0] for b in all_beams])
target_beam = [maxmaj * 1.01, maxmaj * 1.01, 0.]  # add 1%

if args.beam:
    if target_beam[0] > args.beam:
        log.error(f'Specified beam {args.beam} is smaller than largest beam major axis {target_beam[0]}')
        sys.exit(1)
    else:
        target_beam = [args.beam, args.beam, 0.]

log.info('Convole to beam: %.1f" %.1f" (pa %.1f deg)' \
             % (target_beam[0] * 3600., target_beam[1] * 3600., target_beam[2]))

for im in args.stokesi:
    im_obj = lib_fits.Image(im)
    im_obj.convolve(target_beam)
    stokesi.append(im_obj)
    # do the point-spacing and interpolation
    df_list.append(interpolate_path(args.region, stokesi[-1], args.n, args.z))

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

for i, im_obj in enumerate(stokesi):
    im = im_obj.imagefile
    # scale data to 54MHz using beta=-0.6
    scale = 1 #(54e6/im_obj.freq)**(-0.8)
    ax.plot(df['l'], df[f'F_{im_obj.freq:.2e}']*scale, color=f'C{i}', label=im.split('/')[-1])
    ax.fill_between(df['l'], scale*(df[f'F_{im_obj.freq:.2e}']-df[f'F_err_{im_obj.freq:.2e}']), scale*(df[f'F_{im_obj.freq:.2e}']+df[f'F_err_{im_obj.freq:.2e}']), color=f'C{i}', alpha=0.4)
    ax.set_ylabel('Flux density [Jy]', color=f'C{i}')

ax.legend(loc='best')
ax.set_xlim([0,df['l'].max()])

log.info(f'Save plot to {args.out}.pdf...')
plt.savefig(args.out+'.pdf')