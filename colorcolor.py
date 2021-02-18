#!/usr/bin/env python
#
# Script to trace the flux-density evolution of a path in one or more fits-files.
# Required input: a ds9 region file which contains an ordered sequence of points, fits image(s).

import os, sys, argparse, pickle
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

from lib_linearfit import linear_fit_bootstrap, fit_path_to_regions
log.root.setLevel(log.INFO)



parser = argparse.ArgumentParser(description='Trace a path defined by a points in a ds9 region file. \n    trace_path.py <fits image> <ds9 region>')
parser.add_argument('region', help='region to restrict the analysis to')
parser.add_argument('bg', help='Path to ds9 region for background estimation.')
parser.add_argument('stokesi', nargs=3, default=[], help='List of fits images of Stokes I.')
parser.add_argument('-z', '--z', default = 0.1259, type=float, help='Source redshift. Defaults to A1033.')
parser.add_argument('--radec', dest='radec', nargs='+', type=float, help='RA/DEC where to center final image in deg (if not given, center on first image)')
parser.add_argument('-b', '--beam', default = None, type=float, help='If specified, convolve all images to a circular beam of this radius (deg). Otherwise, convolve to a circular beam with a radius equal to the largest beam major axis.')
parser.add_argument('--regionpath', type=str, help='Name of the region defining a path on the sky.')
parser.add_argument('-o', '--out', default='colorcolor', type=str, help='Name of the output image and csv file.')
parser.add_argument('-d', '--debug', action='store_true', help='Debug output.')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbosity.')
parser.add_argument('-r', '--reuse', action='store_true', help='Reuse intermediate steps.')
args = parser.parse_args()

if args.verbose:
    log.root.setLevel(log.INFO)
if len(args.stokesi) != 3:
    log.error('Need three images for color-color analysis')
df_list = []
# sort frequency
freqs = [lib_fits.Image(filepath).freq for filepath in args.stokesi]
all_images = lib_fits.AllImages([args.stokesi[i] for i in np.argsort(freqs)])
freqs = np.sort(freqs)
# convolve images to the same beam (for now force circ beam)
if args.reuse and np.all([os.path.exists(name.replace('.fits', '-recenter-convolve-regrid.fits')) for name in args.stokesi]):
    log.info('Reuse prepared images.')
    all_images = lib_fits.AllImages([name.replace('.fits', '-recenter-convolve-regrid.fits') for name in args.stokesi])
else:
    log.info('Recenter, convolve and regrid all images')
    if args.radec:
        all_images.center_at(*args.radec)
    else: # recenter at first image
        all_images.center_at(all_images[0].img_hdr['CRVAL1'], all_images[0].img_hdr['CRVAL2'])
    if args.debug: all_images.write('recenter')
    all_images.convolve_to(circbeam=True) # elliptical beam seems buggy in some cases. Also, circ beam is nice to treat covariance matrix of pixels
    if args.debug: all_images.write('recenter-convolve')
    all_images.regrid_common(pixscale=4.)
    all_images.write('recenter-convolve-regrid')

mask = np.ones_like(all_images[0].img_data)
for image in all_images:
    image.calc_noise() # update. TODO: which is best way? BG region??
    image.blank_noisy(3) # blank 3 sigma
    image.apply_region(args.region, invert=True)
    isnan = np.isnan(image.img_data)
    mask[isnan] = 0
log.debug(f"{np.sum(mask):.2%} pixel remaining for analysis.")
if args.debug: all_images.write('blank')

# df = pd.concat(df_list, axis=1, join_axes=[df_list[0].index])
# df = df.loc[:,~df.columns.duplicated()]

# log.info(f'Save DataFrame to {args.out}.csv...')
# df.to_csv(f'{args.out}.csv')

if os.path.exists('temp_colorcolor.pickle') and args.reuse:
    with open( 'temp_colorcolor.pickle', "rb" ) as f:
        spidx, spidx_err = pickle.load(f)
else:
    spidx = np.zeros((len(np.nonzero(mask)[0]), 2))  # spidx lo spidx hi
    spidx_err = np.zeros((len(np.nonzero(mask)[0]), 2))  # spidx lo spidx hi
    for i, (x, y) in enumerate(np.transpose(np.nonzero(mask))):
        print('.', end=' ')
        sys.stdout.flush()
        val4reglo = [image.img_data[x, y] for image in all_images[0:2]]
        val4reghi = [image.img_data[x, y] for image in all_images[1:]]
        noise = [image.noise for image in all_images]
        (alo, blo, salo, sblo) = linear_fit_bootstrap(x=freqs[0:2], y=val4reglo, yerr=noise[0:2], tolog=True)
        (ahi, bhi, sahi, sbhi) = linear_fit_bootstrap(x=freqs[1:], y=val4reghi, yerr=noise[1:], tolog=True)
        spidx[i] = alo, ahi
        spidx_err[i] = salo, sahi
    with open( 'temp_colorcolor.pickle', "wb" ) as f:
        pickle.dump([spidx, spidx_err], f)

if args.regionpath:
    path_xy, l = fit_path_to_regions(args.regionpath, all_images[0], args.z, 100)
    print(np.shape(np.nonzero(mask)), np.shape(path_xy))
    distance = np.zeros(len(np.nonzero(mask)[0]))
    for i, pix in enumerate(np.transpose(np.nonzero(mask))):
        idx_closest = np.argmin(np.linalg.norm(pix[np.newaxis] - path_xy, axis=1))
        print(np.linalg.norm(pix[np.newaxis] - path_xy, axis=0))
        distance[i] = l[idx_closest]
        log.debug(f'Closest point on path is a distance {distance[i]}')


# do the plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.scatter(np.nonzero(mask)[0], np.nonzero(mask)[1], c=distance, marker='square')



freqs *= 1e-6
plt.xlabel(r'$\alpha_{' + f'{freqs[0]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + r'}$')
plt.ylabel(r'$\alpha_{' + f'{freqs[1]:.0f}\mathrm{{MHz}}' + '}^{' + f'{freqs[2]:.0f}\mathrm{{MHz}}' + r'}$')
print(distance)
_min, _max = np.min(spidx), np.max(spidx)
plt.scatter(spidx[:,0], spidx[:,1], s=2, c = distance)
plt.plot([_min, _min], [_max, _max], label='injection')
plt.colorbar()

plt.xlim([_min-0.1, _max+0.1])
plt.ylim([_min-0.1, _max+0.1])

log.info(f'Save plot to {args.out}.pdf...')
plt.savefig(args.out+'.pdf')