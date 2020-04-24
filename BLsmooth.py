#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

# Usage: BLavg.py vis.MS
# Load a MS, smooth visibilities according to the baseline lenght,
# i.e. shorter BLs are averaged more, and write a new MS

import os, sys, time
import optparse, itertools
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gfilter
import casacore.tables as pt
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

logging.info('BL-based smoother - Francesco de Gasperin')

def addcol(ms, incol, outcol):
    if outcol not in ms.colnames():
        logging.info('Adding column: '+outcol)
        coldmi = ms.getdminfo(incol)
        coldmi['NAME'] = outcol
        ms.addcols(pt.makecoldesc(outcol, ms.getcoldesc(incol)), coldmi)
    if outcol != incol:
        # copy columns val
        logging.info('Set '+outcol+'='+incol)
        pt.taql("update $ms set "+outcol+"="+incol)

opt = optparse.OptionParser(usage="%prog [options] MS", version="%prog 3.0")
opt.add_option('-f', '--ionfactor', help='Gives an indication on how strong is the ionosphere [default: 0.01]', type='float', default=0.01)
opt.add_option('-s', '--bscalefactor', help='Gives an indication on how the smoothing varies with BL-lenght [default: 1.0]', type='float', default=1.0)
opt.add_option('-i', '--incol', help='Column name to smooth [default: DATA]', type='string', default='DATA')
opt.add_option('-o', '--outcol', help='Output column [default: SMOOTHED_DATA]', type="string", default='SMOOTHED_DATA')
opt.add_option('-w', '--weight', help='Save the newly computed WEIGHT_SPECTRUM, this action permanently modify the MS! [default: False]', action="store_true", default=False)
opt.add_option('-r', '--restore', help='If WEIGHT_SPECTRUM_ORIG exists then restore it before smoothing [default: False]', action="store_true", default=False)
opt.add_option('-b', '--nobackup', help='Do not backup the old WEIGHT_SPECTRUM in WEIGHT_SPECTRUM_ORIG [default: do backup if -w]', action="store_true", default=False)
opt.add_option('-a', '--onlyamp', help='Smooth only amplitudes [default: smooth real/imag]', action="store_true", default=False)
opt.add_option('-t', '--notime', help='Do not do smoothing in time [default: False]', action="store_true", default=False)
opt.add_option('-q', '--nofreq', help='Do not do smoothing in frequency [default: False]', action="store_true", default=False)
(options, msfile) = opt.parse_args()

if msfile == []:
    opt.print_help()
    sys.exit(0)
msfile = msfile[0]

if not os.path.exists(msfile):
    logging.error("Cannot find MS file.")
    sys.exit(1)

# open input/output MS
ms = pt.table(msfile, readonly=False, ack=False)
        
freqtab = pt.table(msfile + '/SPECTRAL_WINDOW', ack=False)
freq = freqtab.getcol('REF_FREQUENCY')[0]
freqpersample = np.mean(freqtab.getcol('RESOLUTION'))
freqtab.close()
wav = 299792458. / freq
timepersample = ms.getcell('INTERVAL',0)

# check if ms is time-ordered
times = ms.getcol('TIME_CENTROID')
if not all(np.diff(times) >= 0):
    logging.critical('This code cannot handle MS that are not time-sorted.')
    sys.exit(1)

# create column to smooth
addcol(ms, options.incol, options.outcol)

# retore WEIGHT_SPECTRUM
if 'WEIGHT_SPECTRUM_ORIG' in ms.colnames() and options.restore:
    addcol(ms, 'WEIGHT_SPECTRUM_ORIG', 'WEIGHT_SPECTRUM')
# backup WEIGHT_SPECTRUM
elif options.weight and not options.nobackup:
    addcol(ms, 'WEIGHT_SPECTRUM', 'WEIGHT_SPECTRUM_ORIG')

# iteration on antenna1
for ms_ant1 in ms.iter(["ANTENNA1"]):
    ant1 = ms_ant1.getcol('ANTENNA1')[0]
    a_ant2 = ms_ant1.getcol('ANTENNA2')
    logging.debug('Working on antenna: %s' % ant1)

    a_uvw = ms_ant1.getcol('UVW')
    a_uvw_dist = np.sqrt(a_uvw[:, 0]**2 + a_uvw[:, 1]**2 + a_uvw[:, 2]**2)
    a_data = ms_ant1.getcol(options.outcol)
    a_weights = ms_ant1.getcol('WEIGHT_SPECTRUM')
    a_flags = ms_ant1.getcol('FLAG')
 
    for ant2 in sorted(set(a_ant2)):
        if ant1 == ant2: continue # skip autocorr
        idx = np.where(a_ant2 == ant2)
        
        uvw_dist = a_uvw_dist[idx]
        data = a_data[idx]
        weights = a_weights[idx]
        flags = a_flags[idx]
   
        # compute the FWHM
        dist = np.mean(uvw_dist) / 1.e3
        if np.isnan(dist): continue # fix for missing anstennas
    
        stddev_t = options.ionfactor * (25.e3 / dist)**options.bscalefactor * (freq / 60.e6) # in sec
        stddev_t = stddev_t/timepersample # in samples
        # TODO: for freq this is hardcoded, it should be thought better 
        # However, the limitation is probably smearing here
        stddev_f = 1e6 / dist  # Hz
        stddev_f = stddev_f / freqpersample  # in samples
        logging.debug("%s - %s (dist = %.1f km) >> Time: sigma=%.1f samples (%.1f s) >> Freq: sigma=%.1f samples (%.2f MHz)" % \
                (ant1, ant2, dist, stddev_t, timepersample*stddev_t, stddev_f, freqpersample*stddev_f/1e6))

        if stddev_t == 0: continue # fix for flagged antennas
        if stddev_t < 0.5: continue # avoid very small smoothing

        flags[ np.isnan(data) ] = True # flag NaNs
        weights[flags] = 0 # set weight of flagged data to 0
        del flags
        
        # Multiply every element of the data by the weights, convolve both the scaled data and the weights, and then
        # divide the convolved data by the convolved weights (translating flagged data into weight=0). That's basically the equivalent of a
        # running weighted average with a Gaussian window function.
        
        # set bad data to 0 so nans do not propagate
        data = np.nan_to_num(data*weights)
        # smear weighted data and weights
        if options.onlyamp:
            dataAMP = np.abs(data)
            dataPH = np.angle(data)
            if not options.notime:
                dataAMP = gfilter(dataAMP, stddev_t, axis=0)
            if not options.nofreq:
                dataAMP = gfilter(dataAMP, stddev_f, axis=1)
        else:
            dataR = np.real(data)
            dataI = np.imag(data)
            if not options.notime:
                dataR = gfilter(dataR, stddev_t, axis=0)#, truncate=4.)
                dataI = gfilter(dataI, stddev_t, axis=0)#, truncate=4.)
            if not options.nofreq:
                dataR = gfilter(dataR, stddev_f, axis=1)#, truncate=4.)
                dataI = gfilter(dataI, stddev_f, axis=1)#, truncate=4.)
    
        if not options.notime:
            weights = gfilter(weights, stddev_t, axis=0)#, truncate=4.)
        if not options.nofreq:
            weights = gfilter(weights, stddev_f, axis=1)#, truncate=4.)
    
        # re-create data
        if options.onlyamp:
            data = dataAMP * ( np.cos(dataPH) + 1j*np.sin(dataPH) )
        else:
            data = (dataR + 1j * dataI)

        data[(weights != 0)] /= weights[(weights != 0)] # avoid divbyzero
    
        #print np.count_nonzero(data[~flags]), np.count_nonzero(data[flags]), 100*np.count_nonzero(data[flags])/np.count_nonzero(data)
        #print "NANs in flagged data: ", np.count_nonzero(np.isnan(data[flags]))
        #print "NANs in unflagged data: ", np.count_nonzero(np.isnan(data[~flags]))
        #print "NANs in weights: ", np.count_nonzero(np.isnan(weights))

        a_data[idx] = data
        if options.weight: a_weights[idx] = weights
    
    #logging.info('Writing %s column.' % options.outcol)
    ms_ant1.putcol(options.outcol, a_data)

    if options.weight:
        #logging.warning('Writing WEIGHT_SPECTRUM column.')
        ms_ant1.putcol('WEIGHT_SPECTRUM', a_weights)

ms.close()
logging.info("Done.")
