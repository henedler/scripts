#!/usr/bin/env python

# Simple lib to estimate plasma ageing

import multiprocessing as mp
# from astropy.constants import c, m_e, sigma_T, mu0, e, eps0
import astropy.units as u
from scipy import special, integrate, interpolate
import logging as log
import numpy as np


c = 299792458.0 # m/s
m_e = 9.1093837015e-31 # kg
sigma_T = 6.6524587321e-29 # m2
mu0 = 1.25663706212e-06 # N A-2
e = 1.602176634e-19 # Coulomb
eps0 = 8.8541878128e-12 # F/m

def nu_c(E, B, alpha):
    # Critical frequency
    return (3*(E / (c ** 2 * m_e)) ** 2 * e * B * np.sin(alpha) / (4 * np.pi * m_e)) # Hardcastle and Longair 3/4, Hardwood 6/4...

def F_accurate(x):
    # numerical integral defining F(x). Use this to calculate lookup-table.
    # TODO: epsrel
    return x*integrate.quad(lambda z: special.kv(5./3., z), x, np.inf, limit = 2000)[0]

def create_F_lookup():
    xvals = np.logspace(-4,1.4,1000)
    with mp.Pool() as p:
        results = p.map(F_accurate, xvals)
    np.save(__file__.replace('lib_ageing.py','')+'lib_ageing_data/F(X)_lookup.npy', np.array([xvals, results]))

def n_e(E, iidx, N0, B, t):
    # C = B**2*(4*sigma_T/(6*m_e**2*c**3*mu0))
    # Hardcastle 2013: if  E*C*t > 1: Ne should be zero. This case can happen for large oit
    beta = B**2*E*t*(4*sigma_T/(6*m_e**2*c**3*mu0)) # Harwood 2013 has nu_c**3 instead of c. But this can't be right
    return N0*E**(-2*iidx+1)*((1-beta)**((2*iidx-1)-2)) # TODO: # Harwod paper has 2*iidx+1-2 ?? wrong in paper?


def S_model(nu, B, iidx, N0, t):
    # [nu] = 1/s
    # [B] = T
    # [t] in Myr
    t *= 1e6*3.154e7 # Myrs to seconds

    C0 = 3**0.5*e**3*B/(8*np.pi*eps0*c*m_e)
    E_min, E_max = 0.5e6*1.60218e-19, 1.e11*1.60218e-19 # eV, TODO: units...

    # use lookup table for F(x)
    with open(__file__.replace('lib_ageing.py','') + 'lib_ageing_data/F(X)_lookup.npy', 'rb') as f:
        xF_dat = np.load(f)
    F_interp = interpolate.interp1d(xF_dat[0], xF_dat[1], assume_sorted=True, bounds_error=True)

    def F(x):
        # Mourad Fouka1and Saad Ouichaoui, 2013
        if x < 1e-4:
            return np.pi*2**(5/3)/(special.gamma(1/3)*np.sqrt(3))*x**(1/3)
        elif x > 25.:
            return 0.
        else:
            return F_interp(x)

    def integrand(E, alpha):
        # print(f'{nu_c(E,B,alpha)/1e6:.4f}')
        return (F(nu/nu_c(E,B,alpha))*0.5*np.sin(alpha)**2*n_e(E, iidx, N0, B, t))

    return C0*integrate.dblquad(integrand,0, np.pi, E_min, E_max)[0]


def S_here(nu):
    return S_model(nu, 2.3e-10, 0.75, 1000, 0)


def plot_S_model():
    nu_range = np.logspace(6,12,60)
    with mp.Pool() as p:
        res = p.map(S_here, nu_range)
    import matplotlib.pyplot as plt
    from agnpy.emission_regions import Blob
    from agnpy.synchrotron import Synchrotron
    blob = Blob(z=0.15, B = 0.5e-5*u.gauss, spectrum_dict = {"type": "PowerLaw", "parameters": {"p": 2.5,"gamma_min": 2,"gamma_max": 1e7}})
    synch = Synchrotron(blob)
    sed = synch.sed_flux(nu_range*u.Hz)

    PL = (nu_range**-0.75)
    PL /= (PL[-1]/sed.value[-1])
    res /= (res[-1]/sed.value[-1])

    plt.close()
    plt.plot(nu_range, res, label='S')
    plt.plot(nu_range, PL, label='powerlaw')
    plt.plot(nu_range, sed.value, label='SED')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('S')
    # plt.xlim([np.min(nu_range), np.max(nu_range)])
    # plt.ylim([np.min(res), 1.05*np.max(res)])
    plt.legend()
    plt.savefig(__file__.replace('lib_ageing.py','')+'lib_ageing_data/synch_vs_nu.png')

