# This code runs the MCMC simulation to calculate the best fit parameters for the disk. It uses the logprob function from logprob_parallel.py.

import numpy as np

import astropy.units as u

import dsharp_helper as dh
from imgcube import imagecube
import emcee

from multiprocessing import Pool

import pickle
from copy import deepcopy

from logprob_parallel import logprob

# Getting the observed mm profile which is used as an input
disk = 'IMLup'
fname = dh.get_datafile(disk)
PA = dh.sources.loc[disk]['PA']
inc = dh.sources.loc[disk]['inc']
distance = dh.sources.loc[disk]['distance [pc]']

data = imagecube(fname, clip=2.5)

x_arcsec, y, dy = data.radial_profile(inc=inc, PA=PA)


profile = (y * u.Jy / data.beam_area_str).cgs.value
profile_err = (dy * u.Jy / data.beam_area_str).cgs.value

# defining number of walkers
nwalkers = 25
ndim     = 7

# setting the priors for some parameters instead of letting them be uniform randoms between (0.1)
sigma_coeff_0   = 10**((np.random.rand(nwalkers)-0.5)*4)
others_0        = np.random.rand(ndim-3,nwalkers)
d2g_coeff_0     = (np.random.rand(nwalkers)+0.5) / 100
d2g_exp_0       = (np.random.rand(nwalkers)-0.5) 

# the input matrix of priors
p0 = np.vstack((sigma_coeff_0,others_0, d2g_coeff_0, d2g_exp_0)).T


print('step1')

# Parallelizing the simluation and running it for 250 iterations
with Pool(processes=100) as pool:
    sampler1 = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=[profile, profile_err, x_arcsec], pool=pool)
    sampler1.run_mcmc(p0, 250)

print(sampler1.iteration)    

print('step2')
sampler2 = deepcopy(sampler1)
sampler2.log_prob_fn = None
with open('sampler.pickle', 'wb') as fid:
    pickle.dump(sampler2, fid)    
'''    
print('step3')
new_p0       = sampler1.chain[::,9][sampler1.lnprobability[:,9]>-9e4]
new_nwalkers = len(new_p0)

filename = "tutorial.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)



max_n = 250

print('step4')
print(new_nwalkers)


new_p0       = (np.tile(new_p0.T,int(np.ceil(20/new_nwalkers))).T)
new_nwalkers = int(np.ceil(20/new_nwalkers)*new_nwalkers)


with Pool(processes=80) as pool:
    sampler = emcee.EnsembleSampler(new_nwalkers, ndim, logprob, args=[profile, profile_err, x_arcsec], pool=pool)
    sampler.run_mcmc(new_p0 , max_n)
    
    

# state = sampler.run_mcmc(p0 , 120)
# sampler.reset()
# sampler.run_mcmc(state, 100)

sampler2 = deepcopy(sampler)
sampler2.log_prob_fn = None
with open('sampler.pickle', 'wb') as fid:
    pickle.dump(sampler2, fid)
'''

#np.save('logprobability.npy', sampler.lnprobability)

#np.save('chains.npy', sampler.get_chain())
