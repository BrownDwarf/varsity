#!/usr/bin/env python

# All of the argument parsing is done in the `parallel.py` module.

import multiprocessing
import time
import numpy as np
import Starfish
from Starfish.model import ThetaParam, PhiParam

import argparse
parser = argparse.ArgumentParser(prog="star_so.py", description="Run Starfish fitting model in single order mode with many walkers.")
parser.add_argument("--samples", type=int, default=5, help="How many samples to run?")
parser.add_argument("--incremental_save", type=int, default=100, help="How often to save incremental progress of MCMC samples.")
parser.add_argument("--resume", action="store_true", help="Continue from the last sample. If this is left off, the chain will start from your initial guess specified in config.yaml.")
args = parser.parse_args()

import os

import Starfish.grid_tools
from Starfish.spectrum import DataSpectrum, Mask, ChebyshevSpectrum
from Starfish.emulator import Emulator
import Starfish.constants as C
from Starfish.covariance import get_dense_C, make_k_func, make_k_func_region

from scipy.special import j1
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet
from astropy.stats import sigma_clip

import gc
import logging

from itertools import chain
#from collections import deque
from operator import itemgetter
import yaml
import shutil
import json

from star_base import Order as OrderBase
from star_base import SampleThetaPhi as SampleThetaPhiBase

Starfish.routdir = ""

# list of keys from 0 to (norders - 1)
order_keys = np.arange(1)
DataSpectra = [DataSpectrum.open(os.path.expandvars(file), orders=Starfish.data["orders"]) for file in Starfish.data["files"]]
# list of keys from 0 to (nspectra - 1) Used for indexing purposes.
spectra_keys = np.arange(len(DataSpectra))

#Instruments are provided as one per dataset
Instruments = [eval("Starfish.grid_tools." + inst)() for inst in Starfish.data["instruments"]]


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", filename="{}log.log".format(
    Starfish.routdir), level=logging.INFO, filemode="w", datefmt='%m/%d/%Y %I:%M:%S %p')

class Order(OrderBase):

    def CC_debugger(self, CC):
        '''
        Special debugging information for the covariance matrix decomposition.
        '''
        print('{:-^60}'.format('CC_debugger'))
        print("See https://github.com/iancze/Starfish/issues/26")
        print("Covariance matrix at a glance:")
        if (CC.diagonal().min() < 0.0):
            print("- Negative entries on the diagonal:")
            print("\t- Check sigAmp: should be positive")
            print("\t- Check uncertainty estimates: should all be positive")
        elif np.any(np.isnan(CC.diagonal())):
            print("- Covariance matrix has a NaN value on the diagonal")
        else:
            if not np.allclose(CC, CC.T):
                print("- The covariance matrix is highly asymmetric")

            #Still might have an asymmetric matrix below `allclose` threshold
            evals_CC, evecs_CC = np.linalg.eigh(CC)
            n_neg = (evals_CC < 0).sum()
            n_tot = len(evals_CC)
            print("- There are {} negative eigenvalues out of {}.".format(n_neg, n_tot))
            mark = lambda val: '>' if val < 0 else '.'

            print("Covariance matrix eigenvalues:")
            print(*["{: >6} {:{fill}>20.3e}".format(i, evals_CC[i],
                                                    fill=mark(evals_CC[i])) for i in range(10)], sep='\n')
            print('{: >15}'.format('...'))
            print(*["{: >6} {:{fill}>20.3e}".format(n_tot-10+i, evals_CC[-10+i],
                                                   fill=mark(evals_CC[-10+i])) for i in range(10)], sep='\n')
        print('{:-^60}'.format('-'))



class SampleThetaPhi(Order, SampleThetaPhiBase):
    pass #put custom behavior here


# Run the program.

model = SampleThetaPhi(debug=True)

model.initialize((0,0))

def lnlike(p):
    try:
        pars1 = ThetaParam(grid=p[0:2], vz=p[2], vsini=p[3], logOmega=p[4])
        model.update_Theta(pars1)
        # hard code npoly=3 (for fixc0 = True with npoly=4)
        pars2 = PhiParam(0, 0, True, p[5:8], p[8], p[9], p[10])
        model.update_Phi(pars2)
        lnp = model.evaluate()
        return lnp
    except C.ModelError:
        model.logger.debug("ModelError in stellar parameters, sending back -np.inf {}".format(p))
        return -np.inf

def book_keeping():
    '''Write a json blob of Starfish run metadata for use later'''
    timestamp = time.strftime('%Y%m%d%H%M')
    dict_out = {'computer_name':os.uname()[1],
    'starfish_version': __file__,
    'path_name':os.getcwd(),
    'start_time':t_start,
    'end_time':t_end,
    'elapsed_time_s': np.round(elapsed_time,1),
    'elapsed_time_hr':np.round(elapsed_time/3600.0,2),
    'timestamp':timestamp,
    'N_samples_request':nsteps,
    'N_dim':ndim,
    'N_threads':n_threads}
    fn_out = 'sf_log_'+timestamp+'.json'
    with open(fn_out, 'w') as ff:
        json.dump(dict_out, ff, indent=1)
    return 1

# Must load a user-defined prior
try:
    sourcepath_env = Starfish.config['Theta_priors']
    sourcepath = os.path.expandvars(sourcepath_env)
    with open(sourcepath, 'r') as f:
        sourcecode = f.read()
    code = compile(sourcecode, sourcepath, 'exec')
    exec(code)
    lnprior = user_defined_lnprior
    print("Using the user defined prior in {}".format(sourcepath_env))
except:
    print("Don't you want to use a user defined prior??")
    raise

# Insert the prior here
def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(p)

import emcee

start = Starfish.config["Theta"]
fname = Starfish.specfmt.format(model.spectrum_id, model.order) + "phi.json"
phi0 = PhiParam.load(fname)

from multiprocessing import Pool
import os
# THIS DOESN'T WORK, NEED TO EMPLOY IN BASH SHELL!
os.environ["OMP_NUM_THREADS"] = "1"

ndim, nwalkers = 11, 40

p0 = np.array(start["grid"] + [start["vz"], start["vsini"], start["logOmega"]] +
             phi0.cheb.tolist() + [phi0.sigAmp, phi0.logAmp, phi0.l])

p0_std = [5, 0.02, 0.5, 0.5, 0.01, 0.005, 0.005, 0.005, 0.01, 0.001, 0.5]

if args.resume:
    try:
        p0_ball = np.load("emcee_chain.npy")[:,-1,:]
    except:
        final_samples = np.load("temp_emcee_chain.npy")
        max_obs = final_samples.any(axis=(0,2)).sum()
        p0_ball = final_samples[:,max_obs-1,:]
else:
    p0_ball = emcee.utils.sample_ball(p0, p0_std, size=nwalkers)

n_threads = multiprocessing.cpu_count()
from multiprocessing import Pool
import os
os.environ["OMP_NUM_THREADS"] = "1"


print(p0_ball[0,:])
print(lnprob(p0_ball[10,:]))
#backend = emcee.backends.HDFBackend("emcee_chain.h5")
#backend.reset(nwalkers, ndim)

with Pool() as pool:
    t_start=time.strftime('%Y %b %d,%l:%M %p')
    t0 = time.time()
    nsteps = args.samples
    ninc = args.incremental_save

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
    #sampler.run_mcmc(p0_ball, nsteps=nsteps, progress=True, store=True)

    for i, (pos, lnp, state) in enumerate(sampler.sample(p0_ball, iterations=nsteps)):
        if (i+1) % ninc == 0:
            t_out = time.strftime('%Y %b %d,%l:%M %p')
            print("{0}: {1:}/{2:} = {3:.1f}%".format(t_out, i, nsteps, 100 * float(i) / nsteps))
            np.save('temp_emcee_chain.npy',sampler.chain)

np.save('emcee_chain.npy',sampler.chain)
t_end=time.strftime('%Y %b %d,%l:%M %p')
t1 = time.time()
elapsed_time = t1-t0
ret_val = book_keeping()

print("The end.")
