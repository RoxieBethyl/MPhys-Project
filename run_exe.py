import os

import numpy as np
import ccdproc as ccdp
import pandas as pd
from astropy.convolution import Moffat2DKernel
import matplotlib.pyplot as plt

import imgprocesslib.imageprocess as ip
from imgprocesslib import homedir, ddir
import imgprocesslib.calibrationcpu as clb
import imgprocesslib.makecatologuecpu as mkcat
from astropy.stats import mad_std as mad


jades_files_path = ccdp.ImageFileCollection(ddir).files_filtered(include_path=True)
aper_size = 5.83

def fn1():
    # Create output directory
    if not os.path.exists(homedir + '/Output'):
        os.makedirs(homedir + '/Output')

    i = 8
    gammas = [3.1, 3.22, 3.225, 3.325, 3.325, 3.4, 2.665, 2.1, 1.8, 1.5, 1.8, 3.25, 1.4, 1., 3.2, 3.3, 3.3, 2.1]
    alphas = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.2]

    params = {#'dpi': 200,
            #'bins': 50,
            'detThesh': 2,
            #'step_size': 1,
            'tolerance': 1e-1,
            'kernel': Moffat2DKernel(gamma=gammas[i], alpha=alphas[i]),
            'SAVE': True,
            'CONVOLVE': True,
            'filetype': ('txt', 'pkl'),
            }

    c_params = {
        'gamma': gammas,
        'alpha': alphas,
    }

    mkcat.MakeCatologue(jades_files_path[i], jades_files_path, aper_size, change_kwargs=c_params, **params)


def fn2():
    i = 8

    params = {
        'dpi': 5000,
        #'objxlim': StarLims[:, 0],
        #'objylim': StarLims[:, 1],
        #'xlim':(5500, 19500), 
        #'ylim':(2750, 18000),
        'detThesh': 2,
        #'block': False,
        'vmin': 35,
        'vmax': 97,
        'kernel': Moffat2DKernel(gamma=1.8, alpha=2.2),
        'CONVOLVE': True,
        #'cmap': cmaps.oranges_dark,
        #'checkBkg': True
    }

    ef = ip.ExtractObjectsFlux(jades_files_path[i], bw=256, fw=4, **params)
    file = homedir + 'Output/convolved_photometric_catologue_5.83_tol_0.1_detThest_2.pkl'
    data = pd.DataFrame(pd.read_pickle(file))

    plt.figure(tight_layout=True)
    ef.fits_plot_data()
    plt.scatter(np.array(data['x-position']), np.array(data['y-position']), s=0.05, marker='.', c='#2BFF0A') 
    plt.ylim(2500, 18000)
    plt.xlim(5500, 19500)
    plt.savefig(homedir + f"Images/{file.rsplit('/', 1)[-1].rsplit('.', 1)[0]}.png", dpi=2000)


def fn3():
    gammas = [3.1, 3.22, 3.225, 3.325, 3.325, 3.4, 2.665, 2.1, 1.8, 1.5, 1.8, 3.25, 1.4, 1., 3.2, 3.3, 3.3, 2.1]
    alphas = [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.2]

    params = {
        #'dpi': 200,
        #'bins': 50,
        'detThesh': 2,
        'step_size': 1,
        'tolerance': 1e-1,
        'SAVE': True,
        'filetype': ('txt', 'pkl'),
        'CONVOLVE': False,   
        }

    c_params = {
        #'detThesh': d_array,
        'gamma': gammas[16:],
        'alpha': alphas[16:],
    }

    clb.init_SkyCalibration(jades_files_path[17:], aper_size, change_kwargs=c_params, **params)

fn3()
