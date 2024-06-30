"""
Created on Thu Oct 07 23:29:49 2023
@author: blybelle
"""

"""
Overview
--------
This Python module, `imageprocess.py`, is designed for processing astronomical images, particularly focusing on the extraction of objects and calculation of their flux from FITS files. It leverages various libraries for handling FITS files, performing numerical operations, and conducting source extraction and photometry.
The core functionality is encapsulated within the `ExtractObjectsFlux` class, which initializes with a data file and optional parameters for background width, filter width, among others. It supports extensive customization through keyword arguments, allowing for tailored processing of astronomical images.

Dependencies
------------
- imgprocesslib: Custom library for basic image processing tasks.
- os, sys: Standard Python libraries for system operations.
- copy.deepcopy: For creating deep copies of objects.
- tqdm: For displaying progress bars.
- pickle: For object serialization and deserialization.
- sep: For source extraction and photometry.
- numpy (np): For numerical operations.
- astropy.io.fits: For handling FITS files.
- astropy.modeling.functional_models (models): For modeling astronomical objects.
- astropy.convolution: For image convolution operations.
- scipy.optimize.minimize: For mathematical optimization.
- matplotlib.pyplot (plt), matplotlib.colors.Normalize, matplotlib (mpl): For plotting and visualization.

Classes
-------
ExtractObjectsFlux
    A class designed for extracting objects from astronomical images and calculating their flux. It allows for significant customization through its initialization parameters and keyword arguments.

    Parameters:
    - datafile (str): Path to the FITS file containing the astronomical image.
    - bw (int): Background width for the extraction process. Default is 64.
    - fw (int): Filter width for the extraction process. Default is 3.
    - **kwargs: Additional keyword arguments for further customization.

    Supported Keyword Arguments:
    - objxlim, objylim: Limits for object extraction in the x and y dimensions.
    - bh, fh: Background and filter heights.
    - dpi: Dots per inch for plotting.
    - gain: Gain of the image sensor.
    - ylim, xlim: Limits for plotting in the y and x dimensions.
    - title: Title for the plot.
    - cmap: Colormap for the plot.
    - vmin, vmax: Minimum and maximum values for color scaling.
    - norm: Normalization for the plot.
    - detThesh: Detection threshold for object extraction.
    - convolve: Boolean indicating whether to convolve the image.
    - kernel: Kernel to use for convolution.
    - minarea: Minimum area for detected objects.
    - inBuffer: Buffer area around detected objects.
    - PLOT: Boolean indicating whether to plot the results.
    - CONVOLVE: Boolean indicating whether to perform convolution.

Usage
-----
To use this module, instantiate the `ExtractObjectsFlux` class with the path to a FITS file and any desired parameters or keyword arguments. The class provides methods (not detailed here) for extracting objects and calculating their flux, which can be called on the instantiated object.

"""


from imgprocesslib import homedir
import os
import sys
from copy import deepcopy
from tqdm import tqdm
import pickle

import sep
import numpy as np
from astropy.io import fits
import astropy.modeling.functional_models as models
from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
from multiprocessing import Pool
import multiprocessing as mp



class ExtractObjectsFlux:
    def __init__(self, datafile, bw=64, fw=3, **kwargs):
        poskwargs = ['objxlim', 'objylim', 'bh', 'fh', 'dpi', 'gain', 'ylim', 'xlim', 
                     'title', 'cmap', 'vmin', 'vmax', 'norm', 'detThesh', 'convolve',
                     'kernel', 'minarea', 'inBuffer', 'PLOT', 'CONVOLVE']
        for key in kwargs.keys():
            assert key in poskwargs, '\'{}\' not a class parameter'.format(key)

        self.bw = bw
        self.fw = fw

        if not isinstance(datafile, str):
            raise TypeError("Data type is too large for one instance; use multifileflux() for this instance.")
        else:
            self.datafile = datafile

        DEFAULTS = {
            'detThesh': 100,
            'bh': self.bw,
            'fh': self.fw,
            'objxlim': None,
            'objylim': None,
            'ylim': (None, None),
            'xlim': (None, None),
            'dpi': 300,
            'title': os.path.basename(self.datafile),
            'cmap': 'afmhot',
            'vmin': 35,
            'vmax': 95,
            'norm': Normalize(vmin=np.percentile(fits.getdata(datafile), 35), 
                              vmax=np.percentile(fits.getdata(datafile), 95)),
            'kernel': None,
            'inBuffer': 5000000,
            'CONVOLVE': False,
            }
        
        for key, default in DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))

        PLOT = kwargs.get('PLOT', False)

        try:
            data = fits.open(self.datafile)
            self.data = data.byteswap().newbyteorder()

        except:
            data = fits.getdata(self.datafile)
            self.data = data.byteswap().newbyteorder()

        if ('vmin' and 'vmax') in kwargs.keys():
            self.norm = Normalize(vmin=np.percentile(fits.getdata(datafile), self.vmin), 
                                  vmax=np.percentile(fits.getdata(datafile), self.vmax))

        if self.CONVOLVE:
            #self.data /= np.max(self.data)
            for key in ['kernel']:
                assert key in kwargs.keys(), "\'{}\' must be given if \'convolve\' is True".format(key)

            self.data = convolve(self.data, kwargs['kernel'], normalize_kernel=True)

        try:
            if isinstance(self.objxlim, np.ndarray) and isinstance(self.objylim, np.ndarray):
                self.zero_y = np.min(self.objylim)-201 if not isinstance(self.objylim, np.ndarray) else np.min(self.objylim)-201
                self.zero_x = np.min(self.objxlim)-201

                self.data = np.ascontiguousarray(np.flip(np.flip(self.data[int(self.zero_y):int(np.max(self.objylim)+201),
                                                                           int(self.zero_x):int(np.max(self.objxlim)+201)], 0), 0))
                self.objxlim_new = self.objxlim - self.zero_x
                self.objylim_new = self.objylim - self.zero_y

        except KeyError:
            pass
        
        print('Processing: %s' %(self.datafile))
        sep.set_extract_pixstack(self.inBuffer)
        self.bkg = sep.Background(self.data, bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)
        #self.data -= self.bkg
        
        print("Collecting objects...")
        self.objs = sep.extract(self.data, self.detThesh, err=self.bkg.globalrms)


    def checkBkg(self):
        bkg_image = self.bkg.back()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gcf().set_dpi(self.dpi)
        plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
        plt.colorbar()
        plt.show()


    def findFlux(self, aptR, fileNo=0, check=False):
        self.aptR = aptR
        flux, fluxerr, _ = sep.sum_circle(self.data, self.objs['x'], self.objs['y'], aptR, err=self.bkg.globalrms)
        
        fluxL =  np.zeros((len(self.objxlim) if isinstance(self.objxlim, np.ndarray) else len(self.data)))
        fluxerrL = deepcopy(fluxL)
        xL = deepcopy(fluxL)
        yL = deepcopy(fluxL)
        notfound = 0

        for tar, x, y in zip(range(len(self.objxlim) if isinstance(self.objxlim, np.ndarray) else len(self.data)), 
                                self.objxlim_new if isinstance(self.objxlim_new, np.ndarray) else self.data, 
                                self.objylim_new if isinstance(self.objylim_new, np.ndarray) else self.data):

            for i, (obj_x, obj_y) in enumerate(zip(self.objs['x'], self.objs['y'])):
                if x[0] < obj_x < x[1] and y[0] < obj_y < y[1]:
                    if check:
                        tqdm.write("{:.0f}: Object {:d} {:f} {:f}: flux = {:f} ± {:f}".format(fileNo, i, obj_x+self.zero_x, obj_y+self.zero_y, flux[i], fluxerr[i]))
                    
                    fluxL[tar] = flux[i]
                    fluxerrL[tar] = fluxerr[i]
                    xL[tar] = obj_x + self.zero_x
                    yL[tar] = obj_y + self.zero_y

                else:
                    notfound += 1
                    continue
        
        return  {'flux': fluxL, 
                 'fluxerr': fluxerrL, 
                 'x': xL,
                 'y': yL, 
                 'Objs not found': notfound
                 }
        
        
    '''
    def fits_plot_wrapper(self, argskwargs):
        args, kwargs = argskwargs
        args = str(args)
        print(args)
        return self.fits_plot_data(args, **kwargs)
    '''

        
    def fits_plot_data(self):
        """
        Plots the image of the FITS file and some useful information is provided if asked for with Boolean
        
        Parameters:
        -----------
        image_path: String
            File path of the image in .FITS format sent for plotting
        
        colour_map: String
            Desired colour map entry for use in plot
            To see more color maps:
            https://matplotlib.org/stable/tutorials/colors/colormaps.html

        Returns:
        --------
        Image_plot(plt.show):
            Image of the FITS file
        """

        plt.ion() # Turns on interative mode

        pltkwargs = dict()        
        pltkwargs.update({'norm': self.norm, 
                          'cmap': self.cmap
                          })

        plt.clf()
        plt.title(self.title)
        plt.imshow(self.data if isinstance(self.objxlim, np.ndarray) else fits.getdata(self.datafile), **pltkwargs)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gcf().set_dpi(self.dpi)
        plt.gca().invert_yaxis()
        plt.xlabel("x-pixels")
        plt.ylabel("y-pixels")
        #plt.colorbar()
        
        self.figure = plt.figure(1)
        plt.tight_layout()
        return self.figure


    def plot_detected_sources(self, size=None):
        """
        Plots detected sources from sep.extract in the initialisation of this class.

        Parameters
        ----------
        size : int or float, optional
             Radius of aperture size in pixels.
        
        Returns
        -------
        `ExtractObjectsFlux.figure`
        """
        
        self.fits_plot_data()
        #self.figure
        #plt.figure(pltkwargs['i'] if 'i' in pltkwargs.keys() else None)
        plt.scatter(self.objs['x'], self.objs['y'], marker = 'x', c='#2BFF0A')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        
        if size is not None:
            size = int(size)
        elif 'aptR' in ExtractObjectsFlux.__dict__:
            size = self.aptR

        if size is not None:
            fig = plt.gcf()
            ax = fig.gca()
            for x, y in zip(self.objs['x'], self.objs['y']):
                circ = plt.Circle((x, y), size, color='aqua', fill=False)
                ax.add_patch(circ)

        return plt.gcf()
    

    #def resize():
    #    if CONVOLVE:


def convolve_moffat(datafile_target, datafiles, apers, Aper_target=10, bw=64, fw=3, change_params=None, SAVE=True, **kwargs):
    len_files = len(datafiles)
    for datafile in datafiles:
        if datafile_target == datafile:
            len_files -= 1

    if len_files == 0:
        raise RuntimeError("The target file and the input file match and therefore cannot be processed for convolution.")
    
    try:
        bounds_gamma = kwargs['gamma_range']
    except KeyError:
        try:
            if isinstance(kwargs['gamma'], list):
                raise TypeError("Found as type \'list\'. Use key \'gamma_range\' in this instance")
            else:
                gamma = kwargs['gamma']
        except KeyError:
            if 'gamma' in change_params.keys():
                pass
            else:
                gamma = 2

    try:
        bounds_alpha = kwargs['alpha_range']
    except KeyError:
        try:
            if isinstance(kwargs['alpha'], list):
                raise TypeError("Found as type \'list\'. Use key \'alpha_range\' in this instance")
            else:
                alpha = kwargs['alpha']
        except KeyError:
            if 'alpha' in change_params.keys():
                pass
            else:
                alpha = 2

    for key in ['gamma_range', 'alpha_range', 'gamma', 'alpha']:
        try:
           del kwargs[key]
        except KeyError:
            pass

    print('Processing traget file')
    target_data = aperFindFlux(datafile_target, apers, bw, fw, fileNo=0, OWN=False, **kwargs)
                            #findflux(R, fileNo=0, check=False))

    for i, data in enumerate(target_data):
        if data['Aperture'] == Aper_target:
            target_i = i
            target_flux = data['flux'][0]

        '''if data['Aperture'] == 5:
            target_i_5 = i
            target_flux_5 = data['flux'][0]'''
            
    print('Processing: %d files to convolve' %(len_files))
    
    conv_data = []
    notfound = 0

    kwargs['convolve'] = True
    #kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

    with tqdm(total=len_files, file=sys.stdout) as pbar:
        fileNo = 0
        for file in datafiles:
            if file == datafile_target:
                continue
            
            for key in change_params:                
                if 'gamma' == key:
                    gamma = change_params[key][fileNo]

                elif 'alpha' == key:
                    alpha = change_params[key][fileNo]
                
                else:
                    kwargs[key] = change_params[key][fileNo]

            fileNo += 1

            kwargs['kernel'] = Moffat2DKernel(gamma=bounds_gamma[0] if 'bounds_gamma' in locals() else gamma,
                                              alpha=bounds_alpha[0] if 'bounds_alpha' in locals() else alpha)

            data_list_conv = aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs)

            if 'bounds_gamma' in locals():
                while np.allclose(data_list_conv[target_i]['flux'][0], target_flux, rtol=1e-3):
                    for alpha in bounds_alpha[1:]:
                        for gamma in bounds_gamma[1:]:
                            kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)
                            data_list_conv = aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs)
            
            noObjs = data_list_conv[-1]['Objs not found']
            for d in data_list_conv:
                del d['Objs not found']
            notfound += noObjs
            
            conv_data.append({
            'file': os.path.basename(file),
            'results': data_list_conv,
            'Objs not found': noObjs
            })

            pbar.update(1)
    
    print('FINISHED')

    if notfound != 0:
        print('OBJECTS NOT FOUND:', notfound)

    if SAVE:

        if not os.path.exists(homedir+'Output'):
            os.makedirs(homedir+'Output')
        with open(homedir+'Output/convolved_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(target_data[0]['x'])), 
                                                                       int(np.average(target_data[0]['y'])), 
                                                                       apers[0], apers[-1]), 'wb+') as file:
            
            pickle.dump({'target': target_data, 'convolved': conv_data}, file)
        
        print('DATA WRITTEN TO FILE: convolved_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(target_data[0]['x'])), 
                                                                         int(np.average(target_data[0]['y'])), 
                                                                         apers[0], apers[-1]))
        
    return {'target': target_data, 'convolved': conv_data}
    

def aperFindFlux(datafile, AptR, bw=64, fw=3, fileNo=0, OWN=True, **kwargs):
    ef = ExtractObjectsFlux(datafile, bw, fw, **kwargs)
    resDataFile = []

    for R in AptR:
        data = ef.findFlux(R, fileNo)
        data['Aperture'] = R
        resDataFile.append(data)

    if OWN:
        if not os.path.exists(homedir+'Output'):
            os.makedirs(homedir+'Output')
        with open(homedir+'Output/{4}_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
                                                                           int(np.average(data[0]['y'])), 
                                                                           AptR[0], AptR[-1], 
                                                                           os.path.basename(datafile).rsplit('_', 2)[0].rsplit('_', 1)[1]), 'wb') as file:
            pickle.dump({'notfound': data[-1]['Objs not found'], 'results': resDataFile}, file)

        print('DATA WRITTEN TO FILE: {4}_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
                                                                             int(np.average(data[0]['y'])), 
                                                                             AptR[0], AptR[-1], 
                                                                             os.path.basename(datafile).rsplit('_', 2)[0].rsplit('_', 1)[1]))

    return resDataFile


def multiAperFileFlux(files, AptR, change_params=None, save=True, **kwargs):
    if not isinstance(change_params, dict):
        raise TypeError("Expected type \'dict\'. Remove key \'change_params\', if parameters do not change.")
    
    ResData = []
    notfound = 0

    print('Processing: %d files' %(len(files)))

    with tqdm(total=len(files), file=sys.stdout) as pbar:
        for fileNo, datafile in enumerate(files):
            for key in change_params:
                kwargs[key] = change_params[key][fileNo]

            data = aperFindFlux(datafile, AptR, OWN=False,**kwargs)

            noObjs = data[-1]['Objs not found']
            for d in data:
                del d['Objs not found']
            notfound += noObjs
            
            ResData.append({
            'file': os.path.basename(datafile),
            'results': data,
            'Objs not found': noObjs
            })

            #pbar.set_description('Processed: %d' % (1 + fileNo))
            pbar.update(1)

    print('FINISHED')
    if save:
        if not os.path.exists(homedir+'Output'):
            os.makedirs(homedir+'Output')
        with open(homedir+'Output/{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
                                                                       int(np.average(data[0]['y'])), 
                                                                       AptR[0], AptR[-1]), 'wb+') as file:
            pickle.dump(ResData, file)
        print('DATA WRITTEN TO FILE: {0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
                                                                         int(np.average(data[0]['y'])), 
                                                                         AptR[0], AptR[-1]))

    if notfound != 0:
        print('OBJECTS NOT FOUND:', notfound)

    return ResData

#


#


#
def make_chunks(a, no_of_chunks=2):
    k, m = divmod(len(a), no_of_chunks)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(no_of_chunks)]


def worker_wrapper(argskwargs):
    print(argskwargs)
    args, kwargs = argskwargs
    return convolve_once(*args, **kwargs)


def convolve_once(datafile_target, datafiles, target_i, target_flux, apers, bw=64, fw=3, gamma=2, alpha=1, change_params=None, **kwargs):
    conv_data = []
    for fileNo, file in enumerate(datafiles):
        kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

        if file == datafile_target:
            continue

        for key in change_params:
            kwargs[key] = change_params[key][fileNo]

        res = minimize(conv_min_, [alpha, gamma], args=(target_i, target_flux, file, apers, bw, fw, kwargs), method='Nelder-Mead', options={'return_all': True})

        print(res.x)

        kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha='some')

        conv_data.append(aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs))

    return conv_data


def conv_min_(alpha_gamma, target_i, target_flux, file, apers, bw, fw, kwargs):
    alpha, gamma = alpha_gamma
    kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)
    data_list_conv = aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs)
    diff = np.abs(data_list_conv[target_i]['flux'][0] - target_flux)
    print(diff)
    
    return diff


def convolve_moffat_(datafile_target, datafiles, apers, Aper_target=10, bw=64, fw=3, gamma=2, alpha=1, change_params=None, SAVE=True, **kwargs):
    len_files = len(datafiles)
    for datafile in datafiles:
        if datafile_target == datafile:
            len_files -= 1

    try:
        PARALLEL = kwargs['PARALLEL']
        del kwargs['PARALLEL']
    except KeyError:
        PARALLEL = False
    
    allargskwargs = []

    if PARALLEL:
        try:
            no_of_chunks = kwargs['chunks']
        except KeyError:
            no_of_chunks = 3    
        
        try:
            processors = kwargs['processors']
        except KeyError:
            processors = mp.cpu_count() - 2
        
        datafiles_args = make_chunks(datafiles, no_of_chunks)

        params_split = {}
        for key in change_params.keys():
            params_split[key] = make_chunks(change_params[key], no_of_chunks)

        del kwargs['chunks']

    print('Processing traget file')
    target_data = aperFindFlux(datafile_target, apers, bw, fw, fileNo=0, OWN=False, **kwargs)
                            #findflux(R, fileNo=0, check=False))

    for i, data in enumerate(target_data):
        if data['Aperture'] == Aper_target:
            target_i = i
            target_flux = data['flux'][0]

    print('Processing: %d files to convolve' %(len_files))
    
    kwargs['convolve'] = True

    if PARALLEL:
        change_params_args = {}
        for i, datafiles in enumerate(datafiles_args):
            for key in params_split.keys():

                change_params_args[key] = params_split[key][i]

            allargskwargs.append([(datafile_target, datafiles, target_i, target_flux, apers, bw, fw, gamma, alpha, change_params_args), kwargs.copy()])

        with Pool(processes=processors) as pool:
            for res in tqdm(pool.imap(worker_wrapper, allargskwargs), total=len_files, file=sys.stdout):
                print(res.x)

                kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=res.x)
                conv_data.append(aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs))
    
    else:
        conv_data = []
        with tqdm(total=len_files, file=sys.stdout) as pbar:
                #allargskwargs.append([(datafile_target, datafiles, target_i, target_flux, apers, bw, fw, gamma, alpha, change_params), kwargs.copy()])
                #sres = worker_wrapper(allargskwargs)
                #pbar.refresh()
                #print(res.x)

                #kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=res.x)
                #conv_data.append(aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs))

                for fileNo, file in enumerate(datafiles):
                    kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

                    if file == datafile_target:
                        continue

                    for key in change_params:
                        kwargs[key] = change_params[key][fileNo]

                    pbar.refresh()
                    res = minimize(conv_min_, [alpha, gamma], args=(target_i, target_flux, file, apers, bw, fw, kwargs), options={'return_all': True}) # method='Nelder-Mead'
                    
                    pbar.refresh()
                    print(res.x)

                    kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha='some')

                    conv_data.append(aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs))
                    pbar.update(1)

    print('FINISHED')

    if SAVE:
        if not os.path.exists(homedir+'Output'):
            os.makedirs(homedir+'Output')
        with open(homedir+'Output/convolved_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(target_data[0]['x'])), 
                                                                       int(np.average(target_data[0]['y'])), 
                                                                       apers[0], apers[-1]), 'wb+') as file:
            
            pickle.dump({'target': target_data, 'convolved': conv_data}, file)
        
        print('DATA WRITTEN TO FILE: convolved_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(target_data[0]['x'])), 
                                                                         int(np.average(target_data[0]['y'])), 
                                                                         apers[0], apers[-1]))
        
    return {'target': target_data, 'convolved': conv_data}

    
                        #np.allclose(data_list_conv[target_i_5]['flux'][0], target_flux_5, rtol=1e-3)): #1*10**-(len(str(target_flux))-1
                        #while data_list_conv[target_i]['flux'][0] != target_flux: