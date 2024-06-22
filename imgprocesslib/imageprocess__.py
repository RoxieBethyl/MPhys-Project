"""
Created on Thu Oct 07 23:29:49 2023
@author: blybelle
"""

import os
import sys
from copy import deepcopy
from tqdm import tqdm
import torch
from torch.cuda import FloatTensor

from imgprocesslib import homedir
import sep
import pickle
import numpy as np
from astropy.io import fits
import astropy.modeling.functional_models as models
from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from scipy.optimize import minimize
#import colormaps as cmaps
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'
from multiprocessing import Pool, TimeoutError
import multiprocessing as mp


class ExtractObjectsFlux:
    def __init__(self, datafile, bw=64, fw=3, **kwargs):
        """
        Extracts objects from a FITS file and finds the flux of the objects in the file.

        Parameters
        ----------
        datafile: str
            File path of the image in .FITS format sent for plotting.
        bw: int, optional
            Background width in pixels. The default is 64.
        fw: int, optional
            Background height in pixels. The default is 3.
        **kwargs: dict
            Optional arguments for the class.
        
        Optional Parameters
        -------------------
        objxlim: tuple, optional
            Tuple of the x-axis limits of the object in pixels. The default is (None, None).
        objylim: tuple, optional
            Tuple of the y-axis limits of the object in pixels. The default is (None, None).
        xlim: tuple, optional
            Tuple of the x-axis limits of the image in pixels. The default is (None, None).
        ylim: tuple, optional
            Tuple of the y-axis limits of the image in pixels. The default is (None, None).
        dpi: int, optional
            Dots per inch of the image. The default is 500.
        title: str, optional
            Title of the image. The default is os.path.basename(self.datafile).
        cmap: str, optional   
            Colour map of the image. The default is 'afmhot'.
        vmin: int, optional
            Lower percentile of the image. The default is 35.
        vmax: int, optional
            Upper percentile of the image. The default is 95.
        norm: matplotlib.colors.Normalize, optional
            Normalisation of the image. The default is Normalize(vmin=np.percentile(fits.getdata(self.datafile), self.vmin), vmax=np.percentile(fits.getdata(self.datafile), self.vmax)).
        detThesh: int, optional
            Detection threshold of the image. The default is 10.
        CONVOLVE: bool, optional 
            Boolean to convolve the image. The default is False.
        
        Returns
        -------
        None.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        poskwargs = ['objxlim', 'objylim', 'bh', 'fh', 'dpi', 'gain', 'ylim', 'xlim', 
                     'title', 'cmap', 'vmin', 'vmax', 'norm', 'detThesh', 'convolve',
                     'kernel', 'PLOT', 'SAVE', 'aptR', 'gamma', 'alpha']
        for key in kwargs.keys():
            assert key in poskwargs, '\'{}\' not a class parameter'.format(key)

        self.bw = bw
        self.fw = fw

        if not isinstance(datafile, str):
            raise TypeError("Data type is too large for one instance; use multifileflux() for this instance.")
        else:
            self.datafile = datafile
        
        DEFAULTS = {
            'detThesh': 10,
            'bh': self.bw,
            'fh': self.fw,
            'datafile': datafile,
            'objxlim': (None, None),
            'objylim': (None, None),
            'xlim': (None, None),
            'ylim': (None, None),
            'dpi': 500,
            'title': os.path.basename(self.datafile),
            'cmap': 'afmhot',
            'vmin': 35,
            'vmax': 95,
            'norm': Normalize(vmin=np.percentile(fits.getdata(self.datafile), self.vmin),
                                vmax=np.percentile(fits.getdata(self.datafile), self.vmax)),
            'aptR': 10,
            }

        for key, default in DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))
        
        CONVOLVE = kwargs.get('SAVE', False)
        #DEBUG = kwargs.get('DEBUG', False)
        SAVE = kwargs.get('SAVE', True)
        gamma = kwargs.get('gamma', 2)
        alpha = kwargs.get('alpha', 2)
        
        kernel = Moffat2DKernel(gamma=gamma,
                                alpha=alpha)
        #PLOT = kwargs.get('SAVE', False)

        try:
            del kwargs['CONVOLVE'], kwargs['kernel'], kwargs['DEBUG'] #, kwargs['PLOT']
        except KeyError:
            pass

        try:
            self.data = torch.from_numpy(fits.open(self.datafile).byteswap().newbyteorder(), dtype=torch.float32).to(self.device)
        except:
            self.data = torch.tensor(fits.getdata(self.datafile).byteswap().newbyteorder(), dtype=torch.float32).to(self.device)

        try:
            if isinstance(self.objxlim, np.ndarray) and isinstance(self.objylim, np.ndarray):
                self.zero_y = np.min(self.objylim)-201
                self.zero_x = np.min(self.objxlim)-201

                self.data = np.ascontiguousarray(np.flip(np.flip(self.data[int(self.zero_y):int(np.max(self.objylim)+201),
                                                                           int(self.zero_x):int(np.max(self.objxlim)+201)], 0), 0))
                self.objxlim_new = self.objxlim - self.zero_x
                self.objylim_new = self.objylim - self.zero_y
        except KeyError:
            pass

        self.bkg = torch.tensor(sep.Background(self.data.cpu().numpy(), bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)).to(self.device)
        #self.data -= torch.from_numpy(self.bkg).cuda()
        self.objs = torch.tensor(sep.extract(self.data.cpu().numpy(), self.detThesh, err=self.bkg.cpu().numpy().globalrms)).to(self.device)

        if CONVOLVE:
            if isinstance(kernel, None):
                assert key in kwargs.keys(), "\'{}\' must be given if \'convolve\' is True".format(key)

            self.data = convolve(self.data, kernel, normalize_kernel=True)

        self.flux, _ , _ = sep.sum_circle(self.data.cpu().numpy(), self.objs['x'].cpu().numpy(), self.objs['y'].cpu().numpy(), self.aptR, err=self.bkg.cpu().numpy().globalrms)

        # fluxL = torch.zeros((len(self.objxlim) if isinstance(self.objxlim, np.ndarray) else len(self.data.cpu().numpy())), device=self.device)
        # xL = torch.zeros_like(fluxL)
        # yL = torch.zeros_like(fluxL)

        x = self.objxlim_new if isinstance(self.objxlim_new, np.ndarray) else self.data.cpu().numpy()
        y = self.objylim_new if isinstance(self.objylim_new, np.ndarray) else self.data.cpu().numpy()
        self.mask = torch.tensor((x[0] < self.objs['x'].cpu().numpy() < x[1] and y[0] < self.objs['y'].cpu().numpy() < y[1])).to(self.device)

        self.flux = self.flux[self.mask.cpu().numpy()]
        self.objs['x'] = self.objs['x'].cpu().numpy()[self.mask.cpu().numpy()] + self.zero_x
        self.objs['y'] = self.objs['y'].cpu().numpy()[self.mask.cpu().numpy()] + self.zero_y

        self.notfound = len(~self.mask.cpu().numpy())
        
        if SAVE:
            self.save(CONVOLVE=CONVOLVE)

        # for tar, x, y in zip(range(len(self.objxlim) if isinstance(self.objxlim, np.ndarray) else len(self.data.cpu().numpy())), 
        #                         self.objxlim_new if isinstance(self.objxlim_new, np.ndarray) else self.data.cpu().numpy(), 
        #                         self.objylim_new if isinstance(self.objylim_new, np.ndarray) else self.data.cpu().numpy()):

        #     for i, (obj_x, obj_y) in enumerate(zip(self.objs['x'], self.objs['y'])):
        #         if x[0] < obj_x < x[1] and y[0] < obj_y < y[1]:
        #             if DEBUG:
        #                 tqdm.write("{:.0f}: Object {:d} {:f} {:f}: flux = {:f} ± {:f}".format(fileNo, i, obj_x+self.zero_x, obj_y+self.zero_y, flux[i], fluxerr[i]))
                    
        #             fluxL[tar] = flux[i]
        #             fluxerrL[tar] = fluxerr[i]
        #             xL[tar] = obj_x + self.zero_x
        #             yL[tar] = obj_y + self.zero_y

        #         else:
        #             notfound += 1
        #             continue
        

    def save(self, CONVOLVE=False):
        # return  {'flux': self.fluxL, 
        #              #'fluxerr': fluxerrL, 
        #              'x': self.objs['x'],
        #              'y': self.objs['y'], 
        #              'Objs not found': len(~mask)
        #              }
            

        output_dir = os.path.join(homedir, 'Output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        word = "convolved_" if CONVOLVE else ""
        filename = f"{word}{os.path.basename(self.datafile).rsplit('_', 2)[0].rsplit('_', 1)[1]}_apers({self.aptR})"
        if len(self.flux) == 1:
            filename += f"_{int(np.average(self.objs['x']))}_{int(np.average(self.objs['y']))}"
                                                                
        filepath = os.path.join(output_dir, filename)
        

        write_data = {'flux': self.flux,
                        'x': self.objs['x'],
                        'y': self.objs['y'],
                        'Objs not found': self.notfound,
                        'Aperture': self.aptR,
                        }
                                
        with open(filepath, 'wb+') as file:
            pickle.dump(write_data, file)

        print(f'DATA WRITTEN TO FILE: {filename}')   


    
    def checkBkg(self):
        bkg_image = self.bkg.back()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gcf().set_dpi(self.dpi)
        plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
        plt.colorbar()
        plt.show()

        
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

        try:
            plt.imshow(self.data.cpu().numpy() if isinstance(self.data, np.ndarray) else fits.getdata(self.datafile), **pltkwargs)
        except Exception as e:
            print(f"Error displaying image: {e}")
            
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gcf().set_dpi(self.dpi)
        plt.gca().invert_yaxis()
        plt.xlabel("x-pixels")
        plt.ylabel("y-pixels")
        #plt.colorbar()
        self.figure = plt.figure(1)
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
        plt.scatter(self.objs['x'].cpu().numpy(), self.objs['y'].cpu().numpy(), marker = 'x', c='#2BFF0A')
        
        if size is not None:
            size = int(size)
        elif 'aptR' in ExtractObjectsFlux.__dict__:
            size = self.aptR

        if size is not None:
            fig = plt.gcf()
            ax = fig.gca()
            for x, y in zip(self.objs['x'].cpu().numpy(), self.objs['y'].cpu().numpy()):
                circ = plt.Circle((x, y), size, color='yellow', fill=False)
                ax.add_patch(circ)

        return plt.gcf()


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
        kwargs.pop(key, None)
    
    if kwargs['CONVOLVE'] == False:
        kwargs['CONVOLVE'] = True

    print('Processing traget file')
    
    target_data = multiAperFileFlux(datafile_target, change_params=None, OWN=True, **kwargs)

    for i, (flux, R) in enumerate(target_data.flux, target_data.aptR):
        if R == Aper_target:
            target_i = i
            target_flux = flux
            break

    print('Processing: %d files to convolve' %(len_files))
    
    data_list_conv = []
    notfound = 0

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

            data_list_conv.append(ExtractObjectsFlux(datafile, bw=bw, fw=fw, **kwargs))

            if 'bounds_gamma' in locals():
                while np.allclose(data_list_conv[target_i]['flux'][0], target_flux, rtol=1e-3):
                    for alpha in bounds_alpha[1:]:
                        for gamma in bounds_gamma[1:]:
                            kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)
                            data_list_conv = aperFindFlux(file, apers, bw, fw, fileNo=0, OWN=False, **kwargs)
            
            noObjs = data_list_conv[-1].notfound
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
        save(target_data, conv_data, apers)

    return {'target': target_data, 'convolved': conv_data}




def aperFindFlux(datafile, AptR, bw=64, fw=3, fileNo=0, OWN=True, **kwargs):
    pass


    # if OWN:
    #     if not os.path.exists(homedir+'Output'):
    #         os.makedirs(homedir+'Output')
    #     with open(homedir+'Output/{4}_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
    #                                                                        int(np.average(data[0]['y'])), 
    #                                                                        AptR[0], AptR[-1], 
    #                                                                        os.path.basename(datafile).rsplit('_', 2)[0].rsplit('_', 1)[1]), 'wb') as file:
    #         pickle.dump({'notfound': data[-1]['Objs not found'], 'results': resDataFile}, file)

    #     print('DATA WRITTEN TO FILE: {4}_{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
    #                                                                          int(np.average(data[0]['y'])), 
    #                                                                          AptR[0], AptR[-1], 
    #                                                                          os.path.basename(datafile).rsplit('_', 2)[0].rsplit('_', 1)[1]))

    # return resDataFile


def multiAperFileFlux(files, bw=64, fw=3, change_params=None, OWN=True, **kwargs):
    if not isinstance(change_params, dict):
        raise TypeError("Expected type \'dict\'. Remove key \'change_params\', if parameters do not change.")
    
    ResData = []
    #notfound = 0

    if OWN: 
        print('Processing: %d files' %(len(files)))

        with tqdm(total=len(files), file=sys.stdout) as pbar:
            for fileNo, datafile in enumerate(files):
                for key in change_params:
                    if key == 'aptR' and isinstance(change_params[key], list):
                        AperList = change_params[key]
                        del kwargs['aptR']
                                    
                    kwargs[key] = change_params[key][fileNo]

                if 'AperList' in locals():
                    for R in AperList:
                        kwargs['AptR'] = R
                        data = ExtractObjectsFlux(datafile, bw=bw, fw=fw **kwargs)

                #notfound += data.notfound
                ResData.append(data)
                
                # ResData.append({
                # 'file': os.path.basename(datafile),
                # 'results': data,
                # 'Objs not found': notfound
                # })

                #pbar.set_description('Processed: %d' % (1 + fileNo))
                pbar.update(1)

        print('FINISHED')
    
    else:
        for fileNo, datafile in enumerate(files):
            for key in change_params:
                if key == 'aptR' and isinstance(change_params[key], list):
                    AperList = change_params[key]
                    del kwargs['aptR']
                                
                kwargs[key] = change_params[key][fileNo]

            if 'AperList' in locals():
                for R in AperList:
                    kwargs['AptR'] = R
                    data = ExtractObjectsFlux(datafile, **kwargs)

            ResData.append(data)

    return ResData

    # if SAVE:
    #     save(file_dir, data, apers, CONVOLVE=False)
    #     save(target_data, conv_data, apers)


    # if save:
    #     if not os.path.exists(homedir+'Output'):
    #         os.makedirs(homedir+'Output')
    #     with open(homedir+'Output/{0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
    #                                                                    int(np.average(data[0]['y'])), 
    #                                                                    AptR[0], AptR[-1]), 'wb+') as file:
    #         pickle.dump(ResData, file)
    #     print('DATA WRITTEN TO FILE: {0}_{1}_apers({2}-{3}).pkl'.format(int(np.average(data[0]['x'])), 
    #                                                                      int(np.average(data[0]['y'])), 
    #                                                                      AptR[0], AptR[-1]))

    # if notfound != 0:
    #     print('OBJECTS NOT FOUND:', notfound)

    # return ResData

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