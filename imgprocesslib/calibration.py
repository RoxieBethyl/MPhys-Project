"""
Created on Mon Nov 05 00:29:45 2023
@author: blybelle
"""

"""
calibration.py
==============

Overview
--------
This module, `calibration.py`, is designed for performing sky calibration on astronomical images, specifically those stored in FITS format. It provides a class, `SkyCalibration`, which facilitates the calibration process by applying various image processing techniques. The module leverages the power of PyTorch for potential GPU acceleration, making it suitable for handling large datasets efficiently.

Dependencies
------------
- torch: For GPU acceleration and profiling.
- imgprocesslib: A custom library for basic image processing tasks.
- os: For interacting with the file system.
- warnings: For suppressing specific warning messages.
- tqdm: For displaying progress bars during processing.
- pickle: For object serialization and deserialization.
- csv: For reading from and writing to CSV files.
- sep: For source extraction and photometry.
- numpy: For numerical operations.
- astropy.io.fits: For reading and writing FITS files.
- astropy.wcs: For World Coordinate System (WCS) transformations.
- astropy.convolution: For convolving images with specific kernels.
- matplotlib.pyplot, matplotlib.colors: For plotting and visualizing images.

Classes
-------
SkyCalibration
    A class that provides methods for performing sky calibration on astronomical images. It initializes with the path to a FITS file and various parameters related to the calibration process.

Usage
-----
To use this module, instantiate the `SkyCalibration` class with the path to a FITS file and the desired aperture size. Optional parameters include background width (bw) and filter width (fw).

Example
-------
```python
from calibration import SkyCalibration

# Initialize the SkyCalibration class with a FITS file and aperture size
calibrator = SkyCalibration('path/to/fitsfile.fits', aper_size=5)

# Additional methods can be called on the calibrator object to perform specific calibration tasks

"""


import torch

import os
import warnings
from tqdm import tqdm

from imgprocesslib import homedir

import pickle
import csv

import sep
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

from astropy.convolution import convolve, Moffat2DKernel
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

#from multiprocessing import Pool, TimeoutError
#import multiprocessing as mp

class SkyCalibration():
    torch.cuda.profiler.start()
    def __init__(self, datafile, aper_size, bw=64, fw=3, **kwargs):
        """
        Parameters:
        -----------
        datafile: str
            The path to the FITS file.
        aper_size: int
            The aperture size in radius.
        bw: int (2*n)
            The background width.
        fw: int (2*n)
            The foreground width.
        **kwargs: dict
            The keyword arguments.
        
        Optional:
        ---------
        objxlim: tuple
            The x limits for the object plot.
        objylim: tuple
            The y limits for the object plot.
        xlim: tuple
            The x limits for the image plot.
        ylim: tuple
            The y limits for the image plot.
        dpi: int
            The dpi of the image plot.
        title: str
            The title of the image plot.
        cmap: str
            The colour map of the image plot.
        vmin: float
            The minimum value of the image plot.
        vmax: float
            The maximum value of the image plot.
        norm: matplotlib.colors.Normalize
            The normalisation of the image plot.
        detThesh: float
            The detection threshold for Source Extractor.
        step_size: int
            The step size for the apertures.
        tolerance: float
            The tolerance for the aperture flux.
        SAVE: boolean
            If True, the data will be saved.
        processors: int
            The number of processors to use.
            
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        poskwargs = ['objxlim', 'objylim', 'bh', 'fh', 'dpi', 'gain', 'ylim', 'xlim', 
                     'title', 'cmap', 'vmin', 'vmax', 'norm', 'detThesh', 'step_size', 
                     'tolerance', 'processors', 'kernel', 'filetype', 'SAVE', 'CONVOLVE',
                     'inBuffer', 'chunk_size']
        for key in kwargs.keys():
            assert key in poskwargs, '\'{0}\' not a class parameter. Possible parameters are {1}'.format(key, poskwargs)

        self.bw = bw
        self.fw = fw
        self.aper_size = aper_size

        if not isinstance(datafile, str):
            raise TypeError("Data type is too large for one instance; use multifileflux() for this instance.")
        self.datafile = datafile
        
        DEFAULTS = {
            'bh': self.bw,
            'fh': self.fw,
            'detThesh': 1,
            'objxlim': (None, None),
            'objylim': (None, None),
            'xlim': (None, None),
            'ylim': (None, None),
            'dpi': 500,
            'title': os.path.basename(self.datafile),
            'cmap': 'afmhot',
            'vmin': 35,
            'vmax': 95,
            'filetype': 'pkl',
            'step_size': 1,
            'tolerance': 1e-5,
            'CONVOLVE': False,
            'inBuffer': 5000000,
            'chunk_size': 100,
            #'processors':  mp.cpu_count() - 1,
            }

        for key, default in DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))

        if self.detThesh >= 100:
            raise ValueError("Detection threshold is too high")

        SAVE = kwargs.get('SAVE', True)

        self.data = torch.tensor(fits.getdata(self.datafile).byteswap().newbyteorder(), dtype=torch.float32).to(self.device)

        if self.CONVOLVE:
            assert 'kernel' in kwargs.keys(), "\'{}\' must be given if \'convolve\' is True".format(key)
            kernel = kwargs.get('kernel', None)

            self.data = torch.tensor(convolve(self.data.cpu().numpy(), kernel, normalize_kernel=True), dtype=torch.float32).to(self.device)

        try:
            del kwargs['SAVE'], kwargs['pbar'], kwargs['CONVOLVE'], kwargs['kernel']
        except KeyError:
            pass

        self.norm = kwargs.get('norm', Normalize(vmin=np.percentile(self.data.cpu().numpy(), self.vmin), 
                                                 vmax=np.percentile(self.data.cpu().numpy(), self.vmax)))

        sep.set_extract_pixstack(self.inBuffer)
        self.bkg = sep.Background(self.data.cpu().numpy(), bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)#

        # Assuming self.data is a 2D numpy array representing your image
        height, width = self.data.shape

        self.objs = []
        self.seg_map = []
        self.x_list = []
        self.y_list = []
        self.sky_flux = []

        for i in range(0, height, self.chunk_size):
            for j in range(0, width, self.chunk_size):
                chunk = self.data.cpu().numpy()[i:i+self.chunk_size, j:j+self.chunk_size]
                bkg_chunk = self.bkg[i:i+self.chunk_size, j:j+self.chunk_size]

                objs, seg_map = sep.extract(chunk, self.detThesh, err=bkg_chunk.globalrms, segmentation_map=True)
                
                # Making list
                # Perform make_apertures computation on GPU
                x_list, y_list = self.make_apertures(chunk, int(self.aper_size*2 + self.step_size))        
                x_list = torch.tensor(x_list, dtype=torch.float32).to(self.device)
                y_list = torch.tensor(y_list, dtype=torch.float32).to(self.device)

                flux, _ , _ = sep.sum_circle(chunk, x_list, y_list, self.aper_size, err=bkg_chunk.globalrms)

                x_list = x_list[~np.isclose(flux, 0., atol=1e-4)]
                y_list = y_list[~np.isclose(flux, 0., atol=1e-4)]

                # Grabbing flux from segementation map to remove any apertures that would collect object flux
                # Want to keep only apertures on sky background
                sky_flux, _ , _ = sep.sum_circle(seg_map, x_list, y_list, self.aper_size, err=bkg_chunk.globalrms)

                x_list = x_list[sky_flux == 0]
                y_list = y_list[sky_flux == 0]

                # Grabbing the flux from sky apertures that should be close to zero
                sky_flux, _ , _ = sep.sum_circle(chunk, x_list, y_list, self.aper_size, err=bkg_chunk.globalrms)

                self.objs.append(objs)
                self.seg_map.append(seg_map)
                self.x_list.append(x_list)
                self.y_list.append(y_list)
                self.sky_flux.append(sky_flux)

        # Get RA and Dec for X and Y lists
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                ra_dec = np.array(WCS(self.datafile).all_pix2world(self.x_list.cpu().numpy(), self.y_list.cpu().numpy(), 0))
                self.ra, self.dec = torch.tensor(ra_dec, dtype=torch.float32).to(self.device)
                #self.ra, self.dec = torch.tensor(WCS(self.datafile).all_pix2world(self.x_list.cpu().numpy(), self.y_list.cpu().numpy(), 0), dtype=torch.float32).to(self.device)
        except FITSFixedWarning as e:
            if 'datfix' not in str(e):
                raise e
        
        if SAVE:
            self.save(file_type=self.filetype)

    def worker_wrapper_extract(self, argkwargs):
        """
        Wrapper for the Source Extractor extraction function. (Function not used in current implementation.)
        
        Parameters:
        -----------
        argkwargs: tuple
            The arguments and keyword arguments for the extraction function.
            
        Returns:
        --------
        objs: numpy.ndarray
            The objects found in the image. 
        seg_map: numpy.ndarray
            The segmentation map of the image.
        """

        self.data, self.detThesh, self.bkg.globalrms, segmentation_map = argkwargs
        return sep.extract(self.data, self.detThesh, err=self.bkg.globalrms, segmentation_map=segmentation_map)
    

    def worker_wrapper_sumcircle(self, argkwargs):
        """
        Wrapper for the Source Extractor sum_circle function. (Function not used in current implementation.)

        Parameters:
        -----------
        argkwargs: tuple
            The arguments and keyword arguments for the sum_circle function.
        
        Returns:
        --------
        flux: numpy.ndarray
            The flux of the object.
        fluxerr: numpy.ndarray
            The error in the flux of the object.
        flag: numpy.ndarray
            The flag of the object.
        """

        self.data, self.x_list, self.y_list, self.aper_size, self.bkg.globalrms = argkwargs
        return sep.sum_circle(self.data, self.x_list, self.y_list, self.aper_size, err=self.bkg.globalrms)

    def save(self, file_type='pkl'):
        """
        Saves the data to a file.

        Parameters:
        -----------
        file_type: str or list
            The file type(s) to save the data to.
        """

        file_type = list([file_type]) if isinstance(file_type, str) else file_type
        if self.CONVOLVE:
            theWord = 'convolved'
        else:
            theWord = 'calibration'
        
        name = lambda ftype: f"{theWord}_depths_{self.datafile.rsplit('_', 2)[0].rsplit('_', 1)[1]}_apers_{self.aper_size}_tol_{self.tolerance}._detThresh_{self.detThesh}.{ftype}"
        filepath = lambda filename: os.path.join(homedir, 'Output', filename)

        if ('pkl' or 'p' or 'pickle') in file_type:
            #print(file_type)
            #print(list(set(file_type).intersection(('pkl', 'p', 'pickle'))))
            filename = name(list(set(file_type).intersection(('pkl', 'p', 'pickle')))[0])

            with open(filepath(filename), 'wb+') as file:
                pickle.dump({'RA': self.ra.cpu().numpy(), 
                            'Dec': self.dec.cpu().numpy(), 
                            'x-position': self.x_list.cpu().numpy(), 
                            'y-position': self.y_list.cpu().numpy(), 
                            'sky_flux': self.sky_flux,
                            }, file)

            print("Data written to \"{}\"".format(filepath(filename)))

        if ('csv') in file_type:
            filename = name(list(set(file_type).intersection(('csv',)))[0])

            with open(filepath(filename), 'w+', newline='') as file:
                writer = csv.writer(file)
                for ra, dec, x, y, flux in zip(self.ra.cpu().numpy(), 
                                            self.dec.cpu().numpy(), 
                                            self.x_list.cpu().numpy(), 
                                            self.y_list.cpu().numpy(), 
                                            self.sky_flux
                                            ):
                    writer.writerow([ra, dec, x, y, flux])

            print("Data written to \"{}\"".format(filepath(filename)))

        if ('txt') in file_type:
            filename = name(list(set(file_type).intersection(('txt',)))[0])

            with open(filepath(filename), 'w+') as file:
                for ra, dec, x, y, flux in zip(self.ra.cpu().numpy(), 
                                               self.dec.cpu().numpy(),
                                               self.x_list.cpu().numpy(), 
                                               self.y_list.cpu().numpy(), 
                                               self.sky_flux
                                               ):
                    file.write(f"{ra} {dec} {x} {y} {flux}\n")
    
            print("Data written to \"{}\"".format(filepath(filename)))
                    
    #def __str__(self):

        
    def make_apertures(self, data, step_size):
        """
        Generates lists of x and y coordinates for apertures based on the given step size.

        Parameters:
        -----------
        data: numpy.ndarray
            The data from which the aperture coordinates will be generated.
        step_size: int
            The distance between each aperture.

        Returns:
        --------
        x_list, y_list: numpy.ndarray
            The lists of x and y coordinates for the apertures.
        """

        assert isinstance(step_size, int) and step_size > 0, f"Step size must be a positive integer. Recieved type {type(step_size)}."
        
        num_steps_x = data.shape[1] // step_size
        num_steps_y = data.shape[0] // step_size

        x_coords = np.zeros((num_steps_x * num_steps_y))
        y_coords = np.zeros((num_steps_x * num_steps_y))

        index = 0
        x_coord = step_size
        for i in range(num_steps_x):
            y_coord = step_size

            for j in range(num_steps_y):
                x_coords[index] = x_coord
                y_coords[index] = y_coord

                y_coord += step_size
                index += 1

            x_coord += step_size

        return x_coords, y_coords


    def fits_plot_image(self, PLOT=False, PLOT_SEG_MAP=False):
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

        plot_kwargs = {'norm': self.norm, 'cmap': self.cmap} if not PLOT_SEG_MAP else {}

        plt.clf()
        plt.title(self.title)
        plt.imshow(self.seg_map.cpu().numpy() if PLOT_SEG_MAP == True else self.data.cpu().numpy(), **plot_kwargs)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.gcf().set_dpi(self.dpi)
        plt.gca().invert_yaxis()
        plt.xlabel("x-pixels")
        plt.ylabel("y-pixels")
        #plt.colorbar()
        self.figure = plt.figure(1)

        if PLOT:
            plt.show()

        return self.figure


    def plot_map(self, **kwargs):
        """
        Plots the image of the FITS file and some useful information is provided if asked for with Booleans as arguments to the function.
        
        Optional:
        ---------
        USE_SEG_MAP: Boolean
            If True, the segmentation map of the image will be plotted.
        PLOT_APERS: Boolean
            If True, apertures across the full map will be plotted.
        PLOT_CENTERS: Boolean
            If True, the coordinates of either object data or the centroids of apertures will be plotted.
        COLORBAR: Boolean
            If True, ```plt.colorbar()``` will be plotted.

        Returns:
        --------
        Image_plot(plt.show):
            Image of the FITS file
        """

        poskwargs = ['USE_SEG_MAP', 'PLOT_APERS', 'PLOT_CENTERS', 'COLORBAR']
        for key in kwargs.keys():
            assert key in poskwargs, '\'{0}\' not a class parameter. Possible parameters are {1}'.format(key, poskwargs)

        USE_SEG_MAP = kwargs.get('USE_SEG_MAP', False)
        PLOT_APERS = kwargs.get('PLOT_APERS', False)
        PLOT_CENTERS = kwargs.get('PLOT_CENTERS', False)
        COLORBAR = kwargs.get('COLORBAR', False)

        self.fits_plot_image(PLOT_SEG_MAP=USE_SEG_MAP)
        self.figure

        if PLOT_CENTERS:
            plt.scatter(self.x_list.cpu().numpy(), self.y_list.cpu().numpy(), marker = 'x', c='#FB48C4', s=self.step_size*0.005) if USE_SEG_MAP \
            else plt.scatter(self.objs['x'], self.objs['y'], marker = 'x', c='#2BFF0A')

        fig = self.figure
        ax = fig.gca()

        if PLOT_APERS:
            x_list = self.x_list
            y_list = self.y_list

        else:
            x_list = self.objs['x']
            y_list = self.objs['y']

        for x, y in zip(x_list, y_list):
            data = plt.Circle((x, y), self.aper_size, color='blue' if USE_SEG_MAP else 'yellow', fill=False)
            ax.add_patch(data)
        
        if COLORBAR:
            plt.colorbar()
        return self.figure
    
    torch.cuda.profiler.stop()


def plot_dist(file, aper_size, bw=64, fw=3, step_size=1, PLOT=False, **kwargs):
    """
    Plots the distribution of the sky flux.

    Parameters:
    -----------
    file: str
        The path to the FITS file.
    aper_size: int
        The aperture size.
    bw: int
        The background width.
    fw: int
        The foreground width.
    step_size: int
        The step size.
    plot: bool
        If True, the distribution will be plotted.
    bins: int
        The number of bins for the histogram.

    Returns:
    --------
    hist: list
        The values of the histogram.   
    sky_flux: list
        The values of the sky flux.
    """
     
    bins = kwargs.get('bins', 50)
    del kwargs['bins']

    img_d = sky_flux(file, aper_size, bw, fw, **kwargs)
    sky_flux, _ = img_d.sky_flux()
    hist = plt.hist(sky_flux, bins=bins)
    
    if PLOT:
        plt.show()

    return hist, sky_flux


def get_hist_info(data, bins=30):
    """
    Derive the appropriate numbers in the data for a histogram plot. 

    Parameters:
    -----------
    data: ndarray
        Array of the x coordinates of the data
    bins (optional): Integer
        Bins of the histogram; changes the number of scatter points to outputs

    Returns:
    --------
    bins: 
        X values of the dataset in the histogram (taken as the of each bin)
    n: 
        Y values of the dataset (frequency or occurences in each bin)
    """
    # Here shifting x values to the center of the bins and
    # getting the height of the bin for y values
    
    n, bins = np.histogram(data, bins=bins)
    bins = [(bins[i] + bins[i+1])/2. for i in range(len(bins)-1)]
    return bins, n

def lim_mag(sigma):
    """
    Converts the standard deviation of the sky flux to a limiting magnitude.

    Parameters:
    -----------
    sigma: float
        The standard deviation of the sky flux.

    Returns:
    --------
    float
        The limiting magnitude.
    """
    return (-2.5 * np.log10(5*sigma) + 23.90)


def worker_wrapper(argkwargs):
    """
    Worker wrapper for the SkyCalibration class. (Function not used in current implementation.)
    
    Parameters:
    -----------
    argkwargs: tuple
        The arguments and keyword arguments for the SkyCalibration class.
        
    Returns:
    --------
    SkyCalibration
        The SkyCalibration object.
    """

    datafile, aper_size, bw, fw, kwargs = argkwargs
    return SkyCalibration(datafile, aper_size, bw, fw, **kwargs)


def init_SkyCalibration(datafiles, aper_val, target_datafile=None, bw=64, fw=3, change_kwargs=None, **kwargs):
    """
    Initialises the SkyCalibration class for multiple data files. 

    Parameters:
    -----------
    datafiles: list
        The list of data files.
    aper_val: int or list
        The aperture size.
    target_datafile: str (optional)
        The target data file.
    bw: int (optional)
        The background width.
    fw: int (optional)
        The foreground width.
    change_kwargs: dict (optional)
        The dictionary of keyword arguments to change.
    **kwargs: dict
        The keyword arguments for the SkyCalibration class.

    Optional:
    ---------
    CONVOLVE: bool
        If True, the data will be convolved.
    gamma: float
        The gamma value for the Moffat2DKernel.
    alpha: float
        The alpha value for the Moffat2DKernel.
    kernel: astropy.convolution.Moffat2DKernel
        The kernel for the convolution.
    filetype: str
        The file type for saving the data.
    SAVE: bool
        If True, the data will be saved.
    processors: int
        The number of processors to use.
    inBuffer: int
        The buffer size for Source Extractor.
    chunk_size: int
        The chunk size for the data.

    Returns:
    --------
    None
    """

    def wrapper():
        img_objs = SkyCalibration(file, aper_size, bw=bw, fw=fw, **kwargs)
        pbar.update()

        # Clear the img_objs variable from memory
        del img_objs
        
    pbar = tqdm(datafiles, total=len(datafiles))
    CONVOLVE = kwargs.get('CONVOLVE', False)

    for fileNo, file in enumerate(datafiles):
        if CONVOLVE:
            if file == target_datafile:
                continue
            
            if isinstance(change_kwargs, dict):
                for key in change_kwargs:                
                    if 'gamma' == key:
                        gamma = change_kwargs[key][fileNo]

                    elif 'alpha' == key:
                        alpha = change_kwargs[key][fileNo]

                    else:
                        kwargs[key] = change_kwargs[key][fileNo]

                kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

        if isinstance(aper_val, list or np.ndarray):
            for aper_size in aper_val:
                pbar.refresh()
                wrapper()    
        else:
            pbar.refresh()
            aper_size = aper_val
            wrapper()

    print("FINISHED")