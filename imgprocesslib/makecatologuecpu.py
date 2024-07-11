"""
Created on Mon Nov 05 00:29:45 2023
Author: blybelle
"""

"""
Overview
--------
This Python module, `makecataloguecpu.py`, is designed for creating astronomical catalogues from images stored in FITS format, focusing on CPU-based processing. It utilizes multiprocessing to efficiently handle large datasets. The core functionality is provided by the `MakeCatalogue` class, which orchestrates the detection of astronomical objects and the calculation of their photometric properties.

Dependencies
------------
- imgprocesslib: Custom library for accessing basic image processing utilities.
- os: Standard Python library for interacting with the operating system.
- warnings: For suppressing specific warning messages, particularly `FITSFixedWarning` from `astropy`.
- tqdm: For displaying progress bars during lengthy operations.
- pickle: For object serialization and deserialization.
- csv: For reading from and writing to CSV files.
- sep: For source extraction and photometry.
- numpy (np): For numerical operations.
- astropy.io.fits: For reading and writing FITS files.
- astropy.wcs: For World Coordinate System (WCS) transformations.
- astropy.convolution: For convolving images with specific kernels, such as `Moffat2DKernel`.
- matplotlib.pyplot (plt), matplotlib.colors.Normalize, matplotlib (mpl): For plotting and visualizing images. Configured to use 'Times New Roman' font.
- multiprocessing (mp), multiprocessing.Pool: For parallel processing to improve performance on multi-core systems.

Classes
-------
MakeCatalogue
    A class designed to create astronomical catalogues from FITS files. It initializes with parameters for the detection image, photometric images, aperture size, background width, filter width, and additional keyword arguments for customization.

    Parameters:
    - detect_image (str): Path to the FITS file used for object detection.
    - phot_images (list): List of paths to FITS files for photometric measurements.
    - aper_size (int): Aperture size for photometry.
    - bw (int, optional): Background width for source extraction. Default is 64.
    - fw (int, optional): Filter width for source extraction. Default is 3.
    - change_kwargs (dict, optional): Dictionary of keyword arguments to modify default processing behavior.
    - **kwargs: Arbitrary keyword arguments for further customization.

Usage
-----
To use this module, instantiate the `MakeCatalogue` class with the required parameters and any optional keyword arguments. The class provides methods (not detailed here) for processing the FITS files and generating the catalogue.

"""


from imgprocesslib import homedir
import os
import warnings
from tqdm import tqdm

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

import multiprocessing as mp
from multiprocessing import Pool



class MakeCatologue():
    """
    Class to make a catologue of objects from a FITS file.
    
    Parameters:
    -----------
    detect_image: str
        The path to the FITS file of detection image.
    phot_images: list
        The list of paths to the FITS files of photometric images.
    aper_size: int
        The aperture size in radius.
    bw: int (2*n)
        The background width.
    fw: int (2*n)
        The foreground width.
    change_kwargs: dict
        The keyword arguments for individual phot_image.
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
    processors: int
        The number of processors to use.
    kernel: astropy.convolution.Moffat2DKernel
        The kernel for convolution.
    filetype: str
        The file type for saving.
    SAVE: boolean
        If True, the data will be saved.
    CONVOLVE: bool
        If True, the data will be convolved.
    CatalogueMap: bool
        If True, a map of the objects will be saved.
            
    Returns:
    --------
    None.
    """

    def __init__(self, detect_image, phot_images, aper_size, bw=64, fw=3, 
                 change_kwargs={}, **kwargs):
        
        poskwargs = ['objxlim', 'objylim', 'bh', 'fh', 'dpi', 'gain', 'ylim', 'xlim', 
                     'title', 'cmap', 'vmin', 'vmax', 'norm', 'detThesh', 'step_size', 
                     'tolerance', 'processors', 'kernel', 'filetype', 'SAVE', 'CONVOLVE',
                     'CATOLOGUEMAP']
        for key in kwargs.keys():
            assert key in poskwargs, '\'{0}\' not a class parameter. Possible parameters are {1}'.format(key, poskwargs)

        self.bw = bw
        self.fw = fw
        self.aper_size = aper_size

        if not isinstance(detect_image, str):
            raise TypeError("Data type is too large for one instance; use multifileflux() for this instance.")
        else:
            self.datafile = detect_image

        self.phot_images = phot_images
        self.change_kwargs = change_kwargs

        DEFAULTS = {
                'bh': self.bw,
                'fh': self.fw,
                'detThesh': 5,
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
                'processors':  mp.cpu_count() - 1,
                'CUT': False,
                }
        
        for key, default in DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default))

        SAVE = kwargs.get('SAVE', True)
        CATOLOGUEMAP = kwargs.get('CATOLOGUEMAP', False)

        if self.CONVOLVE:
            assert 'kernel' in kwargs.keys(), "\'{}\' must be given if \'convolve\' is True".format(key)
            kernel = kwargs.get('kernel', None)

        try:
            del kwargs['SAVE'], kwargs['pbar'], kwargs['CONVOLVE'], kwargs['kernel']
        except KeyError:
            pass
        
        print("Processing Detection Image")

        with fits.open(self.datafile, memmap=True) as hdul:
            data = hdul[0].data
            self.data = data.byteswap().newbyteorder()
        #self.data = fits.getdata(self.datafile).byteswap().newbyteorder()
        
        self.norm = kwargs.get('norm', Normalize(vmin=np.percentile(self.data, self.vmin), 
                             vmax=np.percentile(self.data, self.vmax)))
        
        sep.set_extract_pixstack(self.inBuffer)
        self.bkg = sep.Background(self.data, bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)
        self.objs, self.seg_map = sep.extract(self.data, self.detThesh, err=self.bkg.globalrms, segmentation_map=True)
        self.x_list = np.ascontiguousarray(self.objs['x'])
        self.y_list = np.ascontiguousarray(self.objs['y'])

        print("Objects discovered: {}".format(len(self.objs)))


        # Get RA and Dec for X and Y lists
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                ra_dec = np.array(WCS(self.datafile).all_pix2world(self.x_list, self.y_list, 0))
                self.ra, self.dec = ra_dec

        except FITSFixedWarning as e:
            if 'datfix' not in str(e):
                raise e
            
        print("")
        print("Processing Photometric Images")

        self.flux_phot = self.get_flux(change_kwargs)
 
        if SAVE:
            self.save(file_type=self.filetype)

        if CATOLOGUEMAP:
            self.plot_map(USE_SEG_MAP=True, PLOT_APERS=True, PLOT_CENTERS=True, COLORBAR=True)
            plt.savefig(os.path.join(homedir, 'Output', f"catalogue_object_map_{self.datafile.rsplit('_', 2)[0].rsplit('_', 1)[1]}.png"), dpi=500)
            plt.close()
        
        print("FINISHED")

    def filtering(self):
        # Making list
        # Perform make_apertures computation on GPU
        self.x_list, self.y_list = self.make_apertures(self.data, int(self.aper_size*2 + self.step_size))
        self.x_list = self.x_list
        self.y_list = self.y_list

        # Grabbing flux from data to remove any apertures where the detector doesn't contain data
        flux, _ , _ = sep.sum_circle(self.data, self.x_list, self.y_list, self.aper_size, err=self.bkg.globalrms)
    
        # Begin filtering out apertures that don't contain data
        self.x_list = self.x_list[~np.isclose(flux, 0., atol=1e-4)]
        self.y_list = self.y_list[~np.isclose(flux, 0., atol=1e-4)]

        # Grabbing flux from segementation map to remove any apertures that would collect object flux
        # Want to keep only apertures on sky background
        self.flux, _ , _ = sep.sum_circle(self.seg_map, self.x_list, self.y_list, self.aper_size, err=self.bkg.globalrms)

        self.x_list = self.x_list[self.flux == 0]
        self.y_list = self.y_list[self.flux == 0]

        # Grabbing the flux from sky apertures that should be close to zero
        self.flux, _ , _ = sep.sum_circle(self.data, self.x_list, self.y_list, self.aper_size, err=self.bkg.globalrms)
    
        return self.x_list, self.y_list, self.flux


    def process_file(self, args):
        fileNo, file, gkwargs, f_convolve = args
        kwargs = {}

        if self._sub1:
            fileNo -= 1
            
        data = fits.getdata(file).byteswap().newbyteorder()

        if isinstance(gkwargs, dict):
            if file != self.datafile:
                for key in gkwargs:      
                    if f_convolve:          
                        if 'gamma' == key:
                            gamma = gkwargs[key][fileNo]

                        elif 'alpha' == key:
                            alpha = gkwargs[key][fileNo]

                    else:
                        kwargs[key] = gkwargs[key][fileNo]

                kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

                if self.CONVOLVE:
                    chunk_size = 5000  # Adjust this value based on system's memory
                    convolved_data = np.empty_like(data)
                    for i in range(0, data.shape[0], chunk_size):
                        for j in range(0, data.shape[1], chunk_size):
                            chunk = data[i:i+chunk_size, j:j+chunk_size].copy()
                            convolved_chunk = convolve(chunk, kwargs['kernel'], normalize_kernel=True)
                            convolved_data[i:i+chunk_size, j:j+chunk_size] = convolved_chunk
                    
                    data = convolved_data

            else:
                self._sub1 = True
            
            bkg = sep.Background(data, bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)
            flux, _ , _ = sep.sum_circle(data, self.x_list, self.y_list, self.aper_size, err=bkg.globalrms)
            
            return {f"{file.rsplit('_', 2)[0].rsplit('_', 1)[1]}": flux}

    def get_flux(self, gkwargs):
        if gkwargs is None:
            gkwargs = {}

        self._sub1 = False
        flux_data = {}
        file_args = [(fileNo, file, gkwargs, self.CONVOLVE) for fileNo, file in enumerate(self.phot_images)]
        
        with tqdm(total=len(self.phot_images)) as pbar:
            for args in file_args:
                flux_data.update(self.process_file(args))
            
                pbar.update()

        return flux_data


    '''
    [Needs to be removed]
    def get_flux(self, gkwargs):
        if gkwargs is None:
            gkwargs = {}

        flux_data = {}
        kwargs = {}

        if isinstance(gkwargs, dict):
            for fileNo, file, in enumerate(self.phot_images):
                for key in gkwargs:      
                    if self.CONVOLVE:          
                        if 'gamma' == key:
                            gamma = gkwargs[key][fileNo]

                        elif 'alpha' == key:
                            alpha = gkwargs[key][fileNo]

                    else:
                        kwargs[key] = gkwargs[key][fileNo]
                kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)

                data = fits.getdata(file).byteswap().newbyteorder()

                if self.CONVOLVE:
                    data = convolve(data, kwargs['kernel'], normalize_kernel=True)

                bkg = sep.Background(data, bw=self.bw, bh=self.bh, fw=self.fw, fh=self.fh)
                flux, _ , _ = sep.sum_circle(data, self.x_list, self.y_list, self.aper_size, err=bkg.globalrms)
                
                flux_data.update({f"{file.rsplit('_', 2)[0].rsplit('_', 1)[1]}": flux})

                del data, bkg, flux

        return flux_data
        '''

    def save(self, file_type='pkl'):
        """
        Saves the data to a file.

        Parameters:
        -----------
        file_type: str or list
            The file type for saving the data.

        Returns:
        --------
        None.
        """

        print("")
        file_type = list([file_type]) if isinstance(file_type, str) else file_type

        if self.CONVOLVE:
            theWord = 'photometric_convolved_catalogue'
        else:
            theWord = 'photometric_catalogue'
        
        name = lambda ftype: f"{theWord}_{self.datafile.rsplit('_', 2)[0].rsplit('_', 1)[1]}_apers_{self.aper_size}_tol_{self.tolerance}_detThresh_{self.detThesh}.{ftype}"
        filepath = lambda filename: os.path.join(homedir, 'Output', filename)


        if ('pkl' or 'p' or 'pickle') in file_type:
            filename = name(list(set(file_type).intersection(('pkl', 'p', 'pickle')))[0])

            with open(filepath(filename), 'wb+') as file:
                write_dict = {'id': np.arange(len(self.x_list)),
                              'RA': self.ra, 
                              'Dec': self.dec, 
                              'x-position': self.x_list, 
                              'y-position': self.y_list,
                              }
                write_dict = write_dict | self.flux_phot
                pickle.dump(write_dict, file)

            print("Data written to \"{}\"".format(filepath(filename)))

        if ('csv') in file_type:
            filename = name(list(set(file_type).intersection(('csv',)))[0])

            with open(filepath(filename), 'w+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow("# id, ra, dec, x, y" + ''.join([f", {key}" for key in self.flux_phot]))

                for i, (ra, dec, x, y) in enumerate(zip(self.ra, 
                                                        self.dec, 
                                                        self.x_list, 
                                                        self.y_list,
                                                        )):
                    write_data = [i, ra, dec, x, y]
                    for key in self.flux_phot:
                        write_data.append(self.flux_phot[key][i])

                    writer.writerow(write_data)

            print("Data written to \"{}\"".format(filepath(filename)))

        if ('txt') in file_type:
            filename = name(list(set(file_type).intersection(('txt',)))[0])

            with open(filepath(filename), 'w+') as file:
                write_string = "# id ra dec x y" + ''.join([f" {key}" for key in self.flux_phot])
                file.write(write_string + '\n')

                for i, (ra, dec, x, y) in enumerate(zip(self.ra, 
                                                        self.dec,
                                                        self.x_list, 
                                                        self.y_list
                                                        )):
                    write_string = f"{i} {ra} {dec} {x} {y}"
                    for key in self.flux_phot:
                        write_string += f" {self.flux_phot[key][i]}"
                    
                    file.write(write_string + '\n')
    
            print("Data written to \"{}\"".format(filepath(filename)))
    


    '''
    [Needs to be removed]
    def process_file(self, args):
            fileNo, file, kwargs, device, bw, bh, fw, fh, x_list, y_list, aper_size, CONVOLVE = args
            if isinstance(kwargs, dict):
                for key in kwargs:                
                    if CONVOLVE:          
                        if 'gamma' == key:
                            gamma = kwargs[key][fileNo]

                        elif 'alpha' == key:
                            alpha = kwargs[key][fileNo]
                            
                        kwargs['kernel'] = Moffat2DKernel(gamma=gamma, alpha=alpha)
                    
                    else:
                        kwargs[key] = kwargs[key][fileNo]

            with fits.open(self.datafile, memmap=True) as hdul:
                data = hdul[0].data
                data = data.byteswap().newbyteorder()
            #data = fits.getdata(file).byteswap().newbyteorder()
            
            if CONVOLVE:
                data = convolve(data, kwargs['kernel'], normalize_kernel=True)
            
            bkg = sep.Background(data, bw=bw, bh=bh, fw=fw, fh=fh)
            flux, _ , _ = sep.sum_circle(data, x_list, y_list, aper_size, err=bkg.globalrms)
            
            result = {f"{file.rsplit('_', 2)[0].rsplit('_', 1)[1]}": flux}

            del data, bkg, flux

            return result
    '''

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
        plt.imshow(self.seg_map if PLOT_SEG_MAP == True else self.data, **plot_kwargs)
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
        Plots the image of the FITS file and some useful information is provided if asked for with Booleans
        
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
            plt.scatter(self.x_list, self.y_list, marker = 'x', c='#FB48C4', s=self.step_size*0.005) if USE_SEG_MAP \
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
    Derive the appropriate numbers in the data

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
    return (-2.5 * np.log10(5*sigma) + 23.90)


'''
[Needs to be removed]
def worker_wrapper(argkwargs):
    datafile, aper_size, bw, fw, kwargs = argkwargs
    return SkyCalibration(datafile, aper_size, bw, fw, **kwargs)


def init_SkyCalibration(datafiles, aper_val, target_datafile=None, bw=64, fw=3, change_kwargs=None, **kwargs):
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

    processors = kwargs.get('processors', mp.cpu_count() - 1)
    kwargs.pop('processors', None)

    results = []
    with mp.Pool(processes=processors) as pool:
        args_list = [(datafile, aper_size, bw, fw, kwargs) for datafile in datafiles]
        for res in tqdm(pool.imap(worker_wrapper, args_list), total=len(datafiles)):
            results.append(res)

    print("FINISHED")
    return results

'''