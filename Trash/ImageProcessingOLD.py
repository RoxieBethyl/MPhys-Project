"""
Created on Thu Oct 05 11:48:32 2023
@author: blybelle
"""

#from ImageProcess import homedir, ddir
import os
from tqdm import tqdm
import sep
import numpy as np
from astropy.io import fits
#import colormaps as cmaps
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'

def fits_plot_wrapper(argskwargs):
    args, kwargs = argskwargs
    args = str(args)
    print(args)
    return fits_plot_one(args, **kwargs)
        

def fits_plot_one(image_path, **kwargs): #cmap='afmhot', title='', vmin=None, vmax=None, dpi=80,
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
    
    poskwargs = ['title', 'cmap', 'dpi', 'vmin', 'vmax', 'xlim', 'ylim', 'norm', 'block', 'i']
    for key in kwargs.keys():
        assert key in poskwargs, '\'{}\' not a function parameter'.format(key)
    
    image_data = fits.getdata(image_path)

    if 'title' in kwargs:
        plt.title(kwargs['title'])
    else:
        plt.title(os.path.basename(image_path))

    pltkwargs = dict()
    for key in ['cmap', 'norm']: # Possible arguments for plot
        if key in kwargs.keys():
            pltkwargs.update({key: kwargs[key]})
    
    if 'norm' in kwargs.keys():
        pltkwargs['norm'] = kwargs['norm']
    else:
        pltkwargs.update({'norm': Normalize(vmin=np.percentile(image_data, kwargs['vmin'] if 'vmin' in kwargs.keys() else 35), 
                                      vmax=np.percentile(image_data, kwargs['vmax'] if 'vmax' in kwargs.keys() else 95))})

    if 'cmap' in kwargs.keys():
        pltkwargs['cmap'] = kwargs['cmap']
    else:
        pltkwargs.update({'cmap': 'afmhot'})

    #plt.clf()
    plt.imshow(image_data, **pltkwargs)
    plt.xlim(kwargs['xlim'] if 'xlim' in kwargs else (None, None))
    plt.ylim(kwargs['ylim'] if 'ylim' in kwargs else (None, None)) 
    plt.gcf().set_dpi(kwargs['dpi'] if 'dpi' in kwargs.keys() else 80)
    #plt.gca().invert_yaxis()
    plt.xlabel("x-pixels")
    plt.ylabel("y-pixels")
    #plt.colorbar()  

    '''
    pltshowkwargs = dict()
    for key in ['block']: # for keys in [Potential Keywords]
        if key in kwargs.keys():
            pltshowkwargs.update({key: kwargs[key]})
    '''
        
    return plt.figure(kwargs['i'] if 'i' in kwargs.keys() else None)


def findflux(files, aptR, detThesh=100, **kwargs):
    poskwargs = ['objxlim', 'objylim', 'bw', 'bh', 'fw', 'fh', 'dpi', 'checkBkg', 'gain']
    for key in kwargs.keys():
        assert key in poskwargs, '\'{}\' not a function parameter'.format(key)

    try:
        checkBkg = kwargs['checkBkg']
    except KeyError:
        checkBkg = False
    
    try:
        bh = kwargs['bh']
    except KeyError:
        try:
            bh = kwargs['bw']
        except KeyError:
            bh = 64

    try:
        fh = kwargs['fh']
    except KeyError:
        try:
            fh = kwargs['fw']
        except KeyError:
            fh = 3

    sexkwargs = dict()
    for key in ['gain']: # for keys in [Potential Keywords]
        if key in kwargs.keys():
            sexkwargs.update({key: kwargs[key]})

    extkwargs = dict()
    for key in ['bw', 'bh', 'checkBkg']: # for keys in [Potential Keywords]
        if key in kwargs.keys():
            extkwargs.update({key: kwargs[key]})

    objNo = 0
    foundFile = []
    for fileNo, image in enumerate(files):
        data, bkg, objs = extract_objects(image, detThesh=100, **extkwargs)

        flux, fluxerr, flag = sep.sum_circle(data, objs['x'], objs['y'], aptR, err=bkg.globalrms
                                                #gain=image_data[0].header['GAIN']
                                                )
    
        fluxL =  np.zeros(((len(kwargs['objxlim']) if 'objxlim' in kwargs else len(data)), len(files))) if (len(files) > 1 or len(kwargs['objxlim']) > 1) else np.zeros(len(files))
        fluxerrL = np.zeros(((len(kwargs['objxlim']) if 'objxlim' in kwargs else len(data)), len(files))) if (len(files) > 1 or len(kwargs['objxlim']) > 1) else np.zeros(len(files))
        xL = np.zeros(((len(kwargs['objxlim']) if 'objxlim' in kwargs else len(data)), len(files))) if (len(files) > 1 or len(kwargs['objxlim']) > 1) else np.zeros(len(files))
        yL = np.zeros(((len(kwargs['objxlim']) if 'objxlim' in kwargs else len(data)), len(files))) if (len(files) > 1 or len(kwargs['objxlim']) > 1) else np.zeros(len(files))

        imgNo = 0
        foundFilePrintCheck = False
        for tar, x, y in zip(range(len(kwargs['objxlim']) if 'objxlim' in kwargs else len(data)), 
                                kwargs['objxlim'] if 'objxlim' in kwargs else len(data), 
                                kwargs['objylim'] if 'objylim' in kwargs else len(data)):

            for i in range(objs.size):
                if x[0] < objs['x'][i] < x[1] and y[0] < objs['y'][i] < y[1]:
                    tqdm.write("{:.0f}: object {:d} {:f} {:f}: flux = {:f} ± {:f}".format(fileNo, i, objs['x'][i], objs['y'][i], flux[i], fluxerr[i]))

                    if len(files) > 1:
                        fluxL[tar][fileNo] += flux[i]
                        fluxerrL[tar][fileNo] += fluxerr[i]
                        xL[tar][fileNo] += objs['x'][i]
                        yL[tar][fileNo] += objs['y'][i]

                    else:
                        fluxL[tar] = flux[i]
                        fluxerrL[tar] = fluxerr[i]
                        xL[tar] += objs['x'][i]
                        yL[tar] += objs['y'][i]

                    imgNo += 1
                    objNo += 1
                elif i >= len(objs):
                    foundFile.append(fileNo)
                    foundFilePrintCheck = True
            #print("{:.0f} found\n".format(imgNo))

    tqdm.write("Found {:.0f} targets out of {:.0f} objects for all files.\n".format(objNo, len(files)*len(kwargs['objxlim'])))
        
    if foundFilePrintCheck:
        tqdm.write("Not found in image(s) {0}".format((foundFile)))

    return {'flux': fluxL, 
            'fluxerr': fluxerrL, 
            'x': xL,
            'y': yL}

def multiAptFindFlux(files, AptR, detThesh=100, **kwargs):
    data = []
    for R in tqdm(AptR, total=len(AptR)):
        data += [findflux(files, R, detThesh, **kwargs)]
        #tqdm._instances.clear()
    return data


def plot_detected_sources(image_path, detThesh=100, **kwargs):
    pltkwargs = dict()
    for key in ['xlim', 'ylim', 'dpi', 'i']: # for keys in [Potential Keywords]
        if key in kwargs.keys():
            pltkwargs.update({key: kwargs[key]})

    data, bkg, objs = extract_objects(image_path, detThesh, **kwargs)

    fits_plot_one(image_path, **pltkwargs)
    plt.figure(pltkwargs['i'] if 'i' in pltkwargs.keys() else None)
    plt.scatter(objs['x'], objs['y'], marker = 'x', c='#2BFF0A')
    return plt.figure(pltkwargs['i'] if 'i' in pltkwargs.keys() else None)


def extract_objects(image, detThesh=100, **kwargs):
    try:
        bh = kwargs['bh']
    except KeyError:
        try:
            bh = kwargs['bw']
        except KeyError:
            bh = 64

    try:
        fh = kwargs['fh']
    except KeyError:
        try:
            fh = kwargs['fw']
        except KeyError:
            fh = 3
    try:
        checkBkg = kwargs['checkBkg']
    except KeyError:
        checkBkg = False
    
    try:
        data = fits.open(image)
        data = data.byteswap().newbyteorder()
    except:
        data = fits.getdata(image)
        data = data.byteswap().newbyteorder()
    
    bkg = sep.Background(data, bw=kwargs['bw'] if 'bw' in kwargs else bh, bh=bh, 
                            fw=kwargs['fw'] if 'fw' in kwargs else fh, fh=fh)

    if checkBkg:
        bkg_image = bkg.back()
        plt.xlim(kwargs['xlim'])
        plt.ylim(kwargs['ylim'])
        plt.gcf().set_dpi(kwargs['dpi'] if 'dpi' in kwargs else 500)
        plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
        plt.colorbar()
        plt.show()

    data -= bkg
    objs = sep.extract(data, detThesh, err=bkg.globalrms)
    return data, bkg, objs
        