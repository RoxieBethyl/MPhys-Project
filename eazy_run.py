"""
Created on Fri Mar 15 16:44:40 2024
@author: blybelle
"""

"""
eazy_run.py
============

Overview
--------
This script runs Eazy with parameters read from the EazyRun.param on the input file. 

Dependencies
------------
- os
- pickle
- numpy
- pandas
- imgprocesslib
- eazy

Usage:
------
This script can be called from the command line using the following command:
    python eazy_run.py

The `EazyRun.param` file should contain the following parameters:
    Catalogue: The name of the input file.
    Outfile: The name of the output file.
    MagCut: Whether to apply a magnitude cut. Defaults to True.
    SigCut: Whether to apply a 5 sigma cut. Defaults to True.
    MarkHeader: Whether to mark the header in the output file. Defaults to True.

The output is written to a new file. The output from this script contains the following columns:
    id: The ID of the object.
    z_phot: The photometric redshift of the object.
    chi2_best: The best chi-squared value.
    z_err: The error associated with the redshift values.
    mag_f335m: The magnitude in the F335M filter.
    mag_f356w: The magnitude in the F356W filter.
    mag_f410m: The magnitude in the F410M filter.
    mag_f444w: The magnitude in the F444W filter.

"""


import os
import pickle
import numpy as np
import pandas as pd
from imgprocesslib import homedir
import eazy 

PHOTO_DIR = os.path.join(homedir, 'Photoz/')

def datawriter(data, filename, MarkHeader=True):
    """
    Writes the contents of a dictionary to an ASCII file.

    Parameters:
    -----------
    data (dict):
        The input dictionary containing the data to be written to the file.
    filename (str):
        The name of the output file.
    MarkHeader (bool, optional):
        Whether to mark the header in the output file. Defaults to True.
    
    Returns:
    --------
    None
    """

    with open(filename, 'w+') as file:
        if MarkHeader:
            file.write('# ')
        file.write(' '.join([f"{key} " for key in data.keys()]))  # Convert the list of keys to a string
        file.write("\n")
        for i in range(len(data['id'])): 
            writeline = ''
            for val in data.values():
                writeline += (f"{val[i]} ")
            writeline += '\n'
            file.write(writeline)


def writefile(dataframe, obj, Outfile, z_err, MagCut=True, SigCut=True, MarkHeader=True):
    """
    Writes the contents of a dataframe to an ASCII file with additional calculations and filters applied.

    Parameters:
    -----------
    dataframe (pandas.DataFrame):
        The input dataframe containing the data to be written to the file.
    obj:
        The object containing additional information needed for calculations.
    Outfile (str):
        The name of the output file.
    z_err (float):
        The error associated with the redshift values.
    MagCut (bool, optional):
        Whether to apply a magnitude cut. Defaults to True.
    SigCut (bool, optional):
        Whether to apply a 5 sigma cut. Defaults to True.
    MarkHeader (bool, optional): 
        Whether to mark the header in the output file. Defaults to True.

    Returns:
    --------
    None
    """

    dataframe["z_phot"] = obj.zml
    dataframe["chi2_best"] = obj.chi2_best
    dataframe["z_err"] = z_err

    # 5 SIGMA CUT
    # Drop rows with f356w < f356w_err*
    dataframe = pd.DataFrame(dataframe)
    if SigCut:
        dataframe = dataframe[dataframe['f356w'] > dataframe['f356w_err']*5]

    # calculate magnitudes
    dataframe["mag_f335m"] = -(2.5 * np.log10(dataframe["f335m"])) + 23.9
    dataframe["mag_f356w"] = -(2.5 * np.log10(dataframe["f356w"])) + 23.9
    dataframe["mag_f410m"] = -(2.5 * np.log10(dataframe["f410m"])) + 23.9
    #dataframe["mag_f444w"] = -2.5 * np.log10(dataframe["f444w"]) + 23.9

    # Drop rows with mag_f356w < 29.6
    if MagCut:
        dataframe = dataframe[dataframe['mag_f356w'] < 29.6]

    # fill any nan values with 'NAN' and re-number id column
    for filname in obj.flux_columns:
        dataframe.loc[(dataframe[filname] == -99.0), filname] = 0.0
    # dataframe = dataframe.fillna(value='NAN')
    
    dataframe = dataframe.to_dict(orient='list')
    dataframe['id'] = np.arange(0, len(dataframe['id']))

    # Save dataframe as ascii file ## CHANGE FILE NAME HERE PLEASE ##
    datawriter(dataframe, Outfile, MarkHeader=MarkHeader)
    print('\nNOBJS Remaining:', len(dataframe['id']))
    print(f"File saved as {Outfile}")

            
def _run_eazy(filename, Outfile='./Output/EazyOutfile.txt', MagCut=True, SigCut=True, MarkHeader=True):
    """
    Runs Eazy on the input file and writes the output to a new file.

    Parameters:
    -----------
    filename (str):
        The name of the input file.
    Outfile (str, optional):
        The name of the output file. Defaults to './Output/EazyOutfile.txt'.
    MagCut (bool, optional):
        Whether to apply a magnitude cut. Defaults to True.
    SigCut (bool, optional):
        Whether to apply a 5 sigma cut. Defaults to True.
    MarkHeader (bool, optional):
        Whether to mark the header in the output file. Defaults to True.

    Returns:
    --------
    None
    """

    dataframe = pickle.load(open(filename, 'rb'))
    datawriter(dataframe, 'EazyCatologue.txt', MarkHeader=MarkHeader)

    #grid_file = os.path.join(PHOTO_DIR, 'Output', 'templates_grid_file.npy')
    obj = eazy.photoz.PhotoZ(
        param_file = os.path.join(PHOTO_DIR, "jades_zphot.param"),
        translate_file = os.path.join(PHOTO_DIR, 'translate_jades.translate'),
        #zeropoint_file = os.path.join(PHOTO_DIR, '/zphot.zeropoint'),
        #tempfilt_data= np.load(grid_file, allow_pickle=True),
        n_proc=8,
        )
    
    obj.fit_catalog(obj.idx, n_proc=8)
    z_ = obj.pz_percentiles([16, 50, 84])
    z_err = np.average([np.subtract(z_[:, 1], z_[:, 0]), np.subtract(z_[:, 2], z_[:, 1])], axis=0)

    writefile(dataframe, obj, Outfile, z_err, 
              MagCut=MagCut, SigCut=SigCut, MarkHeader=MarkHeader)


def run_eazy():
    """
    Reads the parameters from the EazyRun.param file and runs Eazy on the input file.

    Returns:
    --------
    None
    """

    data_dict = {}
    with open('./EazyRun.param', 'r') as f:
        for line in f:
            variable, value = line.split()
            
            if value == "True" or value == "False":
                value = True if value == 'True' else False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            else:
                value = value

            data_dict[variable] = value

    _run_eazy(os.path.join(data_dict['Catalogue']), Outfile=os.path.join(data_dict['Outfile']), 
              MagCut=data_dict['MagCut'], SigCut=data_dict['SigCut'], MarkHeader=data_dict['MarkHeader'])
        


if __name__ == "__main__":
    run_eazy()