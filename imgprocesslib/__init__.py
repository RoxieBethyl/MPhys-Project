"""
Created on Tue Sep 02 09:00:00 2020
@author: blybelle

This module is part of a larger application related to image processing and cataloging. It sets up the environment by defining paths to important directories and conditionally imports specific modules based on the availability of a GPU.

Attributes:
    homedir (str): The current working directory, considered as the home directory for the application.
    ddir (str): The path to the directory where data files are stored, named 'Jades_files'.
    imgdir (str): The path to the directory where image files are stored, named 'Images'.


Conditional Imports (Currently commented out):
    If a CUDA-capable GPU is available (as determined by torch.cuda.is_available()), the application imports modules optimized for GPU usage:
        - calibration: Module for calibrating images.
        - imageprocess: Module for processing images.
        - makecatalogue: Module for creating a catalogue of images.
    If a GPU is not available, the application falls back to CPU-optimized versions of these modules, with names suffixed by 'cpu'.

"""
import os
import torch

homedir = os.getcwd() + '/'  # Path to home directory
ddir = homedir + 'Jades_files/'  # Path to data directory
imgdir = homedir + 'Images/'  # Path to images directory

# Conditional import based on GPU availability
# if torch.cuda.is_available():
#     # Import GPU-optimized modules
#     from . import calibration, imageprocess, makecatologue
# else:
#     # Import CPU-optimized modules
#     from .calibrationcpu as calibration
#     from .imageprocesscpu as imageprocess
#     from .makecatologuecpu as makecatologue
"""
import os
import torch

homedir = os.getcwd() + '/' #Path to home dir
ddir = homedir + 'Jades_files/' #Path to data dir
imgdir = homedir + 'Images/'

# if torch.cuda.is_available():
#     # run calibration.py if GPU is available else run calibration_cpu.py
#     from . import calibration, imageprocess, makecatologue
# else:
#     from .calibrationcpu as calibration
#     from .imageprocesscpu as imageprocess
#     from .makecatologuecpu as makecatologue