"""
Created on Tue Sep  2 09:00:00 2020
@author: blybelle

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