import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyUserWarning
import imgprocesslib.imageprocess as ip
from imgprocesslib import homedir, ddir

datapath = ddir+"/jwst_jades-gs_dr1_f335m_30mas_microJy.fits"
res = ip.extract_objects_flux(datapath)

# Fit the data using astropy.modeling
p_init = models.Moffat2D(res.data)
fit_p = fitting.LevMarLSQFitter(stddev=2)

with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
    warnings.filterwarnings('ignore', message='Model is linear in parameters',
                            category=AstropyUserWarning)
    p = fit_p(p_init, x, y, z)

# Plot the data with the best-fit model
plt.figure(figsize=(8, 2.5))
plt.subplot(1, 3, 1)
plt.imshow(z, origin='lower', interpolation='nearest', vmin=-1e4, vmax=5e4)
plt.title("Data")
plt.subplot(1, 3, 2)
plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
           vmax=5e4)
plt.title("Model")
plt.subplot(1, 3, 3)
plt.imshow(z - p(x, y), origin='lower', interpolation='nearest', vmin=-1e4,
           vmax=5e4)
plt.title("Residual")