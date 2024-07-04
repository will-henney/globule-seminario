# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import sys
sys.path.append("../scripts")
import numpy as np
import fit_psf
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = "nearest"

# # Load data from files

from astropy.io import fits
from pathlib import Path

datapath = Path.cwd().parent / "data"

# Get the synthetic PSF that is 512 pixels and used the O3 star spectrum

psf = fit_psf.OversampledPSF(
    fits.open(datapath / "psf-o3v-512-0000-0000-nircam-f090w.fits")["OVERDIST"].data,
    oversample = 4,
)

# Now get the target image that we want to fit

im = fit_psf.PixelImage(
    fits.open(datapath / "wr124-jwst-nircam-2022-f090w.fits")["SCI"].data
)

x0, y0 = 2315.60, 2321.96

peak = fit_psf.Peak(im, x0, y0, psf)

peak.bg0, peak.bscale0

# %%time
peak.reproject_order = "nearest-neighbor"
#peak.reproject_order = "bicubic"
#peak.reproject_order = "bilinear"
peak.update_model_image(peak.bscale0, x0, y0, peak.bg0, theta=0, smoothing=0)

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 12))
norm = matplotlib.colors.LogNorm(vmin=0.1, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
im = ax[0, 0].imshow(peak.obs_im, norm=norm, cmap=cmap)
ax[0, 1].imshow(peak.mod_im, norm=norm, cmap=cmap)
fig.colorbar(im, ax=ax, orientation="horizontal", aspect=50,)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    peak.obs_im, 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    peak.mod_im, 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

np.percentile(peak.obs_im, 1), np.percentile(peak.mod_im, 1)

np.max(peak.obs_im[peak.mask]), np.max(peak.mod_im[peak.mask])

# %%time
result = fit_psf.fit_peak(peak)

result

residual = np.zeros_like(peak.skycutout.data)
residual[peak.mask] = fit_psf.residual(result.params, peak)



# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(peak.skycutout.data, norm=norm, cmap=cmap)
ax[0, 1].imshow(peak.psfskycutout, norm=norm, cmap=cmap)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    peak.skycutout.data, 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    peak.psfskycutout * result.init_values["bscale"] / peak.bscale, 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

result.init_values

peak.bscale

peak.skycutout.data[peak.mask].max()

peak.psfskycutout[peak.mask].max()


