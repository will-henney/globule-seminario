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

# %env WEBBPSF_PATH /Users/will/Work/webbpsf-data

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.modeling import models, fitting
from astropy.nddata import Cutout2D
from astropy.convolution import (
    interpolate_replace_nans, 
    Box2DKernel, 
    Gaussian2DKernel, 
    convolve
)
from astropy.stats import sigma_clip
from regions import PixCoord, CirclePixelRegion, Regions
from reproject import reproject_interp
import webbpsf
from scipy import ndimage as ndi
import skimage as ski
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import astropy.visualization as av
import lmfit

matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = "nearest"

# ## Load the JWST NIRCAM image

datapath = Path.cwd().parent / "data"
hdulist = fits.open(datapath / "wr124-jwst-nircam-2022-f090w.fits")
hdulist.info()

# ## Find the peaks in the image
#
# We can set a brightness threshold and a minimum distance between peaks

peaks = ski.feature.peak_local_max(hdulist["SCI"].data, min_distance=6, threshold_abs=4.0)

# If we bring the threshold down to 4 and the minimum distance to 6, then we get nearly all the stars, but we also start to pick up on some false positives. 

len(peaks)

# So this finds over 5000 peaks, the majority of which are real stars

peaks[:10]

# Write the peaks out as a regions file for DS9

regs = Regions([CirclePixelRegion(center=PixCoord(x ,y), radius=3)
                for y, x in peaks])
regname = "star-peaks.reg"
regs.write(datapath / regname, format="ds9", overwrite=True)

# ### Start with the brightest stars

bright_peaks = ski.feature.peak_local_max(
    hdulist["SCI"].data, 
    min_distance=20, 
    threshold_abs=1000.0,
)

len(bright_peaks)

regs = Regions([CirclePixelRegion(center=PixCoord(x ,y), radius=15)
                for y, x in bright_peaks])
regname = "star-bright-peaks-1000.reg"
regs.write(datapath / regname, format="ds9", overwrite=True)

# This gets a smaller number of brighter stars. It does miss some of the bright saturatedones though, since they do not quite make the criterion. 
#
# So the plan is to fit the psf for all these and remove them, and then to do find_peaks again to get the fainter ones. 

# Demonstration that the peaks are in order of brightness:

print("Brightest peaks:")
for (j, i) in bright_peaks[:5]:
    print(f"({j}, {i}) = {hdulist['SCI'].data[j, i]:.2f}")

print("Faintest peaks:")
for (j, i) in peaks[-5:]:
    print(f"({j}, {i}) = {hdulist['SCI'].data[j, i]:.2f}")

# ## Demo of PSF removal
#
# Following the [webbpsf tutorial notebook](https://nbviewer.org/github/spacetelescope/webbpsf/blob/stable/notebooks/WebbPSF_tutorial.ipynb) and the [User Guide](https://webbpsf.readthedocs.io/en/latest/)

nc = webbpsf.NIRCam()
nc.filter = "F090W"
nc.options

#

# +
# nc.detector_position?
# -



# Start off with a moderate sized box of 512x512 pixels

psf = nc.calc_psf(fov_pixels=512)

# First HDU (0: OVERSAMP) is oversampled by 4. Second HDU (1: DET_SAMP) is with the native NIRCAM pixels

psf.info()

webbpsf.display_psf(psf, ext=1, vmax=0.001, vmin=1e-10)

# Now extract the same-sized window from the observations. 
#
# I have measured the star position in the dithered image.

xy_star_fits = 2316.35, 2323.38

i0, j0 = np.round(xy_star_fits).astype(int)
imslice = slice(j0 - 256, j0 + 256), slice(i0 - 256, i0 + 256)
imslice

# We are going to have to rotate and shift the sub-sampled PSF, which we can do using fake WCS with reproject. Hopefully that is fast enough

imobs = hdulist["SCI"].data[imslice] - 0.2
weights = hdulist["WHT"].data[imslice]
imobs[weights == 0.0] = np.nan

# The `-0.2` is for the background level. 

imobs.shape

impsf = psf["DET_SAMP"].data
#impsf = psf["DET_DIST"].data

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
scale_obs, scale_psf = 1.0, 3.5e-8
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(imobs / scale_obs, norm=norm, cmap=cmap)
ax[0, 1].imshow(impsf / scale_psf, norm=norm, cmap=cmap)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    (imobs / scale_obs), 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    (imobs / scale_obs) - (impsf / scale_psf), 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

w = 5
cslice = slice(256-w, 256+w)

fig,  ax = plt.subplots(2, 1)
ax[0].plot(np.nanmean(imobs[:, cslice], axis=1) / scale_obs)
ax[0].plot(np.nanmean(impsf[:, cslice], axis=1) / scale_psf)
ax[0].set(yscale="log", ylim=[0.3, 5000.0])
ax[1].plot(np.nanmean(imobs[cslice, :], axis=0) / scale_obs)
ax[1].plot(np.nanmean(impsf[cslice, :], axis=0) / scale_psf)
ax[1].set(yscale="log", ylim=[0.3, 5000.0])
...;

# ### Setup fine and coarse WCS 
#
# We will use coarse pixel units for both WCS. 
# The fine grid is oversampled by a factor of 4, so we set the pixel size to be 0.25.
# We put the origin in the center of the grid, which is at half-integer pixel value (pixel corner) since we have an even number of pixels.
# Note that CRPIX is in FITS 1-based indexing.

finewcs = WCS(psf["OVERSAMP"])
finewcs.wcs.cdelt /= 4
finewcs.wcs.crpix = (np.array(finewcs.pixel_shape) + 1) / 2
finewcs.wcs.cunit = ["pixel", "pixel"]
finewcs.wcs.cname = ["x", "y"]
finewcs.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]
finewcs

# Same for the coarse grid, but with pixel size of 1.0

detwcs = WCS(psf["DET_SAMP"])
detwcs.wcs.crpix = (np.array(detwcs.pixel_shape) + 1) / 2
detwcs.wcs.cunit = ["pixel", "pixel"]
detwcs.wcs.cname = ["x", "y"]
detwcs.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]
detwcs

# Check that the corner pixels are symmetric with respect to the origin

detwcs.pixel_to_world([0, 511], [0, 511])

# Check that the first four pixels on the fine grid have a mean position that is the same as the first pixel on the coarse grid

xx, yy = finewcs.pixel_to_world(range(4), range(4))
xx.mean(), yy.mean()

finewcs.world_axis_physical_types

# ### How to rotate the fine WCS
#
# We use the PC matrix to rotate the axes by $\theta$. For an anticlockwise rotation we have
# $$
# \left( 
# \begin{array}{cc}
#  \cos\theta & -\sin\theta \\
#  \sin\theta & \cos\theta
# \end{array}
# \right)
# $$

theta = 1 * u.deg
cth, sth = np.cos(theta),  np.sin(theta)
finewcs.wcs.pc = [[cth, -sth], [sth, cth]]
finewcs

# Check that the bottom left corner moves in the way that I expect.

xx, yy = finewcs.pixel_to_world(range(4), range(4))
xx.mean(), yy.mean()

# The translations can be acheived by changing either CRPPIX or CRVAL. I think that CRVAL makes more sense, since that is equivalent to moving the psf center around on the coarse grid, which is intuitively what we want.

psf["OVERSAMP"].header

# ### Autoconfiguration of instrumental setup (Are we ok using drizzled image?)
#
# We should really be using the stage 2 images of the individual exposures, but that would mean redoing the drizzling step ourselves, which would be annoying. 
#
# We can try to do the magic auto-detection of the instrument configuration that WebbPSF offers:

filename = str(datapath / "wr124-jwst-nircam-2022-f090w.fits")
inst = webbpsf.setup_sim_to_match_file(filename)

# Wow, that actually seems to have worked. It has grabbed the right filter and detector, and has downloaded the wavefront measurements for the date of the observations. 
#
# It has not complained about it being stage 3 rather than stage 2 product. 
#
# The dithering pattern is INTRAMODULEBOX with 8 pointings, similar to those described [here](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-operations/nircam-dithers-and-mosaics/nircam-primary-dithers#NIRCamPrimaryDithers-INTRAMODULEBOXdithers) 
#
# More notes in the org file

psf2 = inst.calc_psf(fov_pixels=512)

psf2.writeto(str(datapath / "psf-512-nircam-f090w.fits"), overwrite=True)

# So I made a new psf, which took about 15 sec. Now compare it with the previous one

impsf2 = psf2["DET_SAMP"].data

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(impsf2 / scale_psf, norm=norm, cmap=cmap)
ax[0, 1].imshow(impsf / scale_psf, norm=norm, cmap=cmap)
vscale = 10
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    (imobs / scale_obs), 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    (impsf2 / scale_psf) - (impsf / scale_psf), 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

# So there are differences, but they are small. 

# +
fig,  ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
xcutpsf2 = np.nanmean(impsf2[:, 256-w:256+w], axis=1) / scale_psf
xcutpsf =np.nanmean(impsf[:, 256-w:256+w], axis=1) / scale_psf
ax[0].plot(xcutpsf2)
ax[0].plot(xcutpsf)
ax[0].set(yscale="log", ylim=[0.3, 5000.0])
ycutpsf2 = np.nanmean(impsf2[256-w:256+w, :], axis=0) / scale_psf
ycutpsf = np.nanmean(impsf[256-w:256+w, :], axis=0) / scale_psf
ax[1].plot(ycutpsf2)
ax[1].plot(ycutpsf)
ax[1].set(yscale="log", ylim=[0.3, 5000.0])
ax[2].plot(
    ycutpsf2 - ycutpsf,
    color="m",
)
ax[2].plot(
    xcutpsf2 - xcutpsf,
    color="c",
)
ax[2].set_yscale("symlog", linthresh=0.3, linscale=1)

...;
# -

# Blue is the new one, orange the old one. Magenta is horizontal cut, cyan is vertical cut.

# #### Detector effects
#
# Charge diffusion sees to be the most important. It is included as a gaussian smoothing of about 0.2 pixels. This is included in the `DET_DIST` and `OVERDIST` extensions. Let us see if it makes any difference. 

impsf2d = psf2["DET_DIST"].data

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(impsf2 / scale_psf, norm=norm, cmap=cmap)
ax[0, 1].imshow(impsf2d / scale_psf, norm=norm, cmap=cmap)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    (imobs / scale_obs), 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    (impsf2 / scale_psf) - (impsf2d / scale_psf), 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

w = 1
cslice = slice(256-w, 256+w)
offset = np.arange(512) - 255.5
xcutpsf2 = np.nanmean(impsf2[:, cslice], axis=1) / scale_psf
ycutpsf2 = np.nanmean(impsf2[cslice, :], axis=0) / scale_psf
xcutpsf2d = np.nanmean(impsf2d[:, cslice], axis=1) / scale_psf
ycutpsf2d = np.nanmean(impsf2d[cslice, :], axis=0) / scale_psf

# +
fig,  ax = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
ax[0].plot(offset, xcutpsf2, drawstyle="steps-mid", label="DET_SAMP")
ax[0].plot(offset, xcutpsf2d, drawstyle="steps-mid", label="DET_DIST")
ax[0].set(yscale="log", ylim=[0.3, 5e6], ylabel="x=0 cut")
ax[1].plot(offset, ycutpsf2, drawstyle="steps-mid", label="DET_SAMP")
ax[1].plot(offset, ycutpsf2d, drawstyle="steps-mid", label="DET_DIST")
ax[1].set(yscale="log", ylim=[0.3, 5e6], ylabel="y=0 cut")
ax[2].plot(
    offset, 
    (ycutpsf2 - ycutpsf2d),
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[2].plot(
    offset, 
    (xcutpsf2 - xcutpsf2d),
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[2].set_ylabel("absolute difference")
ax[3].plot(
    offset, 
    (ycutpsf2 - ycutpsf2d) / ycutpsf2,
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[3].plot(
    offset, 
    (xcutpsf2 - xcutpsf2d) / xcutpsf2,
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[3].set_ylabel("relative difference")
for _ax in ax:
    _ax.axvspan(-1, 1, color="0.8")
    _ax.axvline(0.0, color="k", linewidth=0.3)
    _ax.legend()
for _ax in ax[2:]:
    _ax.axhline(0.0, color="k", linewidth=0.3)
ax[2].set_yscale("symlog", linthresh=20, linscale=1)
ax[3].set_yscale("linear")
ax[0].set_xscale("symlog", linthresh=16, linscale=1)
#ax[0].set_xticks([-256, -64, -16, -4, -1, 1, 4, 16, 64, 256])

...;
# -

# So the differences are pretty small. 
#
# Now compare this `DET_DIST` psf with the observations.

scale_obs, scale_psf = 1.0, 2.8e-8

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(imobs / scale_obs, norm=norm, cmap=cmap)
ax[0, 1].imshow(impsf2d / scale_psf, norm=norm, cmap=cmap)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    (imobs / scale_obs), 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    (imobs / scale_obs) - (impsf2d / scale_psf), 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

# So we can do a reasonable job of removing the diffraction spikes, but at the cost of over-subtracting the central core, as can be seen in these vertical and horizontal profiles

w = 32
#cslice = slice(160-w, 160+w)
cslice = slice(256-w, 256+w)
offset = np.arange(512) - 255.5
masked_impsf2d = np.where(np.isfinite(imobs), impsf2d, np.nan)
xcutobs = np.nanmean(imobs[:, cslice], axis=1) / scale_obs
ycutobs = np.nanmean(imobs[cslice, :], axis=0) / scale_obs
xcutpsf2d = np.nanmean(masked_impsf2d[:, cslice], axis=1) / scale_psf
ycutpsf2d = np.nanmean(masked_impsf2d[cslice, :], axis=0) / scale_psf

# +
fig,  ax = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
ax[0].plot(offset, xcutobs, drawstyle="steps-mid", label="obseved")
ax[0].plot(offset, xcutpsf2d, drawstyle="steps-mid", label="DET_DIST")
ax[0].set(yscale="log", ylim=[0.3, 5e6], ylabel="x=0 cut")
ax[1].plot(offset, ycutobs, drawstyle="steps-mid", label="observed")
ax[1].plot(offset, ycutpsf2d, drawstyle="steps-mid", label="DET_DIST")
ax[1].set(yscale="log", ylim=[0.3, 5e6], ylabel="y=0 cut")
ax[2].plot(
    offset, 
    (ycutobs - ycutpsf2d),
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[2].plot(
    offset, 
    (xcutobs - xcutpsf2d),
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[2].set_ylabel("absolute difference")
ax[3].plot(
    offset, 
    (ycutobs - ycutpsf2d) / ycutobs,
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[3].plot(
    offset, 
    (xcutobs - xcutpsf2d) / xcutobs,
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[3].set_ylabel("relative difference")
for _ax in ax:
    _ax.axvspan(-1, 1, color="0.8")
    _ax.axvline(0.0, color="k", linewidth=0.3)
    _ax.axvline(-64.0, color="k", linewidth=0.3)
    _ax.axvline(64.0, color="k", linewidth=0.3)
    _ax.legend()
for _ax in ax[2:]:
    _ax.axhline(0.0, color="k", linewidth=0.3)
ax[2].set_yscale("symlog", linthresh=20, linscale=1)
ax[3].set_yscale("linear")
ax[0].set_xscale("symlog", linthresh=4, linscale=0.1)

...;
# -

# #### What about if we move the star on the detector?
#
# We want to see how much the psf will change if we adopt the corner pixels. We carry on with the auto-detected instrument configuration

inst.detector

inst.detector_position

inst.detector_position = 0, 0
psf3 = inst.calc_psf(fov_pixels=512)

psf3.writeto(str(datapath / "psf-512-0000-0000-nircam-f090w.fits"), overwrite=True)

impsf3d = psf3["DET_DIST"].data

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
norm = matplotlib.colors.LogNorm(vmin=0.0, vmax=1000.0)
cmap = matplotlib.cm.get_cmap("magma").copy()
cmap.set_bad("gray")
ax[0, 0].imshow(impsf3d / scale_psf, norm=norm, cmap=cmap)
ax[0, 1].imshow(impsf2d / scale_psf, norm=norm, cmap=cmap)
for _ax in ax[0, :]:
    _ax.axhline(255.5, color="k", linewidth=0.3)
    _ax.axvline(255.5, color="k", linewidth=0.3)
vscale = 20
cmap = matplotlib.cm.get_cmap("twilight").copy()
cmap.set_bad("gray")
ax[1, 0].imshow(
    (imobs / scale_obs), 
    vmin=-vscale, vmax=vscale, cmap=cmap)
ax[1, 1].imshow(
    (impsf3d / scale_psf) - (impsf2d / scale_psf), 
    vmin=-vscale, vmax=vscale, cmap=cmap)

...;
# -

w = 5
cslice = slice(256-w, 256+w)
offset = np.arange(512) - 255.5
xcutpsf3d = np.nanmean(impsf3d[:, cslice], axis=1) / scale_psf
ycutpsf3d = np.nanmean(impsf3d[cslice, :], axis=0) / scale_psf
xcutpsf2d = np.nanmean(impsf2d[:, cslice], axis=1) / scale_psf
ycutpsf2d = np.nanmean(impsf2d[cslice, :], axis=0) / scale_psf

# +
fig,  ax = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
ax[0].plot(offset, xcutpsf3d, drawstyle="steps-mid", label="detector_position = 0, 0")
ax[0].plot(offset, xcutpsf2d, drawstyle="steps-mid", label="detector_position = 256, 256")
ax[0].set(yscale="log", ylim=[0.3, 5e6], ylabel="x=0 cut")
ax[1].plot(offset, ycutpsf3d, drawstyle="steps-mid", label="detector_position = 0, 0")
ax[1].plot(offset, ycutpsf2d, drawstyle="steps-mid", label="detector_position = 256, 256")
ax[1].set(yscale="log", ylim=[0.3, 5e6], ylabel="y=0 cut")
ax[2].plot(
    offset, 
    (ycutpsf3d - ycutpsf2d),
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[2].plot(
    offset, 
    (xcutpsf3d - xcutpsf2d),
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[2].set_ylabel("absolute difference")
ax[3].plot(
    offset, 
    (ycutpsf3d - ycutpsf2d) / ycutpsf2,
    color="m",
    drawstyle="steps-mid",
    label="y=0",
)
ax[3].plot(
    offset, 
    (xcutpsf3d - xcutpsf2d) / xcutpsf2,
    color="c",
    drawstyle="steps-mid",
    label="x=0",
)
ax[3].set_ylabel("relative difference")
for _ax in ax:
    _ax.axvspan(-1, 1, color="0.8")
    _ax.axvline(0.0, color="k", linewidth=0.3)
    _ax.legend()
for _ax in ax[2:]:
    _ax.axhline(0.0, color="k", linewidth=0.3)
ax[2].set_yscale("symlog", linthresh=20, linscale=1)
ax[3].set_yscale("linear")
ax[0].set_xscale("symlog", linthresh=16, linscale=1)
#ax[0].set_xticks([-256, -64, -16, -4, -1, 1, 4, 16, 64, 256])

...;
# -

# So the differences are quite small when you average over 10 pixel strips, but are larger for individual columns and rows. 
#
# For the outer spikes, it mainly seems to be a difference in the rotation distorsion term.

# #### What about the variation with the star spectrum?

# %env PYSYN_CDBS

# I had to download the phoenix spectral grids from [here](https://archive.stsci.edu/hlsp/reference-atlases) and install them 
#
# ```
#  will @ gris in ~/Work/synphot-data/grid [19:44:11] 
# $ tar xf /Users/will/Downloads/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar --strip-components=4
# ```

import synphot

# Use an O3V star as a proxy for the WR star, sice I do not have any WR models. Any hot star should have $F_\lambda \propto \lambda^{-4}$ in the infrared, so the details do not matter much.

src = webbpsf.specFromSpectralType('O3V', catalog='phoenix')

src.plot(ylog=True, xlog=True, bottom=1e16, left=300, right=1e5)

inst.detector_position

psf4 = inst.calc_psf(fov_pixels=512, source=src)

psf4.writeto(str(datapath / "psf-o3v-512-0000-0000-nircam-f090w.fits"), overwrite=True)

# This makes basically no difference at all. 

# # Empirical psf from medium brightness stars
#
# We will use the ones that are not the central star but are still bright. Look at smaller cutout stamps. 

len(bright_peaks)

im_sub = fits.open(datapath / "wr124-jwst-nircam-2022-f090w_BGSUB.fits")[0].data

# We use the bg-subtracted image for constructing the empirical psf

im_stamps = []
wstamp = 64
for j, i in bright_peaks:
    im_stamps.append(Cutout2D(im_sub, (i, j), size=wstamp, copy=True, mode="partial"))

fig, axes = plt.subplots(7, 7, sharex=True, sharey=True, figsize=(15, 15))
cmap = matplotlib.cm.get_cmap("viridis").copy()
cmap.set_bad("red")
for ax, im in zip(axes.flat, im_stamps[1:]):
    try:
        norm = av.simple_norm(im.data, percent=99.5, stretch='log')
    except:
        norm = None
    ax.imshow(im.data, norm=norm, cmap=cmap)

# These look really good and suggest that an empirical psf can be derived relatively easily. 
#
# 1. [X] We should use the BGSUB image
# 2. [ ] Interpolate away any nans
# 3. [ ] We need to find the centroid of each star to sub-pixel precision - easiest to fit a 2d gaussian to the central box
# 4. [ ] Then normalize all the images, using the peaks of the same Gaussians, and regrid to a common finer grid (pegged to the star centroid)
# 5. [ ] Do some sigma clipping on each pixel to reject outliers, such as other stars in the field
# 6. [ ] Then add them all together (unnormalized and with each one muptiplied by its corresponding weight image)
# 7. [ ] We could also propagate the error image an
#
# This should give a subsampled empirical psf, which would probably work better than the webbpsf one for the central part at least. 

# ## First fit the centroids
#

# We can do this on the big image if we make use of bounding_box to restrict the evaluation of the model. 

# ### We also want to interpolate the nans away
#
# Remember that this is only for finding the centroids, not for actually modeling the psf. I use a box kernel, since we are not interested in the interpolared values really. 

im_subi = interpolate_replace_nans(im_sub, kernel=Box2DKernel(9))

# Lots of nans remain around the edges and in the bright central star, but we need to check that it has fixed all the other bright stars:

# +
stamps = []
wstamp = 64
for j, i in bright_peaks:
    stamps.append(Cutout2D(im_subi, (i, j), size=wstamp, copy=True, mode="partial"))
    
fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(15, 15))
for ax, im in zip(axes.flat, stamps[1:]):
    try:
        norm = av.simple_norm(im.data, percent=99.5, stretch='log')
    except:
        norm = None
    ax.imshow(im.data, norm=norm, cmap=cmap)
# -

# That seems to have worked - all red cores eliminated. 

# ### Test how the bounding box works for `Gaussian2D` model

g = models.Gaussian2D(amplitude=100, x_mean=100, y_mean=20, x_stddev=1, y_stddev=2)
g.bounding_box.bounding_box()

# It seems to take a 5 sigma box around the initial central position. I am guessing that this will not move during the fitting, so we do not need to set it explicitly. If we want to make it bigger or smaller, we can just change the initial sigma. 

# ### Do the fitting
#
# It turns out that the previous two sections were unnecessary. It is better to use the original array wit nans in for the saturated sources. And the `bounding_box` does not get used in the fitting, so I use `Cutout2D()` to get a small stamp to work with 

# We do need the pixel coordinates as explicit arrays. I used to do a complicated thing with arange and griddata to get these, but is now much simpler using `np.indices()`

ypixels, xpixels = np.indices(im_subi.shape)
xpixels, ypixels


# Here is the function that does the fitting. Required arguments are `im` the full-size image and (`i0`, `j0`) the pixel coordinates of the peak that we wish to fit (for saturated stars these will be diplaced from the center, so you may want to increase `stamp_size` in those cases to ensure that the star is fully enclosed).

def fit_star_centroid(im, i0, j0, initial_sigma=1.5, stamp_size=15, fit=fitting.LMLSQFitter()):
    # Get pixel coordinates of the full image
    yfull, xfull = np.indices(im.shape)
    # Cutout stamp around the peak pixel
    imcutout = Cutout2D(im, (i0, j0), size=stamp_size, copy=True, mode="partial")
    xcutout = Cutout2D(xfull, (i0, j0), size=stamp_size, copy=True, mode="partial")
    ycutout = Cutout2D(yfull, (i0, j0), size=stamp_size, copy=True, mode="partial")
    # Initialize gaussian model
    g0 = models.Gaussian2D(
        amplitude=im[j0, i0], 
        x_mean=i0, y_mean=j0, 
        x_stddev=initial_sigma, y_stddev=initial_sigma,
    )
    # Fit model to non-NaN pixels
    m = np.isfinite(imcutout.data)
    g = fit(g0, xcutout.data[m], ycutout.data[m], imcutout.data[m])
    # Stuff extra attributes onto the cutout stamp
    imcutout.xdata = xcutout.data
    imcutout.ydata = ycutout.data
    # Save image of the fitted model (note we do not use g.render())
    imcutout.fit_data = g(imcutout.xdata, imcutout.ydata)
    return {"cutout": imcutout, "model": g}



# #### Test with the bright central star
#
# This is not a typical case

j0, i0 = bright_peaks[0]
i0, j0

cutout, model = fit_star_centroid(im_sub, i0, j0, stamp_size=511).values()

model

xc, yc = cutout.to_cutout_position((model.x_mean, model.y_mean))
fig, axes = plt.subplots(1, 2)
norm = av.simple_norm(cutout.data, percent=99, stretch='sqrt')
axes[0].imshow(cutout.data, norm=norm)
#axes[0].scatter(xc, yc, marker="x", color="r")
axes[1].imshow(cutout.fit_data, norm=norm)
ny, nx = cutout.shape
levels = model.amplitude * np.array((0.1, 0.5))
axes[0].contour(cutout.data, colors="r", levels=levels)
axes[1].contour(cutout.fit_data, colors="r", levels=levels)
for ax in axes:
    ax.axhline(yc, linewidth=0.3, color="r")
    ax.axvline(xc, linewidth=0.3, color="r")
    ax.axhline((ny-1)/2, linewidth=0.3, color="w")
    ax.axvline((nx-1)/2, linewidth=0.3, color="w")


# Better to use the original image with NaNs

# #### Apply to all the bright sources
#
# Note we cannot use `model` as a variable name since it is a package name in `astropy.modeling`

cutouts = []
gmodels = []
for j0, i0 in bright_peaks[1:]:
    rslt = fit_star_centroid(im_sub, i0, j0)
    cutouts.append(rslt["cutout"])
    gmodels.append(rslt["model"])

# #### Make a table of the fit results

dict(zip(gmodels[0].param_names, gmodels[0].parameters))

import pandas as pd

df = pd.DataFrame(dict(zip(_.param_names, _.parameters)) for _ in gmodels) 

# We are mainly interested in the average sigma, but also calculate the ellipicity a/b

df["sigma"] = np.hypot(df["x_stddev"], df["y_stddev"])
df["a_b"] = df["x_stddev"] / df["y_stddev"]

# Count the NaNs in each cutout to detect which stars are saturated

df["nan_count"] = [np.sum(~np.isfinite(_.data)) for _ in cutouts]

saturated = df["nan_count"] > 0

df[saturated]

df[~saturated]

fig, ax = plt.subplots()
ax.scatter("amplitude", "sigma", data=df[~saturated])
ax.scatter("amplitude", "sigma", data=df[saturated])
ax.axhline(1.25)
ax.set_yscale('log')

# So we can choose the stars with sigma < 1.25, since they are not saturated. 
#
# The unsaturated stars are slightly non-circular, with a/b clustered around 1.1. The saturated stars (orange have a wider range of a/b)

fig, ax = plt.subplots()
ax.scatter("amplitude", "a_b", data=df[~saturated])
ax.scatter("amplitude", "a_b", data=df[saturated])
#ax.set_ylim(0.0, 1.5)

# Note that going forward the only thing we use from these fits is the position of the centroid

# ## Then project them onto a finer grid, so we can align them and average them better
#
# One way of doing this is to setup a wcs with origin at the gaussian centroid and with 8 times over-sampling. 
#
# We also need a WCS of the entire original image, but in pixels instead of celestial coordinates. We set CRPIX = 1 and CRVAL = 0, so it gives us python-style 0-based pixel numbering. 

wcs_pix = WCS()
wcs_pix.pixel_shape = im_sub.shape
wcs_pix.wcs.crpix = [1, 1]
wcs_pix.wcs.cunit = ["pixel", "pixel"]
wcs_pix.wcs.cname = ["x", "y"]
wcs_pix.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]
wcs_pix

# ### Test of doing a cutout with the wcs

gmodel = gmodels[0]
x0, y0 = gmodel.x_mean.value, gmodel.y_mean.value
x0, y0 

# First check that the pixel WCS for the original image is working as it should (converting pixels to pixels!)

wcs_pix.world_to_pixel(x0 * u.pix, y0 * u.pix)

# Yes, that works as expected (identity operation). 
#
# So now check that the wcs is transformed correctly for the cutout image. 

cutout = Cutout2D(
    im_sub, 
    (gmodel.x_mean.value, gmodel.y_mean.value), 
    size=15, wcs=wcs_pix, copy=True, mode="partial",
)
cutout.wcs

# If all is well, then the central pixel (7, 7) should be close to the model centroid.

cutout.wcs.pixel_to_world(7, 7)


# This is as close as can be managed since the cutout is confined to integer pixels. 

# ### Function to get an oversampled, centered cutout image

def oversampled_centered_cutout(im, wcsi, xcenter, ycenter, oversample=8, size=15):
    """Produce a cutout from `im` of `size` x `size` pixels oversampled by `oversample`"""
    # Set up the output oversampled cutout wcs
    wcso = WCS()
    wcso.pixel_shape = oversample * size, oversample * size
    wcso.wcs.cdelt = [1 / oversample, 1 / oversample]
    wcso.wcs.crpix = (np.array(wcso.pixel_shape) + 1) / 2
    wcso.wcs.crval = xcenter, ycenter
    wcso.wcs.cunit = ["pixel", "pixel"]
    wcso.wcs.cname = ["x", "y"]
    wcso.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]

    # Get cutout of original pixel grid
    # We add borders of 1 pixel to make sure oversampled grid is fully enveloped
    cutout = Cutout2D(
        im, 
        (xcenter, ycenter), 
        size=size + 2, wcs=wcsi, copy=True, mode="partial",
    )    

    # Reproject onto oversampled grid
    imo = reproject_interp(
        (cutout.data, cutout.wcs), 
        wcso, 
        wcso.pixel_shape, 
        order="nearest-neighbor",
        return_footprint=False,
    )

    # retutn the oversampled image and the original cutout data
    return imo, cutout.data



imo, imc = oversampled_centered_cutout(im_sub, wcs_pix, x0, y0)

imo

fig, ax = plt.subplots(1, 2)
norm = av.simple_norm(imo, percent=99.5, stretch='log')
ax[0].imshow(imo, norm=norm)
ax[1].imshow(imc, norm=norm)

# Yay, this looks great. They look identical, except that the oversampled one has more and smaller pixels. 

# ### Apply the oversampling to all the unsaturated stars
#
# Separate the models into saturated and unsaturated stars

xy_saturated = [
    (gm.x_mean.value, gm.y_mean.value) 
    for gm, sat in zip(gmodels, saturated.tolist()) 
    if sat
]
xy_unsaturated = [
    (gm.x_mean.value, gm.y_mean.value) 
    for gm, sat in zip(gmodels, saturated.tolist()) 
    if not sat
]
len(xy_unsaturated)

# And get a stack of the oversampled images

ocutouts = np.stack(
    list(
        oversampled_centered_cutout(im_sub, wcs_pix, x0, y0, size=128, oversample=8)[0] 
        for (x0, y0) in xy_unsaturated
    )
)



# ### Normalize and sigma-clip the stack
#
# Use the central core to normalize

n = ocutouts.shape[-1]
cslice = slice(n//2 - 1, n//2 + 2)
peaks = np.mean(ocutouts[:, cslice, cslice], axis=(1, 2), keepdims=True)
ocutouts_norm = ocutouts / peaks
np.nanmax(ocutouts_norm, axis=(1, 2))

# These are mostly close to unity, which is good enough.
#
# Now do sigma clipping, which turns outlier pixels to NaNs

ocutouts_norm = sigma_clip(ocutouts_norm, sigma=3.0, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
osum = np.nansum(ocutouts_norm, axis=0)
omean = np.nanmean(ocutouts_norm, axis=0)
ostd = np.nanstd(ocutouts_norm, axis=0)
neff = ocutouts.shape[0]
osem = ostd / np.sqrt(neff)
s_n = omean / osem
norm = av.simple_norm(omean, percent=99.5, stretch='log')
ax[0].imshow(omean, norm=norm)
ax[1].imshow(osem, norm=norm)
ax[0].contour(s_n, levels=[10], colors='r', linewidths=0.5)

# So his looks pretty good. We have a s/n > 10 (red contour) out to a radius of 300 oversasmpled pixels, or 40 original pixels. I do not really believe that about the s/n. It looks better than that. 
#
# Now, propagate the NaNs back to the stackof peaks, so we can properly weight the average. 

peaks_or_nan = np.where(np.isfinite(ocutouts_norm), peaks, np.nan)

omean2 = np.nansum(ocutouts_norm * peaks_or_nan, axis=0) / np.nansum(peaks_or_nan, axis=0)
ovar2 = np.nansum(((ocutouts_norm - omean2) ** 2) * peaks_or_nan, axis=0) / np.nansum(peaks_or_nan, axis=0)
ostd2 = np.sqrt(ovar2)

# We need the effective number of observations to convert from std to s.e.m.  
#
# This can be approximated as the sum of the normalizations dividied by the maximum normalization

neff = np.sum(peaks) / np.max(peaks)
neff

osem2 = ostd2 / np.sqrt(neff)

np.max(omean2), np.max(omean)

fig, ax = plt.subplots(1, 2, figsize=(15, 8))
norm = av.simple_norm(omean, percent=99.5, stretch='log')
ax[0].imshow(omean, norm=norm)
ax[1].imshow(omean2, norm=norm)

# So the right-hand one is properly weighted, so should have slightly smaller noise. 

# +
fig, ax = plt.subplots()
ipix = 512
ax.plot(omean[:, ipix])
ax.plot(omean2[:, ipix])
ax.fill_between(np.arange(1024), omean[:, ipix] - osem[:, ipix], omean[:, ipix] + osem[:, ipix], color='b', alpha=0.3)
ax.fill_between(np.arange(1024), omean2[:, ipix] - osem2[:, ipix], omean2[:, ipix] + osem2[:, ipix], color='orange', alpha=0.3)

ax.set_yscale("symlog", linthresh=1e-5, linscale=1)

# -

slices

fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(15, 15))
for ax, im in zip(axes.flat, ocutouts_norm):
    ax.imshow(im, vmin=-0.003, vmax=0.003, cmap=cmap)

fig, axes = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(15, 15))
for ax, im in zip(axes.flat, ocutouts_norm):
    fac = np.nansum(im) / np.sum(omean2)
    ax.imshow(im - fac * omean2, vmin=-0.003, vmax=0.003, cmap=cmap)

# ### Repeat for the saturated stars
#
# Use a bigger cutout and less oversampling
#

scutouts = np.stack(
    list(
        oversampled_centered_cutout(im_sub, wcs_pix, x0, y0, size=256, oversample=4)[0] 
        for (x0, y0) in xy_unsaturated
    )
)

# For the saturated stars, we normalize by sums, although nothing is really ideal in this case if we are missing significant flux to the saturated pixels

scutouts_norm = scutouts / np.nanmax(ocutouts, axis=(1, 2), keepdims=True)

scutouts_norm = sigma_clip(scutouts_norm, sigma=3.0, axis=0)

fig, ax = plt.subplots(1, 2)
smean = np.nanmean(scutouts_norm, axis=0)
sstd = np.nanstd(scutouts_norm, axis=0) / np.sqrt(len(xy_unsaturated))
s_n = smean / sstd
norm = av.simple_norm(smean, percent=99.5, stretch='log')
ax[0].imshow(smean, norm=norm)
ax[1].imshow(sstd, norm=norm)
ax[0].contour(s_n, levels=[10], colors='r', linewidths=0.5)

# Compare the saturated and unsaturated profiles, also with the oversampled models and the central star observations

modpsf = psf2["OVERDIST"].data

fig, ax = plt.subplots(figsize=(12, 6))
ipix, width = 512, 8
sat_prof = np.nanmean(smean[:, ipix - width//2:ipix + width//2], axis=1) / smean.sum()
unsat_prof = np.nanmean(omean2[:, ipix - width:ipix + width], axis=1) / smean.sum()
mprof = np.nanmean(modpsf[:, 2 * ipix - width//2:2 * ipix + width//2], axis=1)
obs_prof = np.nanmean(imobs[:, ipix//2 - width//8:ipix//2 + width//8], axis=1)
ax.plot((np.arange(1024) - 511.5)/4, sat_prof, linewidth=0.5, 
        label=f"empirical psf ({len(xy_saturated)} saturated stars)")
ax.plot((np.arange(1024) - 511.5)/8, unsat_prof, linewidth=0.5, 
        label=f"empirical psf ({len(xy_unsaturated)} unsaturated stars)")
ax.plot((np.arange(2048) - 1023.5)/4, mprof, linewidth=0.5, label="model psf (webbpsf)")
ax.plot(
    (np.arange(512) - 255.5), 
    obs_prof * 2.5e-9, 
    drawstyle="steps-mid",
    linewidth=0.5, 
    label="central star psf",
)
ax.set_yscale("symlog", linthresh=1e-7, linscale=1)
ax.set(xlabel="original pixel offset")
ax.set_title("Vertical cuts")
ax.legend()
...;

# So they show very good agreement between saturated/unsaturated out to the full extent of the unsaturated cutouts. This suggests that I could have extended these further. 
#
# The agreement with the model PSF is less good. The model shows oscillations at the right positions but the exact shape is a bit off. However, some of the deeper dips in the empirical psf may be the negative halo that is seen (probably due to deficiencies in subtracting the background).
#
# The central star psf does not show these dips, agreeing better with the model for offsets around 25. However for offsets > 50 it follows the empirical psfs much better than the model one. Even though the empirical ones are noisy for offsets > 100

fig, ax = plt.subplots(figsize=(12, 6))
ipix, width = 512, 8
sat_prof = np.nanmean(smean[ipix - width//2:ipix + width//2, :], axis=0) / smean.sum()
unsat_prof = np.nanmean(omean2[ipix - width:ipix + width, :], axis=0) / smean.sum()
mprof = np.nanmean(modpsf[2 * ipix - width//2:2 * ipix + width//2, :], axis=0)
obs_prof = np.nanmean(imobs[ipix//2 - width//8:ipix//2 + width//8, :], axis=0)
ax.plot((np.arange(1024) - 511.5)/4, sat_prof, linewidth=0.5, 
        label=f"empirical psf ({len(xy_saturated)} saturated stars)")
ax.plot((np.arange(1024) - 511.5)/8, unsat_prof, linewidth=0.5, 
        label=f"empirical psf ({len(xy_unsaturated)} unsaturated stars)")
ax.plot((np.arange(2048) - 1023.5)/4, mprof, linewidth=0.5, label="model psf (webbpsf)")
ax.plot(
    (np.arange(512) - 255.5), 
    obs_prof * 2.5e-9, 
    drawstyle="steps-mid",
    linewidth=0.5, 
    label="central star psf",
)
ax.set_yscale("symlog", linthresh=1e-7, linscale=1)
ax.set(xlabel="original pixel offset", xlim=[-150, 150])
ax.set_title("Horizontal cuts")
ax.legend()
...;


# The horizontal cuts show even more clearly the problem with the negative halo in the empirical psfs

# ## Finally try and subtract the psf on the original pixel grid
#
# The moment of truth, where we find out how well this has worked.

# ### Function to project fine $\to$ coarse
#
# We want to do the inverse of `oversampled_centered_cutout()` where we reproject the mean oversampled psf back to a cutout of the big image at the original pixel scale. 

def coarse_cutout_from_oversampled_psf(psfim, im, wcsi, xcenter, ycenter, oversample=8, order="bilinear"):
    """Reproject `psfim` back to coarse image `im`, centered on `xcenter`, `ycenter`
    
    Also requires coarse-pixel WCS `wcsi` for original image and `oversample` factoeior.
    Returns cutout that encloses `psfim` on the original image."""

    # Set up the output oversampled cutout wcs
    wcso = WCS()
    wcso.pixel_shape = psfim.shape
    assert wcso.pixel_shape[0] == wcso.pixel_shape[1], "psfim must be square"
    assert wcso.pixel_shape[0] % oversample == 0, "psfim size must be integer multiple of oversample factor"
    size = wcso.pixel_shape[0] // oversample
    wcso.wcs.cdelt = [1 / oversample, 1 / oversample]
    wcso.wcs.crpix = (np.array(wcso.pixel_shape) + 1) / 2
    wcso.wcs.crval = xcenter, ycenter
    wcso.wcs.cunit = ["pixel", "pixel"]
    wcso.wcs.cname = ["x", "y"]
    wcso.wcs.ctype = ["pos.cartesian.x", "pos.cartesian.y"]

    # Get cutout of original pixel grid
    cutout = Cutout2D(
        im, 
        (xcenter, ycenter), 
        size=size, wcs=wcsi, copy=True, mode="partial",
    )    

    # Reproject PSF from oversampled grid back to cutout
    cutout.psfdata = reproject_interp(
        (psfim, wcso), 
        cutout.wcs, 
        (size, size), 
        order=order,
        return_footprint=False,
    )

    # return the cutout with psf attached
    return cutout


# ### Test of residuals from subtracting mean empirical psf
#
# Test it. Find a star that is close to the center

xx, yy = (np.array(xy_unsaturated) - np.array([2316.35, 2323.38])).T
iclosest = np.argmin(np.hypot(xx, yy))
iclosest, xy_unsaturated[iclosest]

# So 53 is the closest, and we get similar results for 25 

# +
x0, y0 = xy_unsaturated[53]
#x0, y0  = 2316.35, 2323.38

dx, dy = -0.00, -0.02 # for 53
# dx, dy = 0.00, 0.03 # for 25
cutout = coarse_cutout_from_oversampled_psf(omean2, im_sub, wcs_pix, x0 + dx, y0 + dy, oversample=8)
# -

cutout.wcs

norm = matplotlib.colors.SymLogNorm(linthresh=3, linscale=0.5, vmin=-5000, vmax=5000)
ticks=[-1000, -100, -10, 0, 10, 100, 1000]
cmap = matplotlib.cm.twilight
cmap

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
fac = 0.85 * np.nanmax(cutout.data) / np.nanmax(cutout.psfdata)
ax[0].imshow(cutout.data, norm=norm, cmap=cmap)
ax[1].imshow(fac * cutout.psfdata, norm=norm, cmap=cmap)
res = ax[2].imshow(cutout.data - fac * cutout.psfdata, norm=norm, cmap=cmap)
ax[0].set(xlim=[48, 80], ylim=[48, 80])
ax[0].set_title("observed")
ax[1].set_title("psf")
ax[2].set_title("residual")
fig.colorbar(
    res, ax=ax, orientation="horizontal", ticks=ticks, aspect=50,
)
...;

# So this has a systematic problem that the empirical psf is slightly too broad, so it undersubtracts in the center and over-subtracts in a ring. 
#
# Use 1d profile to investigate further

# +
fig, (ax, axx) = plt.subplots(2, 1, sharex=True)

ax.plot(cutout.data[:, 64], drawstyle="steps-mid", label="data")
ax.plot(fac * cutout.psfdata[:, 64], drawstyle="steps-mid", label='empirical psf')
ax.legend()
ax.set(xlim=[48, 80])
ax.set_yscale("symlog", linthresh=10, linscale=1)
ax.plot(cutout.data[:, 64] - fac * cutout.psfdata[:, 64], drawstyle="steps-mid", color="0.8", label="difference")

axx.plot(cutout.data[64, :], drawstyle="steps-mid", label="data")
axx.plot(fac * cutout.psfdata[64, :], drawstyle="steps-mid", label='empirical psf')
axx.set(xlim=[48, 80])
axx.set_yscale("symlog", linthresh=10, linscale=1)
axx.plot(cutout.data[64, :]- fac * cutout.psfdata[64, :], drawstyle="steps-mid", color="0.8", label="difference")


...;
# -

# Give just a tad more smoothing to the data, to match the psf better

esigx, esigy = 0.29, 0.29
kernel = Gaussian2DKernel(x_stddev=esigx, y_stddev=esigy, x_size=3, y_size=3, mode='integrate')

kernel.array

cutout.sdata = convolve(cutout.data, kernel)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))
fac = 0.9 * np.nanmax(cutout.data) / np.nanmax(cutout.psfdata)
fac = np.nansum(cutout.data) / np.nansum(cutout.psfdata)
ax[0].imshow(cutout.sdata, norm=norm, cmap=cmap)
ax[1].imshow(fac * cutout.psfdata, norm=norm, cmap=cmap)
ax[2].imshow(cutout.sdata - fac * cutout.psfdata, norm=norm, cmap=cmap)
ax[0].set(xlim=[48, 80], ylim=[48, 80])
ax[0].set_title("smoothed observed")
ax[1].set_title("psf")
ax[2].set_title("residual")
fig.colorbar(
    res, ax=ax, orientation="horizontal", ticks=ticks, aspect=50,
)
...;

# +
fig, (ax, axx) = plt.subplots(2, 1, sharex=True)


ax.plot(cutout.sdata[:, 64], drawstyle="steps-mid", label="smoothed data")
ax.plot(fac * cutout.psfdata[:, 64], drawstyle="steps-mid", label='empirical psf')
ax.legend()
ax.set(xlim=[48, 80])
ax.set_yscale("symlog", linthresh=10, linscale=1)
ax.plot(cutout.sdata[:, 64] - fac * cutout.psfdata[:, 64], drawstyle="steps-mid", color="0.8", label="difference")

axx.plot(cutout.sdata[63, :], drawstyle="steps-mid", label="smoothed data")
axx.plot(fac * cutout.psfdata[63, :], drawstyle="steps-mid", label='empirical psf')
axx.set(xlim=[48, 80])
axx.set_yscale("symlog", linthresh=10, linscale=1)
axx.plot(cutout.sdata[63, :] - fac * cutout.psfdata[63, :], drawstyle="steps-mid", color="0.8", label="difference")
# -

# So maybe we can get te residuals down to 1% with a combination of sub-pixel shifts and convolution

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fac = 0.9 * np.nanmax(cutout.data) / np.nanmax(cutout.psfdata)
fac = np.nansum(cutout.data) / np.nansum(cutout.psfdata)
residual = cutout.sdata - fac * cutout.psfdata
fnorm = matplotlib.colors.SymLogNorm(linthresh=0.01, linscale=0.5, vmin=-1, vmax=1)
im = ax[0].imshow(residual / cutout.sdata, vmin=-2, vmax=2, cmap=cmap)
ax[1].imshow(residual / cutout.sdata, vmin=-2, vmax=2, cmap=cmap)
ax[0].contour(residual / cutout.sdata, levels=[-0.3, 0.3], colors="g")
ax[1].contour(residual / cutout.sdata, levels=[-0.3, 0.3], colors="g")
ax[1].set(xlim=[48, 80], ylim=[48, 80])
ax[0].set_title("fractional residual")
ax[1].set_title("zoomed fractional residual")
fig.colorbar(
    im, ax=ax, orientation="horizontal",  aspect=50,
)
...;

# The above figure shows the fractional residual
#
# So it turns out that it works ok within a radius of about 15 in this case, but outside of that it is completely useless. This may be because we are looking at a star that is superimposed on the nebula, so there will be residual emission around it. 
#
#

np.indices(cutout.wcs.pixel_shape)

x, y = cutout.wcs.array_index_to_world(*np.indices(cutout.wcs.pixel_shape))
xx, yy = x.value - x0, y.value - y0
r = np.hypot(xx, yy)
r

fig, ax = plt.subplots()
ax.scatter(r.flat, cutout.psfdata.flat, s=1, marker='.', alpha=0.1)
ax.scatter(r.flat, (cutout.data.flat / fac) - cutout.psfdata.flat, s=1, marker='.', alpha=0.1)
ax.set(xscale="log")
ax.set_yscale("symlog", linthresh=1e-3, linscale=2)
ax.axhline(color="k", linewidth=0.5)
...;


# ## Set up a fitting framework
#
# So there are a few different ways I see to do this
#
# 1. We could just use lmfit, so we get full control over the objective function and we can get estimates of the uncertainties and correlations between parameters. Either just from the hessian or from mcmc
# 2. We could try and get our head around how astropy.fitting works. 
# 3. We could look into using photutils to make a fittable model out of an image. 
#
# To be honest, I think #1 looks like the best bet - better the devil you know!
#
# Although photutils.psf does do similiar things already

def fit_oversampled_psf()
