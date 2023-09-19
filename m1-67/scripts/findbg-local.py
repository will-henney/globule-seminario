import sys
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from astropy.wcs import WCS

fname = sys.argv[1]

hdulist = fits.open(fname)
for hdu in hdulist:
    if hdu.data is not None:
        # Choose the first hdu that has some data
        break


subsample = slice(None, None), slice(None, None)
data = hdu.data.astype(float)[subsample]
data[data == 0.0] = np.nan
# If there is a weight image then use it to cut out noisy pixels
if "WHT" in hdulist:
    data[hdulist["WHT"].data <= 0.0] = np.nan

hdr = WCS(hdu.header).slice(subsample, numpy_order=True).to_header()

bkg = Background2D(
    data,
    box_size=(15, 15),
    filter_size=(3, 3),
    sigma_clip=SigmaClip(sigma=3.0),
    bkg_estimator=MedianBackground(),
)

# Mask out on the BG image all regions that are masked out on the original
bkg.background[~np.isfinite(data)] = np.nan
bkg.background_rms[~np.isfinite(data)] = np.nan

# Save BG, BG-subtracted original, and BG sigma
for im, label in [
    [bkg.background, "BG"],
    [data - bkg.background, "BGSUB"],
    [bkg.background_rms, "BGSIG"],
]:
    fits.PrimaryHDU(data=im, header=hdr).writeto(
        fname.replace(".fits", f"_{label}.fits"),
        overwrite=True,
    )
