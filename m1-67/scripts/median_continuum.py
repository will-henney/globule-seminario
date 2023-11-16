"""Determine spectral continuum by median filtering

These are based on earlier routines that I wrote for the
00-check-peter-cube notebook. The difference is that this time I will
not use the mpdaf library, so it should be more general and more
efficient

Will Henney: 2022-10-04

CHANGES 2023-11-03: Made to be more general in filename and directory
handling. Now cube_path argument is required and output is written to
current directory by default, with name derived from stem of input
file.

"""
from __future__ import annotations
from typing import Optional
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndi
import typer


def get_median_continuum(
    data: np.ndarray,
    window_size=11,
    percentile: Optional[float] = None,
):
    """Take windowed median along first axis of data array"""
    ndim = len(data.shape)
    # Conform the window size parameter to the shape of the data
    size = (window_size,) + (1,) * (ndim - 1)
    if percentile is None:
        return ndi.median_filter(data, size=size, mode="nearest")
    else:
        return ndi.percentile_filter(data, percentile, size=size, mode="nearest")


def main(
    window_size: int,
    cube_path: Path,
    out_label: str = "median",
    save_path: Path = Path.cwd(),
    two_pass: bool = False,
    percentile: Optional[float] = None,
    first_window_size: int = 11,
    shave_threshold: float = 0.1,
    hdu_key: str = "SCI",
    hdu_index: Optional[int] = None,
    verbose: bool = True,
):
    """Find and remove continuum from cube by median filtering"""

    hdulist = fits.open(cube_path)
    if hdu_index is not None:
        hdu = hdulist[hdu_index]
    else:
        hdu = hdulist[hdu_key]

    if two_pass:
        # First filter pass
        cont_data = get_median_continuum(hdu.data, first_window_size, percentile)
        # Save off the lines that go more than shave_threshold above continuum
        shaved_data = np.minimum(hdu.data, cont_data * (1.0 + shave_threshold))
        # Second filter pass
        cont_data = get_median_continuum(shaved_data, window_size, percentile)
    else:
        cont_data = get_median_continuum(hdu.data, window_size, percentile)

    # Write out the new cubes
    for data, label, long_label in [
        (cont_data, "cont", "median filtered cube"),  # Continuum
        (hdu.data - cont_data, "csub", "residual cube"),  # Original minus continuum
        (hdu.data / cont_data, "cdiv", "ratio cube"),  # Original over continuum
    ]:
        out_file = f"{cube_path.stem}-{out_label}-{label}-{window_size:04d}.fits"
        if verbose:
            print("Saving", long_label, "to", out_file)
        fits.PrimaryHDU(header=hdu.header, data=data).writeto(
            out_file,
            overwrite=True,
        )


if __name__ == "__main__":
    typer.run(main)
