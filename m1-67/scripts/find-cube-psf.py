from __future__ import annotations
from typing import Optional
import typer
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits
from numpy.polynomial import Chebyshev as T
import itertools


def get_wavelength_array(wcs: WCS):
    """
    Get explicit array of wavelengths corresponding to a wcs

    Return value is explicitly in microns
    """
    # Isolate wavelength part of WCS
    wspec = wcs.spectral
    # How many pixels along wavelength axis
    nwave = wspec.pixel_shape[0]
    # Return the SpectralCoord array
    return (wspec.pixel_to_world_values(range(nwave)) * u.m).to(u.micron)


def main(
    prefix: str,
    mid_pass_suffix: str = "median-cont-0021-median-csub-0151",
    star_name: str = "wr124",
    radius_arcsec: float = 0.1,
    min_line_strength: float = 0.1,
    hdu_key: str = "SCI",
    polynomial_degree: int = 2,
    max_radius_arcsec: float = 2.0,
    max_wavelength_micron: Optional[float] = None,
    verbose: bool = True,
):
    """Construct empirical stellar psf from spectral cube

    The cube should have been mid-pass filtered to isolate the broad
    stellar wind lines
    """
    cube_path = Path(f"{prefix}_s3d-{mid_pass_suffix}.fits")

    hdulist = fits.open(cube_path)
    hdu = hdulist[hdu_key]

    # The data cube
    cube = hdu.data
    nv, ny, nx = cube.shape

    # Construct coordinate array
    origin = SkyCoord.from_name(star_name)
    wcs = WCS(hdu.header)
    coords = wcs.celestial.pixel_to_world(*np.meshgrid(range(nx), range(ny)))
    radii = origin.separation(coords)
    # Select all pixels within a certain radius of the star
    star_mask = radii < radius_arcsec * u.arcsec
    # Construct mean spectrum of star
    star_spec = np.nansum(
        cube * star_mask[None, :, :].astype(float),
        axis=(1, 2),
    ) / np.sum(star_mask)
    # Construct wavelength array
    waves = get_wavelength_array(wcs)

    # Save the 1D stellar wind line spectrum in case needed for debugging
    tab = QTable({"Wavelength": waves, "Intensity": star_spec})
    tab.write(f"{prefix}-star-line-spectrum.ecsv", format="ascii.ecsv", overwrite=True)

    # To define the PSF, we want to use only the peaks in the star
    # line spectrum, so we select those wavelengths where star line
    # spectrum is brighter than min_line_strength times the maximum
    peaks_mask = star_spec > min_line_strength * np.max(star_spec)
    # We have an option to cut off spectrum past a certain wavelength
    # to avoid edge effects
    if max_wavelength_micron is not None:
        peaks_mask = peaks_mask & (waves <= max_wavelength_micron * u.micron)

    # At each spaxel, fit a polynomial in wavelength to the selected
    # parts of the cube
    psf_cube = np.empty_like(cube)
    for j, i in itertools.product(range(ny), range(nx)):
        # Check for NaNs
        mask = peaks_mask & np.isfinite(cube[:, j, i])
        # Check we have enough valid wavelength points and we are not
        # too far out in the wings
        if (
            np.sum(mask) > polynomial_degree + 1
            and radii[j, i] <= max_radius_arcsec * u.arcsec
        ):
            # Extract relative spectrum of PSF
            spaxel_spec = cube[:, j, i][mask] / star_spec[mask]
            # Weighted according to brightness (docs say to weight by 1/sigma)
            weights = np.sqrt(cube[:, j, i][mask])
            # Fit polynomial
            try:
                # Attempt to perform the polynomial fit
                p = T.fit(
                    waves.value[mask], spaxel_spec, deg=polynomial_degree, w=weights
                )
                # and fill in the full PSF spectrum of this spaxel
                psf_cube[:, j, i] = p(waves.value)
            except np.linalg.LinAlgError:
                # If that failed, try a constant value of the average
                psf_cube[:, j, i] = np.sum(cube[:, j, i][mask]) / np.sum(
                    star_spec[mask]
                )
        else:
            try:
                psf_cube[:, j, i] = np.sum(cube[:, j, i][mask]) / np.sum(
                    star_spec[mask]
                )
            except:
                # Spaxels with no data at all
                psf_cube[:, j, i] = np.nan

    # Normalize to give sum unity over each channel image
    psf_cube /= np.nansum(psf_cube, axis=(1, 2), keepdims=True)

    # Write to FITS file
    fits.PrimaryHDU(header=hdu.header, data=psf_cube).writeto(
        f"{prefix}-PSF.fits", overwrite=True
    )


if __name__ == "__main__":
    typer.run(main)
