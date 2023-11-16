from __future__ import annotations
from typing import Optional
import typer
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable


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
    noneb_cube_suffix: str = "median-cont-0021",
    star_name: str = "wr124",
    radius_arcsec: float = 0.1,
    hdu_key: str = "SCI",
    verbose: bool = True,
):
    """Use PSF to remove the instrumental scattered starlight from spectral cube

    The PSF is previously generated from find-cube-psf.py
    """

    # The original spectral cube
    cube_path = Path(f"{prefix}_s3d.fits")
    # The cube after filtering out the nebular lines
    noneb_cube_path = Path(f"{prefix}_s3d-{noneb_cube_suffix}.fits")
    #  The cube of the empirically determined PSF
    psf_cube_path = Path(f"{prefix}-PSF.fits")

    hdulist = fits.open(cube_path)
    hdu = hdulist[hdu_key]
    cube = hdu.data
    nv, ny, nx = cube.shape

    # Repeat for noneb spectrum and psf (assume that header and wcs
    # are always the same as original cube)
    noneb_cube = fits.open(noneb_cube_path)[hdu_key].data
    psf_cube = fits.open(psf_cube_path)[hdu_key].data
    assert psf_cube.shape == noneb_cube.shape == cube.shape

    # Construct coordinate array
    origin = SkyCoord.from_name(star_name)
    wcs = WCS(hdu.header)
    coords = wcs.celestial.pixel_to_world(*np.meshgrid(range(nx), range(ny)))
    radii = origin.separation(coords)
    # Select all spaxels within a certain radius of the star
    star_mask = radii < radius_arcsec * u.arcsec

    # We will take the average over those spaxels of the
    # nebular-filtered cube to be the stellar spectrum
    star_spec = np.nansum(
        noneb_cube * star_mask[None, :, :].astype(float),
        axis=(1, 2),
    ) / np.sum(star_mask)

    # Construct wavelength array
    waves = get_wavelength_array(wcs)

    # Save the 1D line-plus-continuum stellar spectrum in case needed for anything
    tab = QTable({"Wavelength": waves, "Intensity": star_spec})
    tab.write(f"{prefix}-star-full-spectrum.ecsv", format="ascii.ecsv", overwrite=True)

    # Now we combine the 1D star spectrum with the 3D PSF to get the
    # 3D stellar contamination cube
    #
    # The PSF is currently normalized so that the sum over each image
    # channel is unity, but we will not assume that is true.  We want
    # to renormalize it so that the sum over the star_mask region is
    # unity
    psf_norm = np.nansum(
        psf_cube * star_mask[None, :, :].astype(float),
        axis=(1, 2),
    ) / np.nansum(psf_cube, axis=(1, 2))

    # Then the stellar cube is just the product of these, divided by
    # the normalization
    star_cube = (
        psf_cube
        * star_spec[:, None, None]
        * np.sum(star_mask)
        / psf_norm[:, None, None]
    )

    # Write out the new cubes
    for data, label, long_label in [
        (star_cube, "star", "star contribution to cube"),
        (cube - star_cube, "nostar", "star-subtracted nebula cube"),
    ]:
        out_file = f"{prefix}-{label}.fits"
        if verbose:
            print("Saving", long_label, "to", out_file)
        fits.PrimaryHDU(header=hdu.header, data=data).writeto(
            out_file,
            overwrite=True,
        )


if __name__ == "__main__":
    typer.run(main)
