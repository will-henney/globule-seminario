from __future__ import annotations
import typer
from typing import Optional, Union
from pathlib import Path
import yaml
import numpy as np
import npyaml
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.table import QTable
import regions as rg


def get_spectrum_from_region(
    cube: np.ndarray,
    region: rg.PixelRegion | rg.SkyRegion,
    wcs: Optional[WCS] = None,
    sum_method: callable = np.nansum,
    debug: bool = False,
) -> np.ndarray:
    """Obtain one-dimensional spectrum of 2D region from 3D cube

    Efficient method that first slices the cube with the smallest
    possible cutout box that encloses fully encloses the region. This
    means that the region mask does not need to be applied to the
    entire cube

    """
    # Adapted from a similar function in muse-hii-regions project
    # (from file notebooks/ngc346-new/00-check-peter-cube.py)
    assert len(cube.shape) == 3, "Spectral cube must be 3-dimensional"
    try:
        # Case of pixel region
        region_mask = region.to_mask()
    except AttributeError:
        # Case of sky region ...
        assert wcs is not None, "For sky regions, wcs must not be None"
        region_mask = region.to_pixel(wcs.celestial).to_mask()
    nv, ny, nx = cube.shape
    assert ny, nx == wcs.celestial.pixel_shape
    # Slices into 2D arrays
    slices_large, slices_small = region_mask.get_overlap_slices((ny, nx))
    if debug:
        print(region.meta["label"])
        print("2D slice:", slices_large)
    slices_cube = (slice(None, None),) + slices_large
    image_mask_large = region_mask.to_image((ny, nx))
    image_mask_small = image_mask_large[slices_large]
    cube_cutout = cube[slices_cube]
    spec = sum_method(
        cube_cutout * image_mask_small[None, :, :], axis=(1, 2)
    ) / sum_method(image_mask_small)
    return spec


def get_wavelength_array(wcs: WCS):
    """
    Get explicit array of wavelengths corresponding to a wcs
    """
    # Isolate wavelength part of WCS
    wspec = wcs.spectral
    # How many pixels along wavelength axis
    nwave = wspec.pixel_shape[0]
    # Return the SpectralCoord array
    return wspec.pixel_to_world(range(nwave))


def main(
    cube_file: Path,
    region_file: Path,
    save_prefix: Optional[str] = None,
    debug: bool = False,
):
    """Extract one-dimensional spectra from spectral cube

    Calculates one-dimensional spectra from line cube for extraction
    regions read from DS9 file.

    """
    # Read in spectral cube
    cube_hdu = fits.open(cube_file)["SCI"]
    wcs = WCS(cube_hdu.header, fix=False)

    # Obtain wavelength array in microns
    waves = get_wavelength_array(wcs).to(u.micron)
    # Extract spectrum of each region with correct units
    spec_dict = {
        reg.meta["label"]: u.Quantity(
            get_spectrum_from_region(cube_hdu.data, reg, wcs, debug=debug),
            unit=cube_hdu.header["BUNIT"],
        )
        for reg in rg.Regions.read(region_file)
    }

    # Save results as table
    tab = QTable({"Wavelength": waves, **spec_dict})
    if save_prefix is None:
        save_prefix = cube_file.stem
    tab.write(save_prefix + ".ecsv", format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    typer.run(main)
