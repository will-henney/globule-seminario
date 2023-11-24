from __future__ import annotations
from typing import Optional
import typer
from pathlib import Path
import numpy as np
from astropy.coordinates import SpectralCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from filter_throughput import get_filter_throughput


def main(
    filter_name: str,
    cube_path: Path,
    hdu_key: str = "SCI",
    out_prefix: Optional[str] = None,
    out_path: Path = Path.cwd() / "FilterMaps",
    verbose: bool = True,
):
    """For a given filter make a simulated brightness map from a spectral cube

    Calculates the mean brightness across the filter bandpass
    """

    # Ensure that output path exists
    out_path.mkdir(exist_ok=True)

    # Read datacube
    hdulist = fits.open(cube_path)
    hdu = hdulist[hdu_key]
    cube = hdu.data
    nv, ny, nx = cube.shape
    wcs = WCS(hdu.header, fix=False)

    # Get wavelengths from SpectralCoordinate axis
    waves = wcs.spectral.pixel_to_world(range(nv)).to(u.micron)
    # Effective transmission at each wavelength
    eff = get_filter_throughput(waves, filter_name)
    # Check that the filter overlaps with this cube
    assert np.max(eff) > 1e-6, "No overlap of filter band pass with cube"
    if verbose:
        print(f"Maximum throughput: {np.max(eff):.6f} at {waves[np.argmax(eff)]:.3f}")

    # Average brightness through cube
    im = np.nansum(cube * eff[:, None, None], axis=0) / np.sum(eff)

    # Construct a minimal FITS header for image
    hdr = wcs.celestial.to_header()
    # Assume original cube is in MJy/sr unless otherwise specified
    hdr["BUNIT"] = hdu.header.get("BUNIT", "MJy/sr")

    # Construct an output filename if none provided
    if out_prefix is None:
        out_prefix = f"{cube_path.stem}-{filter_name}"
    out_file = f"{out_prefix}.fits"
    if verbose:
        print("Writing filter image to", f"{out_path.stem}/{out_file}")

    # Save the image to file
    fits.PrimaryHDU(header=hdr, data=im).writeto(out_path / out_file, overwrite=True)


if __name__ == "__main__":
    typer.run(main)
