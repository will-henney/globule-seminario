from __future__ import annotations
from typing import Optional
import typer
from pathlib import Path
import slugify
import numpy as np
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs import WCS
import astropy.units as u
from astropy import constants
from astropy.io import fits
from astropy.table import Table


def get_waveslice(wave0: u.Quantity, wcs: WCS, vrange: u.Quantity) -> slice:
    """
    Get the pixel slice along the spectral axis of `wcs` that
    correponds to a given velocity range `vrange` and rest wavelength
    `wave0`

    Raises IndexError if slice is out of bounds
    """
    assert len(vrange) == 2, "Velocity range must have 2 elements"
    waves = wave0 * (1.0 + vrange / constants.c.to(u.km / u.s))
    indices = wcs.spectral.world_to_array_index_values(waves.to_value(u.m))
    if min(indices) < 0 or max(indices) > wcs.spectral.pixel_shape[0]:
        raise IndexError("Velocity range gives indices that are out of bounds")
    return slice(*indices)


def main(
    cube_path: Path,
    hdu_key: str = "SCI",
    line_table_path: Path = Path("jwst-miri-lines.tab"),
    velocity_range_kms: tuple[float, float] = (0.0, 400.0),
    out_path: Path = Path.cwd() / "LineMaps",
    verbose: bool = True,
):
    """Construct a brightness map for each line in table

    Each line is integrated over VELOCITY_RANGE_KMS
    """

    # Ensure that output path exists
    out_path.mkdir(exist_ok=True)

    # Read datacube
    hdulist = fits.open(cube_path)
    hdu = hdulist[hdu_key]
    cube = hdu.data
    nv, ny, nx = cube.shape
    wcs = WCS(hdu.header)

    # Read line db
    tab = Table.read(line_table_path, format="ascii.tab")

    # Extraction window is heliocentric velocity in km/s
    vrange = velocity_range_kms * u.km / u.s

    # Get SpectralCoordinate axis
    scoords = wcs.spectral.pixel_to_world(range(nv)).replicate(
        doppler_convention="optical"
    )

    # For each line in DB that is within the wavelength range, extract
    # map from cube
    for linedata in tab:
        # The only columns we use from the table are the label and the
        # rest wavelength
        species = linedata["ID"]
        wave0 = linedata["Wave_0"] * u.micron
        channel = linedata["Ch"]
        label = slugify.slugify(
            f"{species} lambda {wave0:07.4f} CH{channel:02d}",
            lowercase=False,
            separator="-",
        )
        try:
            waveslice = get_waveslice(wave0, wcs, vrange)
        except IndexError:
            # Skip over any line that is not in this cube
            continue
        except u.UnitsError as err:
            print("Failed with", wave0, err)
            continue
        # Find the velocity array
        velocities = (
            scoords[waveslice].to(u.km / u.s, doppler_rest=wave0).quantity.value
        )
        # Portion of cube corresponding to vrange
        subcube = cube[waveslice, ...]

        # Find the moments ...
        # ... 0: sum
        mom0 = np.sum(subcube, axis=0)
        # ... 1: mean
        mom1 = np.sum(subcube * velocities[:, None, None], axis=0) / mom0
        # ... 2: sigma
        mom2 = np.sqrt(
            np.sum(subcube * (velocities[:, None, None] - mom1) ** 2, axis=0) / mom0
        )

        # Multiply by pixel delta wave to convert to integral over wavelength
        dwave = (wcs.spectral.pixel_scale_matrix[0, 0] * u.m).to_value(u.micron)
        mom0 *= dwave

        hdr = wcs.celestial.to_header()
        # Assume original cube is in MJy/sr unless otherwise specified
        cube_bunit = hdu.header.get("BUNIT", "MJy/sr")
        hdr["BUNIT"] = f"micron.{cube_bunit}"

        # Save to files
        for moment, suffix in [mom0, "bsum"], [mom1, "vmean"], [mom2, "sigma"]:
            fits.PrimaryHDU(header=hdr, data=moment).writeto(
                out_path / f"{label}_{suffix}.fits", overwrite=True
            )


if __name__ == "__main__":
    typer.run(main)
