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


def get_waveslice(waverange: u.Quantity, wcs: WCS) -> slice:
    """
    Get the pixel slice along the spectral axis of `wcs` that
    correponds to a given wavelength range `waverange`

    Raises IndexError if slice is out of bounds
    """
    # This is simpler than the spectral line version, since we do not
    # need to deal with velocities
    assert len(waverange) == 2, "Wavelength range must be a 2-tuple"
    indices = wcs.spectral.world_to_array_index_values(waverange.to_value(u.m))
    if min(indices) < 0 or max(indices) > wcs.spectral.pixel_shape[0]:
        raise IndexError("Wavelength range gives indices that are out of bounds")
    return slice(*indices)


def main(
    cube_path: Path,
    hdu_key: str = "SCI",
    band_table_path: Path = Path("jwst-miri-bands.tab"),
    out_path: Path = Path.cwd() / "BandMaps",
    verbose: bool = True,
):
    """Construct a brightness map for each wave band in table

    Each line is integrated over wave range read in from table
    """

    # Ensure that output path exists
    out_path.mkdir(exist_ok=True)

    # Read datacube
    hdulist = fits.open(cube_path)
    hdu = hdulist[hdu_key]
    cube = hdu.data
    nv, ny, nx = cube.shape
    wcs = WCS(hdu.header)

    # Read band db
    tab = Table.read(band_table_path, format="ascii.tab")

    # Get SpectralCoordinate axis
    scoords = wcs.spectral.pixel_to_world(range(nv)).replicate(
        doppler_convention="optical"
    )

    # For each band in DB that is within the wavelength range, extract
    # map from cube
    for banddata in tab:
        # The only columns we use from the table are the label and the
        # rest wavelength
        band = banddata["Band"]
        channel = banddata["Ch"]
        label = slugify.slugify(
            f"{band} micron CH{channel:02d}",
            lowercase=False,
            separator="-",
        )
        waverange = (banddata["wavmin"], banddata["wavmax"]) * u.micron
        try:
            waveslice = get_waveslice(waverange, wcs)
        except IndexError:
            # Skip over any band that is not in this cube
            continue
        except u.UnitsError as err:
            print("Failed with", waverange, err)
            continue

        # Find the wavelength array
        waves = scoords[waveslice].to(u.micron).quantity.value

        # Portion of cube corresponding to vrange
        subcube = cube[waveslice, ...]

        # To be honest, there is little need for any but the 0 moment,
        # but we will find 1 and 2 too. Maybe there is interesting
        # variation in the AIB shape ...

        # ... 0: sum
        mom0 = np.sum(subcube, axis=0)
        # ... 1: mean
        mom1 = np.sum(subcube * waves[:, None, None], axis=0) / mom0
        # ... 2: sigma
        mom2 = np.sqrt(
            np.sum(subcube * (waves[:, None, None] - mom1) ** 2, axis=0) / mom0
        )

        # Save to files
        for moment, suffix in [mom0, "sum"], [mom1, "wavmean"], [mom2, "sigma"]:
            fits.PrimaryHDU(header=wcs.celestial.to_header(), data=moment).writeto(
                out_path / f"{label}_{suffix}.fits", overwrite=True
            )


if __name__ == "__main__":
    typer.run(main)
