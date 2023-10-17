from pathlib import Path
from typing import Union
import numpy as np
from astropy.table import QTable, Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.stats import mad_std
import typer


def match_catalogs(
    c1: SkyCoord, c2: SkyCoord, max_sep: u.Quantity = 1.0 * u.arcsec
) -> tuple[np.ndarray, np.ndarray]:
    """Find indices of coincident sources between two lists of sky coordinates

    Returns a pair of index arrays of the matched sources in each catalog.
    Each source is guaranteed to appear only once.
    """
    # Find closest source in c2 to each source in c1
    idx2, d2d, d3d = c1.match_to_catalog_sky(c2)
    # Make index of sources in c1: [0, 1, 2, ...]
    idx1 = np.arange(len(c1))
    # Mask that is True when closest source is close enough for match
    isclose = d2d < max_sep
    # Remove duplicate sources in the idx2 list by making a BACKWARDS mapping from 2 -> 1
    # This will keep only the last source in c2 that was matched by a source in c1
    backmap = dict(zip(idx2[isclose], idx1[isclose]))
    # Retrieve the two matched index lists from the dict
    imatch2, imatch1 = zip(*backmap.items())
    # Return arrays of matched indices, which can be used to obtain the matched sources
    return np.asarray(imatch1), np.asarray(imatch2)


def get_coords_from_fits_catalog(fname, sname):
    """Get coordinates from sources detected in a FITS image"""
    hdulist = fits.open(f"{fname}.fits")
    for hdu in hdulist:
        # Find first HDU with data
        if hdu.data is not None:
            break
    w = WCS(hdu.header, fix=False)
    stab = QTable.read(f"{fname}-sources-{sname}.ecsv")
    return w.pixel_to_world(stab["xcentroid"], stab["ycentroid"])


def get_coords_from_gaia_catalog(fname, sname):
    stab = QTable.read(f"{fname}-sources-{sname}.ecsv")
    return SkyCoord(stab["ra"], stab["dec"])


def main(
    file_prefixes: tuple[str, str],
    catalog_suffixes: tuple[str, str],
    maximum_separation_arcsec: float = 1.0,
    maximum_radius_arcsec: Union[float, None] = None,
    minimum_radius_arcsec: Union[float, None] = None,
    object_name: str = "ngc 346",
    combo_prefix: Union[str, None] = None,
    guess_offset: float = 0.0,
    guess_pa: float = 0.0,
):
    fname1, fname2 = file_prefixes
    sname1, sname2 = catalog_suffixes
    if fname1 == "gaia":
        c1 = get_coords_from_gaia_catalog(fname1, sname1)
    else:
        c1 = get_coords_from_fits_catalog(fname1, sname1)
    if fname2 == "gaia":
        c2 = get_coords_from_gaia_catalog(fname2, sname2)
    else:
        c2 = get_coords_from_fits_catalog(fname2, sname2)

    shifted = False
    if guess_offset:
        # The guessed offset should be from 1 -> 2, so we first apply the
        # opposite offset to 2
        c2 = c2.directional_offset_by((guess_pa - 180) * u.deg, guess_offset * u.arcsec)
        shifted = True

    max_sep = maximum_separation_arcsec * u.arcsec
    ii1, ii2 = match_catalogs(c1, c2, max_sep=max_sep)
    c1m = c1[ii1]
    c2m = c2[ii2]

    if shifted:
        # Now undo the shift that we did before the matching
        c2m = c2m.directional_offset_by(guess_pa * u.deg, guess_offset * u.arcsec)

    # Offsets between two images
    offsets = c1m.spherical_offsets_to(c2m)
    ra, dec = [_.milliarcsecond for _ in offsets]
    # Offsets from the source
    c0 = SkyCoord.from_name(object_name)
    offsets0 = c0.spherical_offsets_to(c1m)
    ra0, dec0 = [_.arcsecond for _ in offsets0]
    r = c0.separation(c1m).arcsec

    # Optionally restrict to range of radii from source
    mask = np.ones_like(r, dtype=bool)
    if maximum_radius_arcsec is not None:
        mask = mask & (r < maximum_radius_arcsec)
    if minimum_radius_arcsec is not None:
        mask = mask & (r > minimum_radius_arcsec)
    ra = ra[mask]
    dec = dec[mask]
    ra0 = ra0[mask]
    dec0 = dec0[mask]
    r = r[mask]

    n = len(ra)
    print(f"Statistics based on {n} coincident sources")
    print(
        f"Mean displacement in RA: {np.mean(ra):.2f} +/- {np.std(ra) / np.sqrt(n):.2f} marcsec"
    )
    print(
        f"Mean displacement in Dec: {np.mean(dec):.2f} +/- {np.std(dec) / np.sqrt(n):.2f} marcsec"
    )
    print(
        f"Median displacement in RA: {np.median(ra):.2f} +/- {mad_std(ra) / np.sqrt(n):.2f} marcsec"
    )
    print(
        f"Median displacement in Dec: {np.median(dec):.2f} +/- {mad_std(dec) / np.sqrt(n):.2f} marcsec"
    )

    # Save table of offsets for later analysis
    if combo_prefix is None:
        combo_prefix = f"{fname1}-TO-{fname2}"
    Table(
        data={
            "RA, arcsec": ra0,
            "Dec, arcsec": dec0,
            "d RA, mas": ra,
            "d Dec, mas": dec,
            "Radius, arcsec": r,
        },
    ).write(f"{combo_prefix}-OFFSETS.ecsv", format="ascii.ecsv", overwrite=True)


if __name__ == "__main__":
    typer.run(main)
