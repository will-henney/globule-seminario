from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field
import typer
from pathlib import Path
import numpy as np
from astropy.io import fits


def get_first_image_hdu(hdulist):
    """
    Try each hdu in hdulist in turn, returning the first 2D image found
    """
    for hdu in hdulist:
        if hdu.data is not None and len(hdu.data.shape) == 2:
            return hdu
    return None


@dataclass
class Component:
    """An emission component that contributes to a filter"""

    label: str
    filter_name: str
    channel: int
    data_path: Path
    prefix: str
    file_path: Path = field(init=False)
    hdu: fits.PrimaryHDU | fits.ImageHDU = field(init=False)

    def __post_init__(self):
        self.file_path = (
            self.data_path
            / f"{self.prefix}-{self.label}-{self.filter_name}-ch{self.channel}.fits"
        )
        self.hdu = get_first_image_hdu(fits.open(self.file_path))
        assert (
            self.hdu is not None
        ), f"No image data found in FITS file: {self.file_path.name}"

    def save_ratio(self, total: Component) -> None:
        ratio = self.hdu.data / total.hdu.data
        out_path = self.file_path.with_stem(self.file_path.stem + "_frac")
        fits.PrimaryHDU(header=self.hdu.header, data=ratio).writeto(
            out_path, overwrite=True
        )


def main(
    filter_name: str,
    channel: int,
    data_path: Path = Path("FilterMaps"),
    prefix: str = "nostar",
    all_label: str = "all",
    frac_labels: list[str] = ["cont", "bands", "lines"],
):
    """Calculate fractional continuum/bands/lines contribution to a filter"""

    total = Component(all_label, filter_name, channel, data_path, prefix)
    for label in frac_labels:
        c = Component(label, filter_name, channel, data_path, prefix)
        c.save_ratio(total)


if __name__ == "__main__":
    typer.run(main)
