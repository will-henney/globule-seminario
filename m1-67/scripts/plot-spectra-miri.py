import numpy as np
import yaml
from astropy.table import Table, QTable
import astropy.table.serialize as ats
import astropy.units as u
from matplotlib import pyplot as plt
from filter_throughput import get_filter_throughput
from pathlib import Path
import seaborn as sns

# Split the wavelength range into vertically stacked plots
wavranges = [[5, 10.5], [10.5, 16], [16, 21.5], [21.5, 27.0]]
npanels = len(wavranges)

# Read in the MIRI spectra
# Astropy 5.4 is missing SpectralCoord from the list of allowed mixin classes
ats.__construct_mixin_classes += (
    "astropy.coordinates.spectral_coordinate.SpectralCoord",
)
spec_tabs = [Table.read(f"wr124-miri-spec-ch{ichan}.ecsv") for ichan in (1, 2, 3, 4)]

# Read in list of line IDs
line_list = yaml.safe_load(open("jwst-lines.yaml"))

sns.set_color_codes("bright")
fig, axes = plt.subplots(npanels, 1, figsize=(7, 1 + 3 * npanels))

for ax, [wavmin, wavmax] in zip(axes, wavranges):
    # Plot all the MIRI spectra
    for tab in spec_tabs:
        w = tab["Wavelength"]
        # Only plot spectra that fall in the visible window
        if w.max() > wavmin and w.min() < wavmax:
            apertures = [_ for _ in tab.colnames if not _ == "Wavelength"]
            colors = sns.color_palette("magma", n_colors=len(apertures))
            for aperture, color in zip(apertures, colors):
                # (F_nu / wave) is proportional to SED
                f = tab[aperture] / w
                if "star" in aperture.lower():
                    f /= 1000
                    lw = 0.3
                else:
                    f /= 40
                    lw = 0.1
                alpha = 0.5 if "bg" in aperture.lower() else 1.0
                ax.plot(
                    w,
                    f,
                    color=color,
                    alpha=alpha,
                    lw=lw,
                    drawstyle="steps-mid",
                    label=aperture,
                )

    # Now plot the filter throughputs for JWST MIRI
    wave = np.linspace(5, 25, 500) * u.micron
    for fname, color in zip(
        ["f770w", "f1130w", "f1280w", "f1800w"],
        "cgrm",
    ):
        eff = get_filter_throughput(wave, fname)
        wmean = np.average(wave.value, weights=eff)
        if wmean > wavmin and wmean < wavmax:
            ax.fill_between(wave.value, 4 * eff, alpha=0.2, color=color)
            ax.text(wmean, 0.1, fname, ha="center")

    # And finally the emission line IDs
    for linedata in line_list:
        ion_stage = linedata["label"].split()[-1].strip("[]")
        alpha = 0.5
        linestyle = "dashed"
        if linedata["wave"] < wavmin or linedata["wave"] > wavmax:
            # Outside wavelength window
            continue
        if "elow" in linedata and linedata["elow"] > 1e4:
            # Highly excited configuration
            continue
        elif ion_stage in ["V", "VI"] or linedata["label"] == "He II":
            # Too Highly ionized
            continue
        elif ion_stage in ["IV"]:
            linewidth = 0.3
            y0 = 1.6
            fontsize = "xx-small"
        elif ion_stage in ["III"]:
            linewidth = 0.5
            y0 = 1.6
            fontsize = "xx-small"
        elif linedata["label"].startswith("PAH"):
            # PAH band
            linewidth = 10.0
            y0 = 1.6
            fontsize = "small"
            alpha = 0.1
            linestyle = "-"
        elif linedata["label"].startswith("H"):
            # Hydrogen, Helium, or H_2 line
            linewidth = 1.5
            y0 = 1.8 if linedata["label"].startswith("He") else 1.3
            fontsize = "xx-small"
        else:
            # Ground configuration
            linewidth = 1.0
            y0 = 1.6
            fontsize = "xx-small"

        ax.axvline(
            linedata["wave"],
            linestyle=linestyle,
            color="r",
            linewidth=linewidth,
            alpha=alpha,
            ymin=0.6,
        )
        ax.text(
            linedata["wave"],
            y0,
            f"{linedata['label']} {linedata['wave']:.3f}",
            ha="center",
            va="center",
            rotation="vertical",
            fontsize=fontsize,
        )

    ax.set(
        xlim=[wavmin, wavmax],
        ylim=[0, 2.0],
        # yscale="log",
    )
    ax.set_xticks(range(int(np.ceil(wavmin)), 1 + int(np.floor(wavmax))))
    ax.grid(axis="x", which="major", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(axis="x", which="minor", linewidth=0.1)

axes[-1].legend(ncol=2, fontsize="small")
axes[1].set(
    ylabel=(
        r"Brightness SED, $\nu F_\nu = \lambda F_\lambda$,"
        "\n"
        r"or Filter throughput, $T_\lambda$,"
        "\n"
        "(Arbitrary units)"
    ),
)
axes[-1].set(xlabel="Wavelength, micron")
axes[0].set_title("M1-67: JWST MIRI MRS spectra compared with JWST MIRI filters\n")

figfile = "jwst-miri-mrs-spec.pdf"
fig.savefig(figfile, bbox_inches="tight")
print(figfile, end="")
