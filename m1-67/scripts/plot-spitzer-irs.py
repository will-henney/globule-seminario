import numpy as np
import yaml
from astropy.table import Table
import astropy.units as u
from matplotlib import pyplot as plt
from filter_throughput import get_filter_throughput

# Split the wavelength range in three vertically stacked plots
wavranges = [[5.0, 10.0], [10.0, 15.0], [15.5, 20.5]]

fig, axes = plt.subplots(3, 1, figsize=(7, 10))

for ax, [wavmin, wavmax] in zip(axes, wavranges):
    # Plot all the Spitzer spectra
    for spec in "SH", "SL1", "SL2", "LH", "LL1", "LL2":
        tab = Table.read(f"wr124_{spec}_complete.tbl", format="ascii.ipac")
        w = tab["WAVELENGTH"]
        # (F_nu / wave) is proportional to SED
        f = tab["FLUX"] / w
        if "H" in spec:
            # The Hi-res spectra are higher brightness because smaller aperture, so reduce by 10
            f /= 10
            lw = 0.4
        else:
            lw = 1.0
        ax.plot(w, f, lw=lw, drawstyle="steps-mid")

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
    line_list = yaml.safe_load(open("spitzer-lines.yaml"))

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
    )
    ax.set_xticks(range(int(np.ceil(wavmin)), 1 + int(np.floor(wavmax))))
    ax.grid(axis="x", which="major", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(axis="x", which="minor", linewidth=0.1)

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
axes[0].set_title("M1-67: Spitzer IRS spectra compared with JWST MIRI filters\n")

figfile = "irs-spec.pdf"
fig.savefig(figfile, bbox_inches="tight")
print(figfile, end="")
