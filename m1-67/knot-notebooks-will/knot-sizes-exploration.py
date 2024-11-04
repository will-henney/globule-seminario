# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Knot sizes
#
# Comparison between different ways of measuring this. 

# +
import numpy as np
import typer
from astropy.table import Table, QTable, join
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from matplotlib import pyplot as plt
import seaborn as sns
import seaborn.objects as so
import pandas as pd
# -

ORIGIN = SkyCoord.from_name("wr124", cache=True)

from pathlib import Path

datapath = Path.cwd().parent / "data" / "reproject"

# ## Separations between the knot peaks in different tracers

AA_D_table = QTable.read(datapath / "combo-AA-neutral-combo-D-neutral-knot-seps.ecsv")
DD_D_table = QTable.read(datapath / "combo-DD-neutral-combo-D-neutral-knot-seps.ecsv")
EE_D_table = QTable.read(datapath / "combo-E-ionized-combo-D-neutral-knot-seps.ecsv")

AA_D_table.rename_column("dR", "dR(AA-D)")
DD_D_table.rename_column("dR", "dR(DD-D)")
EE_D_table.rename_column("dR", "dR(EE-D)")

dr_table = join(join(AA_D_table, DD_D_table[["label", "dR(DD-D)"]]), EE_D_table[["label", "dR(EE-D)"]])

dr_table.show_in_notebook()

df = dr_table.to_pandas().set_index("label")

df.describe()

# Remove outliers

drmin, drmax = -0.1, 0.3
for col in df.columns:
    if col.startswith("dR"):
        m = (df[col] < drmin) | (df[col] > drmax)
        df.loc[m, col] = np.nan

df.describe()

df.corr()

# Correlations between displacements are OK: 0.4 to 0.6. Best between AA-D and DD-D

sns.pairplot(df)

# ## Sizes from fitting ellipses

wanted = ["label", "Ellipse Sigma X", "Ellipse Sigma Y"]
etables = {
    "AA": QTable.read(datapath / "combo-AA-neutral-knot-ellipses.ecsv")[wanted],
    "DD": QTable.read(datapath / "combo-DD-neutral-knot-ellipses.ecsv")[wanted],
    "D": QTable.read(datapath / "combo-D-neutral-knot-ellipses.ecsv")[wanted],
    "E": QTable.read(datapath / "combo-E-ionized-knot-ellipses.ecsv")[wanted],
    "H": QTable.read(datapath / "wr124-hst-2008-f656n-radec-knot-ellipses.ecsv")[wanted],
}


for key, tab in etables.items():
    tab.rename_columns(wanted[1:3], [f"sigx({key})", f"sigy({key})"])

sig_table = None
for key, tab in etables.items():
    if sig_table is None:
        sig_table = tab
    else:
        sig_table = join(sig_table, tab)

sig_table.show_in_notebook()

sdf = sig_table.to_pandas().set_index("label")

sdf.describe()

sdf.columns

# Discard extreme outliers by imposing max of 0.3 arcsec for the neutral ones and 0.6 for the ionized ones because they are bigger

sigmin, sigmax = 0.0, 0.3
esigmax = 0.6
for col in sdf.columns:
    if col.startswith("sig"):
        if "(E)" in col or "(H)" in col:
            smax = esigmax
        else:
            smax = sigmax
        m = (sdf[col] < sigmin) | (sdf[col] > smax)
        sdf.loc[m, col] = np.nan

sdf.describe()

sdf.corr()

# We have a good correlation between x and y for the same image but we are really interested in correlations between images. So divide into the x and y sigmas to look at separately. 

xcols = [_ for _ in sdf.columns if _.startswith("sigx")]
ycols = [_ for _ in sdf.columns if _.startswith("sigy")]

sdf[xcols].corr()

sdf[ycols].corr()

# For sigx, the best correlation is between E and H (r=0.67), which is good since they are both supposed to trace the ionized gas. 
#
# There is moderate correlation between AA and DD (r=0.38), which both should trace the near PDR. 
#
# The correlation with D is very weak (r~=0.1), probably because D is lower resolution, so sig never drops below 0.1 arcsec. 
#
# Results for sigy are very similar

sns.pairplot(sdf[xcols])

sns.pairplot(sdf[ycols])

# ## Combine the sigmas with the displacements

dff = df.join(sdf[xcols])

dff

# Eliminate negative dR

for col in dff.columns:
    if col.startswith("dR"):
        m = dff[col] <= 0.0
        dff.loc[m, col] = np.nan

dff.corr()

drcols = [_ for _ in dff.columns if _.startswith("dR")]

sns.pairplot(dff, x_vars=drcols, y_vars=xcols)

cdf = dff[drcols + xcols].corr()
sol = (cdf.where(np.triu(np.ones(cdf.shape), k=1).astype(bool))
                  .stack()
                  .sort_values(ascending=False))
sol.head(15)

# So the numbers say that the best dR-sigx correlation 
#
# ```
#  	 	 	dR(AA-D)	dR(DD-D)	dR(EE-D)
#  	 	 	-------- 	-------- 	--------
# sigx(AA)	-0.011335	0.119774	0.183262
# sigx(DD)	-0.067508	0.091240	0.110798
# sigx(D)		0.011790	0.155118	0.118572
# sigx(E) 	-0.071275	-0.153383	0.026624
# sigx(H) 	-0.110820	0.020317	-0.037470	
#
# ```
# is between dR(EE-D) and sigx(AA) with r=0.18 (not great!)
#
# But all the dR(DD-D) and dR(EE-D) correlations with sigx(AA), sigx(DD), sigx(D) are similar (r=0.1 to 0.2)
#
# Apart from that, there are non-existent or negative correlations for dR(AA-D) with all sigx, and for all dR with sigx(E) and (H).
#
# *We clearly need better measurements of the widths*

# ## Use the new sizes from peak photometry

ptables = {
    "AA": QTable.read(datapath / "combo-AA-neutral-knot-peak-stats.ecsv"),
    "DD": QTable.read(datapath / "combo-DD-neutral-knot-peak-stats.ecsv"),
    "D": QTable.read(datapath / "combo-D-neutral-knot-peak-stats.ecsv"),
    "E": QTable.read(datapath / "combo-E-ionized-knot-peak-stats.ecsv"),
    "H": QTable.read(datapath / "wr124-hst-2008-f656n-radec-knot-peak-stats.ecsv"),
}


ptables["H"].show_in_notebook()

# Switch to pandas dataframes to make life easier

peakdfs = {k: v.to_pandas().set_index("label") for k, v in ptables.items()}

# ### Look at the summary statistics of all columns for each tracer

peakdfs["H"].describe()

peakdfs["E"].describe()

# So E and H are very similar, as would be expected, with r = 0.1 to 0.2, but a tail down to low values (probably spurious). The effective radius r_eff seems better behaved than the rms radius r_rms, which has some large outliers. 

peakdfs["D"].describe()

# D is slightly smaller, but not much

peakdfs["DD"].describe()

peakdfs["AA"].describe()

# ### Do sigma clipping of each column

# This is to eliminate outliers in the radius measurements

from astropy.stats import sigma_clip


def clip_dataframe(df: pd.DataFrame, columns: list, sig: float=3.0):
    "Perform in-place sigma clipping on some columns of a dataframe"
    for col in columns:
        clipped = sigma_clip(df[col].values, sigma=sig)
        df.loc[clipped.mask, col] = np.nan


def purge_high_nanfracs(df: pd.DataFrame, columns, max_allowed: float=1.0):
    m = df["NaN fraction"] > max_allowed
    for col in columns:
        df.loc[m, col] = np.nan   


rcols = ["r_eff", "r_rms", "Gauss sigma"]
bcols = ["Bright Peak", "Bright BG", "Bright Gauss", "MAD BG"]
fcols = ["Core flux", "Core flux Gauss", "Total flux Gauss"]

for combo in peakdfs:
    clip_dataframe(peakdfs[combo], rcols, sig=4)
    purge_high_nanfracs(peakdfs[combo], rcols + bcols, max_allowed=0.1)


# Check that it worked.

peakdfs["H"].describe()

# ### Look at correlations or r_eff with r_rms within each combo

peakdfs["E"][rcols + bcols + fcols].corr()

peakdfs["E"].loc[peakdfs["E"]["Peak SNR"] > 10][rcols + bcols + fcols].corr()

sns.set_color_codes()


def rpairplot(combo, limits=[0.07, 0.2], dr=0.03):
    g = sns.pairplot(
        peakdfs[combo], vars=rcols, corner=True, 
        plot_kws=dict(alpha=0.8, hue=peakdfs[combo]["Peak SNR"]),
        diag_kws=dict(color="r"),
    )
    xx = np.array(limits)
    g.axes[1, 0].plot(xx, xx, "--", color="k", lw=1)
    g.axes[1, 0].fill_between(xx, xx - dr, xx + dr, color="k", lw=0, alpha=0.05, zorder=-1)    
    g.axes[2, 0].plot(xx, xx / np.sqrt(2), "--", color="k", lw=1)
    g.axes[2, 0].fill_between(xx, (xx - dr) / np.sqrt(2), (xx + dr) / np.sqrt(2), color="k", lw=0, alpha=0.05, zorder=-1)
    g.axes[2, 1].plot(xx, xx / np.sqrt(2), "--", color="k", lw=1)
    g.axes[2, 1].fill_between(xx, (xx - dr) / np.sqrt(2), (xx + dr) / np.sqrt(2), color="k", lw=0, alpha=0.05, zorder=-1)


rpairplot("E")

sns.pairplot(peakdfs["E"], x_vars=bcols, y_vars=rcols, hue="Peak SNR")

peakdfs["H"][rcols + bcols + fcols].corr()

rpairplot("H")

sns.pairplot(peakdfs["H"], x_vars=bcols, y_vars=rcols, hue="Peak SNR")

sns.pairplot(peakdfs["H"], x_vars=fcols, y_vars=rcols, hue="Peak SNR")

g = sns.pairplot(peakdfs["H"], 
             vars=fcols, 
             corner=True, 
             plot_kws=dict(hue=peakdfs["H"]["Peak SNR"], alpha=0.2),
            )
for idxx in [(1, 0), (2, 0), (2, 1)]:
    g.axes[idxx].plot([0, 50], [0, 50], zorder=-1, ls="dashed", c='k')

peakdfs["D"][rcols + bcols + fcols].corr()

rpairplot("D")

# Weird that correlation is negative with D. This is probably because the psf width is comparable, with fwhm = 0.146 arcsec, so sigma = 0.06. I suspect that the 2d rms is sqrt(2) times bigger, so 0.09, which means that the smaller values of r_eff are spurious. 
#
# Peak of both is about 0.12, which after correcting for psf is:

np.sqrt(0.12**2 - 0.09**2)

# So about 0.1 arcsec. But still need to think how that relates to r_0
#

sns.pairplot(peakdfs["D"], x_vars=bcols, y_vars=rcols, hue="Peak SNR")

peakdfs["AA"][rcols + bcols].corr()

rpairplot("AA", [0.05, 0.15])

sns.pairplot(peakdfs["AA"], x_vars=bcols, y_vars=rcols, hue="Peak SNR")

peakdfs["DD"][rcols + bcols].corr()

rpairplot("DD", [0.05, 0.15])

# All these show a wide dispersion in r_rms when r_eff is small. Why is this?
#
# Actually, for AA and DD it looks like  and r_eff are the best, and show good consistency, especially for the highest s/n. These are both smaller than r_rms (even taking into account the sqrt(2) factor). 
#
#

sns.pairplot(peakdfs["DD"], x_vars=bcols, y_vars=rcols, hue="Peak SNR")


# ### Correlations in r_eff between the combos

def cross_corr(combo1, combo2, cols=rcols + bcols):
    return peakdfs[combo1][cols].corrwith(peakdfs[combo2][cols], axis=0)


cross_corr("H", "E")

cross_corr("E", "D")

# Not much correlation between ionized and neutral

# Better to make a new dataframe with all the r_eff columns

rdf = pd.DataFrame(
    {combo: peakdfs[combo]["r_eff"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)

rdf.describe()

rdf.corr()

g = sns.pairplot(rdf, corner=True)
vmin, vmax = 0, 0.25
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])


# So the best correlations are E-H and AA-DD, as expected. They all show some degree of correlation, generally r > 0.3, except for D, which is uncorrelated with anything. 

# ### Correlations in r_rms between the combos
#
# Just for completeness, we will also do r_rms. 

rrdf = pd.DataFrame(
    {combo: peakdfs[combo]["r_rms"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)

rrdf.describe()

rrdf.corr()

g = sns.pairplot(rrdf, corner=True)
vmin, vmax = 0, 0.25
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])

# This is not so dissimilar, except thar r_rms does not have the tail towards low values that r_eff does. 
#
# The AA-DD correlation is even better than with r_eff, although the E-H correlation is a bit weaker. Again, D has low correlation with everything

grdf = pd.DataFrame(
    {combo: peakdfs[combo]["Gauss sigma"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)

grdf.describe()

grdf.corr()

g = sns.pairplot(grdf, corner=True)
vmin, vmax = 0, 0.25
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])







# ## Combine r_eff with dr

drcols = [_ for _ in dr_table.colnames if _.startswith("dR")]
drdf = dr_table.to_pandas().set_index("label")[drcols]

drdf.describe()

clip_dataframe(drdf, drcols, sig=3)
drdf.describe()

# ### Correlations between r_eff and dr

df = rdf.join(drdf)

df.corr()

# Uh, oh. Those correlations do not look great

combos = list(peakdfs.keys())

g = sns.pairplot(df, x_vars=combos, y_vars=drcols)

ddf = rrdf.join(drdf)

ddf.corr()

g = sns.pairplot(ddf, x_vars=combos, y_vars=drcols)


# Unfortunately, there are no strong correlations here at all

# ## A new way of doing dR

# We can just find the scalar separation between the peaks. 

def sep_arcsec(combo1, combo2, coord_name="Peak Center"):
    return ptables[combo1][coord_name].separation(ptables[combo2][coord_name]).arcsec


# Package these up in a new dataframe, sepdf

pairs = [["AA", "D"], ["DD", "D"], ["E", "D"]]
paircols = [f"{pair[0]}-{pair[1]}" for pair in pairs]
sepdf = pd.DataFrame(
    {col:  sep_arcsec(*pair) for col, pair in zip(paircols, pairs)},
    index=peakdfs["E"].index,
)
sepdf.describe()

# And repeat for the Gaussians

gsepdf = pd.DataFrame(
    {col:  sep_arcsec(*pair, coord_name="Gauss Center") for col, pair in zip(paircols, pairs)},
    index=peakdfs["E"].index,
)
gsepdf.describe()

sepdf.corrwith(gsepdf)

# ### Correlations within these new separations

sepdf.corr()

g = sns.pairplot(sepdf, corner=True)
vmin, vmax = 0, 0.42
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])

# They all show some corrrelation. DD-D versus AA-D is the tightest. Even the worst case of E-D versus AA-D looks like it is a reasonable correlation plus a bunch of outliers.
#
# ~Also note the discretization in the values, which is due to using pixel centers.~ Not aanymore since I swirched to subpixel center.

gsepdf.corr()

g = sns.pairplot(gsepdf, corner=True)
vmin, vmax = 0, 0.42
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])

# The gaussian version shows tighter correlation of AA-D with DD-D, but worso on the others

# ### Per-combo r_eff values versus new separations

df2 = rdf.join(sepdf)

df2.corr()

# Not good - nearly all negative

g = sns.pairplot(df2, x_vars=combos, y_vars=paircols)

# ### Per-combo r_rms values versus new separations

ddf2 = rrdf.join(sepdf)

ddf2.corr()

# At least some of these are positive, but none are strong. 

g = sns.pairplot(ddf2, x_vars=combos, y_vars=paircols)

# Note that we would not necessarily expect for there to be a perfect relation between the dR offsets and the sizes, since it would depend a bit on the inclination angle. The model predictions are a bit complex, but if the knot is well resolved then the displacement should be higher as a fraction of the knot size when the inclination angle is low (close to the plane of sky)

# In case we ever want to extract the separations from the labels without bothering to load a different file, then this is one way to do it:

R_knots = 0.1 * drdf.index.str.slice(start=-3).values.astype(int)
R_knots

# ### Intercomparison of the two offset methods

df3 = drdf.join(sepdf)

df3.corr()

g = sns.pairplot(df3, x_vars=drcols, y_vars=paircols)
vmin, vmax = 0.0, 0.35
for ax in g.axes.flat:
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)

# What to concentrate on is the leading diagonal of subplots, which compares two different measures of the displacement between the same pair of combo images. 
#
# This seems to offer a way to get a quality filter on the displacements. The x axis is the displacement between the flux barycenters along a radius from the star, while the y axis is the scalar separation (irrespective of direction) between the peak intensity pixels.
#
# So any shift between peak and barycenter or any non-radial component to the displacement will produce deviation from the red line. In the latter case, the points will only move up.

# For completeness, repeat for offsets between gaussian centers

gdf3 = drdf.join(gsepdf)

gdf3.corr()

g = sns.pairplot(gdf3, x_vars=drcols, y_vars=paircols)
vmin, vmax = 0.0, 0.35
for ax in g.axes.flat:
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    ax.set(ylim=[None, 0.4]) # trim outliers


# ## Combining the two dr methods

# So we can take all knots were the x-y discrepancy is less than a certain amount, say 0.05

def best_dr(pair, max_diff=0.05):
    """Return average of two dr estimates, but only if they more or less agree"""
    dr1 = df3[pair]
    dr2 = df3[f"dR({pair.replace('E', 'EE')})"]    
    mean_ = (dr1 + dr2) / 2
    diff_ = np.abs(dr1 - dr2)
    return np.where(diff_ <= max_diff, mean_, np.nan)


bsepdf = sepdf.copy()

for col in bsepdf.columns.values:
    bsepdf[col] = best_dr(col, 0.08)

bsepdf.corr()

bsepdf.describe()

g = sns.pairplot(bsepdf, corner=True)
vmin, vmax = 0, 0.29
for idx in zip(*np.tril_indices_from(g.axes, -1)):
    g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
    g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])
g.axes[-1, -1].set(xlim=[vmin, vmax], ylim=[vmin, vmax])

# This is much better.

df4 = rdf.join(bsepdf)
ddf4 = rrdf.join(bsepdf)
gdf4 = grdf.join(bsepdf)

df4.corr()

ddf4.corr()

gdf4.corr()

g = sns.pairplot(df4, x_vars=combos, y_vars=paircols)
vmin, vmax = 0.0, 0.29
for ax in g.axes.flat:
    ax.set(xlim=[vmin, vmax], ylim=[vmin, vmax])
    ax.axvline(0.15, ls="--", color="r", lw=1)
    ax.axhline(0.15, ls="--", color="r", lw=1)

g = sns.pairplot(ddf4, x_vars=combos, y_vars=paircols)
vmin, vmax = 0.0, 0.29
for ax in g.axes.flat:
    ax.set(xlim=[vmin, vmax], ylim=[vmin, vmax])
    ax.axvline(0.15, ls="--", color="r", lw=1)
    ax.axhline(0.15, ls="--", color="r", lw=1)

g = sns.pairplot(gdf4, x_vars=combos, y_vars=paircols)
vmin, vmax = 0.0, 0.29
for ax in g.axes.flat:
    ax.set(xlim=[vmin, vmax], ylim=[vmin, vmax])
    ax.axvline(0.15, ls="--", color="r", lw=1)
    ax.axhline(0.15, ls="--", color="r", lw=1)

# So the rms radii look more convincing for most pairs

# ## Theoretical relation between the different widths

# We can investigate this empirically

from astropy.convolution import Gaussian2DKernel, convolve


# Start with pure gaussian profile 

def gaussian_radii(sigma, aperture_radius=0.25, pixel_scale=0.033):
    """Find the 2d RMS radius and effective radius of gaussian in aperture"""
    # Enough pixels to span the radius from the center
    nr = 1 + int(aperture_radius / pixel_scale)
    # Size of array to fully enclose the aperture, such that [nr, nr] is center
    n = 2 * nr + 1
    # Construct cartesian pixel coordinates wrt center 
    x, y = np.indices((n, n))
    x -= nr
    y -= nr
    # Radius from center of each pixel
    r = np.hypot(x, y)
    # Image is delta function in central pixel
    im = np.zeros((n, n))
    im[nr, nr] = 1.0
    # Convolve with gaussian of rms width sigma
    kernel = Gaussian2DKernel(x_stddev=sigma / pixel_scale)
    im = convolve(im, kernel, boundary="extend")
    # Crop at edge of aperture
    im[r > aperture_radius / pixel_scale] = 0.0
    
    # Calculate RMS weighted radius
    r_rms = np.sqrt(np.average(r ** 2, weights=im)) * pixel_scale
    # Cqlculate effective area in pixels
    A_eff = np.sum(im) / np.max(im)
    # And effective radius
    r_eff = np.sqrt(A_eff / np.pi) * pixel_scale
    return f"{r_eff=} {r_rms=}"


# Parameters for explaining D results (aperture of 0.25)

gaussian_radii(0.095, aperture_radius=0.25)

# AA, DD used smaller aperture ...

gaussian_radii(0.095, aperture_radius=0.2)

# So we see that the reduction in rms radius is about 10%

gaussian_radii(0.095, aperture_radius=0.15)


# Now try a homogeneous sphere

def sphere_radii(rsphere, sigma=0.0, aperture_radius=0.25, pixel_scale=0.033):
    """Find the 2d RMS radius and effective radius of a filled sphere source in aperture"""
    # Enough pixels to span the radius from the center
    nr = 1 + int(aperture_radius / pixel_scale)
    # Size of array to fully enclose the aperture, such that [nr, nr] is center
    n = 2 * nr + 1
    # Construct cartesian pixel coordinates wrt center 
    x, y = np.indices((n, n))
    x -= nr
    y -= nr
    # Radius from center of each pixel
    r = np.hypot(x, y)
    # Image is filled homogeneous emission sphere
    im = np.where(
        r <= (rsphere / pixel_scale),
        np.sqrt((rsphere / pixel_scale) ** 2 - r ** 2),
        0.0,
    )
    if sigma > 0.0:
        # Convolve with gaussian of rms width sigma
        kernel = Gaussian2DKernel(x_stddev=sigma / pixel_scale)
        im = convolve(im, kernel, boundary="extend")
    # Crop at edge of aperture
    im[r > aperture_radius / pixel_scale] = 0.0
        
    # Calculate RMS weighted radius
    r_rms = np.sqrt(np.average(r ** 2, weights=im)) * pixel_scale
    print(np.pi * (aperture_radius / pixel_scale)**2)
    # Calculate effective area in pixels
    A_eff = np.sum(im) / np.max(im)
    npix = np.sum(im > 0)
    # And effective radius
    r_eff = np.sqrt(A_eff / np.pi) * pixel_scale
    return f"{A_eff=} {r_eff=} {r_rms=}  {npix=}"


# First we try a very large sphere in a unit radius aperture

sphere_radii(1e10, aperture_radius=1.0)

# This gives r_eff = 1 as expected, and it seems that r_rms is sqrt(2)/2
#
# Now try a smaller sphere

sphere_radii(1, sigma=0.0, aperture_radius=2) 





# ## Summary so far
#
# - The 2d rms radius works best for most combos. After filtering out some outliers this gives
#   - D: 0.16 +/- 0.02
#   - E: 0.15 +/- 0.03
#   - H: 0.14 +/- 0.02
#   - AA: 0.11 +/- 0.02
#   - DD: 0.12 +/- 0.02
# - The relation between 2d rms radius and sigma is r_rms = sqrt(2) sigma
# - These need to corrected for the psf broadening, which is characterised by fwhm
#   - D, E: 0.146 -> 0.09 (2d rms)
#   - AA, DD: 0.07 -> 0.04 (2d rms)
#   - H is compicated: 0.07 but then undersampled in pixels of 0.10 => 0.12 -> 0.07
#   - But we need to conform the rms widths to fwhm: larger by 2 sqrt(ln(2)) = 1.67
# - Resultant psf-subtracted 2D RMS widths
#   - D: 0.13
#   - E: 0.12
#   - H: 0.12
#   - AA: 0.10
#   - DD: 0.11
# - These are all remarkably similar, so we could use an average of 0.12 on the 2D rms scale
# - It remains to be seen how this relates to the radius r0 of the ionization front
#   - [ ] We can do models for this. The easiest would be for a filled homogeneous emission sphere, but we could also do the photoevaporation models. And take into acount the psf broadening
#   - Looks like we get r_rms = 0.63 r0 for the filled sphere case so r0 = 1.6 r_rms
#   - So, approximately we have r0 = FWHM, so it would be better to put all widths on the fwhm scale from the start.
#   - **So we find r0 = 0.22 +/- 0.03**
# - Then the offsets are found to have a much broader distribution
#   - dr = 0.10 +/- 0.05
#   - dr/r0 ranges roughly between 0 and 1, just as we would expect
#   - Hopefully it can be used as a diagnostic of the inclination
#
#
#



ddf4[combos].describe()

ddf4[paircols].describe()

# Conversion from 2D RMS to 1D FWHM

fac = 2*np.sqrt(np.log(2))
fac

# ### Combine results from different combos to get more definitive answer

# Create a final dataframe with median of FWHM psf-coreected widths and the median offsets

r0_label = "knot radius, $r_0$, arcsec"
dr_label = "ionized–neutral offset, $dr$, arcsec"
fpdf = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    # np.sqrt((fac * ddf4["AA"]) ** 2 - 0.073**2),
                    # np.sqrt((fac * ddf4["DD"]) ** 2 - 0.073**2),
                    np.sqrt((fac * ddf4["D"]) ** 2 - 0.146**2),
                    np.sqrt((fac * ddf4["E"]) ** 2 - 0.146**2),
                    np.sqrt((fac * ddf4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [ddf4["AA-D"], ddf4["DD-D"], ddf4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fpdf.describe()

sns.set_context("talk")

g = sns.jointplot(fpdf, x=r0_label, y=dr_label, xlim=[0, 0.3], ylim=[0, 0.3])
g.ax_joint.axvline(fpdf[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fpdf[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("RMS radius method: D,E,H")
g.figure.tight_layout()
...;

g.savefig("mean-r0-dr-distribution.pdf")

# Repeat for the gaussians

fpdf2 = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    np.sqrt((fac * ddf4["AA"]) ** 2 - 0.073**2),
                    np.sqrt((fac * ddf4["DD"]) ** 2 - 0.073**2),
                    # np.sqrt((fac * ddf4["D"]) ** 2 - 0.146**2),
                    # np.sqrt((fac * ddf4["E"]) ** 2 - 0.146**2),
                    # np.sqrt((fac * ddf4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [ddf4["AA-D"], ddf4["DD-D"], ddf4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fpdf2.describe()

g = sns.jointplot(fpdf2, x=r0_label, y=dr_label, xlim=[0, 0.3], ylim=[0, 0.3])
g.ax_joint.axvline(fpdf2[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fpdf2[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("RMS radius method: AA,DD")
g.figure.tight_layout()
...;



gfac = np.sqrt(2) * fac
gfac

fgpdf = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    np.sqrt((gfac * gdf4["AA"]) ** 2 - 0.073**2),
                    np.sqrt((gfac * gdf4["DD"]) ** 2 - 0.073**2),
#                    np.sqrt((gfac * gdf4["D"]) ** 2 - 0.146**2),
#                    np.sqrt((gfac * gdf4["E"]) ** 2 - 0.146**2),
#                    np.sqrt((gfac * gdf4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [gdf4["AA-D"], gdf4["DD-D"], gdf4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fgpdf2 = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    # np.sqrt((gfac * gdf4["AA"]) ** 2 - 0.073**2),
                    # np.sqrt((gfac * gdf4["DD"]) ** 2 - 0.073**2),
                   np.sqrt((gfac * gdf4["D"]) ** 2 - 0.146**2),
                   np.sqrt((gfac * gdf4["E"]) ** 2 - 0.146**2),
                   np.sqrt((gfac * gdf4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [gdf4["AA-D"], gdf4["DD-D"], gdf4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fgpdf.describe()

fgpdf2.describe()

g = sns.jointplot(fgpdf, x=r0_label, y=dr_label, xlim=[0, 0.5], ylim=[0, 0.5],
                  joint_kws=dict(hue=peakdfs["DD"]["Peak SNR"]),
                  marginal_kws=dict(color="pink")
                 )                  
g.ax_joint.axvline(fgpdf[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fgpdf[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("Gausian method: AA,DD")
g.figure.tight_layout()

g = sns.jointplot(fgpdf2, x=r0_label, y=dr_label, xlim=[0, 0.5], ylim=[0, 0.5], 
                  joint_kws=dict(hue=peakdfs["E"]["Peak SNR"]),
                  marginal_kws=dict(color="pink")
                 )
g.ax_joint.axvline(fgpdf2[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fgpdf2[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("Gaussian method: D,E,H")
g.figure.tight_layout()

# And effective radius

fepdf = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    # np.sqrt((fac * df4["AA"]) ** 2 - 0.073**2),
                    # np.sqrt((fac * df4["DD"]) ** 2 - 0.073**2),
                    np.sqrt((fac * df4["D"]) ** 2 - 0.146**2),
                    np.sqrt((fac * df4["E"]) ** 2 - 0.146**2),
                    np.sqrt((fac * df4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [df4["AA-D"], df4["DD-D"], df4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fepdf2 = pd.DataFrame(
    {
        r0_label: np.nanmean(
            np.stack(
                [
                    np.sqrt((fac * df4["AA"]) ** 2 - 0.073**2),
                    np.sqrt((fac * df4["DD"]) ** 2 - 0.073**2),
#                    np.sqrt((fac * df4["D"]) ** 2 - 0.146**2),
#                    np.sqrt((fac * df4["E"]) ** 2 - 0.146**2),
#                    np.sqrt((fac * df4["H"]) ** 2 - 0.068**2 - 0.1**2),
                ], axis=0,
            ), axis=0,
        ),
        dr_label: np.nanmean(
            np.stack(
                [df4["AA-D"], df4["DD-D"], df4["E-D"]], axis=0,
            ), axis=0,
        ),
    },
    index=peakdfs["E"].index,
)

fepdf.describe()

fepdf2.describe()

g = sns.jointplot(fepdf, x=r0_label, y=dr_label, xlim=[0, 0.3], ylim=[0, 0.3])
g.ax_joint.axvline(fepdf[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fepdf[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("Effective radius method: D,E,H")
g.figure.tight_layout()

g = sns.jointplot(fepdf2, x=r0_label, y=dr_label, xlim=[0, 0.3], ylim=[0, 0.3])
g.ax_joint.axvline(fepdf2[r0_label].mean(), color="r", zorder=-1)
g.ax_joint.axhline(fepdf2[r0_label].mean(), color="r", zorder=-1)
g.figure.suptitle("Effective radius method: AA,DD")
g.figure.tight_layout()

# So this one gives the smallest radii and the largest relative dispersion. 

# # Knot fluxes

# First eliminate outliers

for combo in peakdfs:
    clip_dataframe(peakdfs[combo], fcols, sig=3)

fluxdf = pd.DataFrame(
    {combo: peakdfs[combo]["Core flux"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)
fluxdf.describe()

fluxdf.corr()

g = sns.pairplot(fluxdf, corner=True, plot_kws=dict(alpha=0.25))
# vmin, vmax = 0, 0.25
# for idx in zip(*np.tril_indices_from(g.axes, -1)):
#     g.axes[idx].plot([vmin, vmax], [vmin, vmax], "--", color="r", lw=1)
#     g.axes[idx].set(xlim=[vmin, vmax], ylim=[vmin, vmax])


# Look at the the span of the distributions: take ratio between percentiles 95 and 5

np.nanpercentile(fluxdf['H'], [25, 75])


def distro_span(data, frac=0.9):
    percentiles = [50 * (1 - frac), 50 * (1 + frac)]
    a, b = np.nanpercentile(data, percentiles)
    return b / a



distro_span(fluxdf['H']), distro_span(rrdf['H']), distro_span(rdf['H'])

20**(1/3), 20**(1/2)

# So factor of 20 in flux, but only 1.5 in radius. We would need 20**(1/3) = 2.7 to explain the flux variation. Alternatively, factor 4.5 in separation. 
#
#

distro_span(fluxdf['D']), distro_span(rrdf['D']), distro_span(rdf['D'])

distro_span(fluxdf['AA']), distro_span(rrdf['AA']), distro_span(rdf['AA'])

# ## Correlations of flux with knot radius

# Effective radius shows good correlation, but I think this is partly due to the operational definition of the effective radius as sqrt(flux / peak brightness), which means that observational errors on the two parameters are positively correlated.

fluxdf.corrwith(rdf)

# RMS radius shows no correlation with flux. Is this because RMS radius is a poor measure of radius, or is it because there really is no correlation.

fluxdf.corrwith(rrdf)

fluxdf.loc[peakdfs["E"]["Peak SNR"] > 8].corrwith(rrdf)

fluxdf.loc[peakdfs["E"]["Peak SNR"] > 8].describe()

# Very weak correlation for the gaussian-derived radius, but at least it is always positive

fluxdf.corrwith(grdf)

fluxdf.columns.values

jdf = fluxdf.join(rrdf, lsuffix="(Flux)", rsuffix="(r_rms)")
g = sns.pairplot(
    jdf,
    x_vars=[_ for _ in jdf.columns.values if "rms" in _],
    y_vars=[_ for _ in jdf.columns.values if "Flux" in _],
    plot_kws=dict(alpha=0.7, hue=peakdfs["E"]["Peak SNR"]),
    )

jdf = fluxdf.join(rdf, lsuffix="(Flux)", rsuffix="(r_eff)")
g = sns.pairplot(
    jdf,
    x_vars=[_ for _ in jdf.columns.values if "r_eff" in _],
    y_vars=[_ for _ in jdf.columns.values if "Flux" in _],
    plot_kws=dict(alpha=0.7, hue=peakdfs["E"]["Peak SNR"]),
    )

gfluxdf = pd.DataFrame(
    {combo: peakdfs[combo]["Total flux Gauss"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)

gcfluxdf = pd.DataFrame(
    {combo: peakdfs[combo]["Core flux Gauss"].values for combo in peakdfs},
    index=peakdfs["E"].index,
)

jdf = gfluxdf.join(rdf, lsuffix="(Flux)", rsuffix="(r_eff)")
g = sns.pairplot(
    jdf,
    x_vars=[_ for _ in jdf.columns.values if "r_eff" in _],
    y_vars=[_ for _ in jdf.columns.values if "Flux" in _],
    plot_kws=dict(alpha=0.7, hue=peakdfs["E"]["Peak SNR"]),
    )

jdf = gfluxdf.join(grdf, lsuffix="(Tot Gauss Flux)", rsuffix="(r_gauss)")
g = sns.pairplot(
    jdf,
    x_vars=[_ for _ in jdf.columns.values if "r_gauss" in _],
    y_vars=[_ for _ in jdf.columns.values if "Flux" in _],
    plot_kws=dict(alpha=0.7, hue=peakdfs["E"]["Peak SNR"]),
    )
g.figure.suptitle("Flux–radius correlations: Gaussian method (includes flux outside aperture)")
g.figure.tight_layout()

gfluxdf.corrwith(grdf)

gcfluxdf.corrwith(grdf)

jdf = gcfluxdf.join(grdf, lsuffix="(Gauss Flux)", rsuffix="(r_gauss)")
g = sns.pairplot(
    jdf,
    x_vars=[_ for _ in jdf.columns.values if "r_gauss" in _],
    y_vars=[_ for _ in jdf.columns.values if "Flux" in _],
    plot_kws=dict(alpha=0.25, hue=peakdfs["E"]["Peak SNR"]),
    )
g.figure.suptitle("Flux–radius correlations: Gaussian method")
g.figure.tight_layout()


# ## Comparing parameter ranges

# Look at the ratio of values between the 75th/25th quartiles for all columns and all combos

def df_spans(dfdict, frac=0.9, min_snr_percentile=0):
    d = {}
    for combo in dfdict:
        if min_snr_percentile > 0:
            min_snr = np.percentile(dfdict[combo]["Peak SNR"], min_snr_percentile)
        else:
            min_snr = 0.0
        mask = dfdict[combo]["Peak SNR"] >= min_snr
        d[combo] = {
            "Number of knots": np.sum(mask),
            **{
                col: distro_span(dfdict[combo][col].loc[mask], frac=frac) 
                for col in dfdict[combo].columns
            }
        }
    return pd.DataFrame(d)



df_spans(peakdfs, 0.5)

# And the 95th/5th

df_spans(peakdfs, 0.9)

# So now things are not looking so bad, so long as we take the Gaussian values!
#
# We get a ratio of about 10 for flux and about 2 for radius (for H, for instance).
#
# This means that the radius variation can explain a knot Ha flux variation of 2**2 = 4 at constant separation (and hence constant ionizing flux). So the other factors need to explain a factor of 2.5 in flux, which is just about feasible. 
#
# The other factors might be:
# * Range of true radii. Would have to be a factor of sqrt(2.5) = 1.6 if this were the only factor. Remember this is the roughly (mean + 2 sigma) / (mean - 2 sigma), so that implies (sigma/mean) = 0.14
# * Shadowing by another knot in front
# * Viewing angle-dependent extinction (only important for side-on or tail-on angles, which should be rare). 
#
#
# 1 + 2 x = A - 2 A x => x = (A - 1) / (2 A + 1)

df_spans(peakdfs, 0.9, 75)

# So we get very comparable results when we restrict ourselves to the 25% of knots with the best s/n, which suggests that most of the variation is real. 

# However, the implication of this is that the radius variations are a at least half of the variations in flux. Actually, maybe I should be comparing variances, in which case the situation is much worse!
