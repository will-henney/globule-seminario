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

from astropy.table import Table

import pandas as pd

import uncertainties as un
from uncertainties import unumpy

from pathlib import Path

datapath = Path.cwd().parent / "data" / "kinematics"

df = pd.read_csv(datapath / "zavala-clump-velocities.tsv", delimiter="\t")

df.describe()

# ## Convert columns to ufloat to propagate uncertainties

# First we have to make sure all the nan values are strings, otherwise the conversion will not work><

df = df.fillna("nan +/- nan")

df.loc[4, "R_s (const)"]

# Select the columns that have uncertainties values

ucols = [_ for _ in df.columns if _.endswith(')')]
ucols

# Apply the `ufloat_fromstr` function to all these to get uncertainties objects

for col in ucols[:2]:
    df.loc[:, col] = df[col].apply(un.ufloat_fromstr)
df

# We can split up the nominal value and the uncertainty like this

unumpy.nominal_values(df["R_s (hubb)"])

unumpy.std_devs(df["R_s (hubb)"])

# Note that `unumpy.nominal_values` is different from `un.nominal_value` (plural versus singular), which works on scalar values, so needs to be broadcast by hand:

df["R_s (hubb)"].apply(un.nominal_value).head()

# ## Recalculate the derived columns

# The measured columns are the `R_s` ones, while the others are derived in a table in my org file. So, we will check on these by recalculating them here. In theory, `Emacs calc` and `uncertainties` should give the same answer 

df.loc[:, "inc (hubb)"] = unumpy.degrees(unumpy.arccos(df["R_proj"] / df["R_s (hubb)"]))
df.loc[:, "inc (const)"] = unumpy.degrees(unumpy.arccos(df["R_proj"] / df["R_s (const)"]))
df.loc[:, "V_los (hubb)"] = 46 * unumpy.sqrt(df["R_s (hubb)"]**2 - df["R_proj"]**2) / 20
df.loc[:, "V_los (const)"] = 46 * unumpy.sqrt(df["R_s (const)"]**2 - df["R_proj"]**2) / df["R_s (const)"]

# +
blue_mask = df["V"].astype(str).str.startswith("-")

df.loc[blue_mask, "inc (hubb)"] *= -1
df.loc[blue_mask, "inc (const)"] *= -1
# -

df

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_color_codes()
sns.set_context("talk")



fig, ax = plt.subplots()
x = df["R_proj"]
y = unumpy.nominal_values(df["inc (hubb)"])
dy = unumpy.std_devs(df["inc (hubb)"])
y2 = unumpy.nominal_values(df["inc (const)"])
dy2 = unumpy.std_devs(df["inc (const)"])
ax.scatter(x, y)
ax.errorbar(x, y, yerr=dy, fmt="none")
ax.scatter(x, y2, s=10)
ax.errorbar(x, y2, yerr=dy2, fmt="none", alpha=0.2)
ax.axhline(0.0, lw=1)
ax.set(
    xlabel="Projected separation, arcsec",
    ylabel="Inclination angle, deg",
    xlim=[0, None],
    ylim=[-90, 90],
)
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
sns.despine()
fig.savefig("zavala-clump-inclinations.pdf")
...;

# +
fig, ax = plt.subplots()

x = unumpy.nominal_values(df["R_s (hubb)"])
dx = unumpy.std_devs(df["R_s (hubb)"])
y = unumpy.nominal_values(df["inc (hubb)"])
dy = unumpy.std_devs(df["inc (hubb)"])
ax.scatter(x, y)
ax.errorbar(x, y, xerr=dx, yerr=dy, fmt="none")
ax.set(
    ylabel="Inclination",
    xlabel="True separation",
    xlim=[0, 40],
    ylim=[0, 90],
)
...;
# -

fig, ax = plt.subplots()
x = df["R_proj"]
y = unumpy.nominal_values(df["R_s (hubb)"])
dy = unumpy.std_devs(df["R_s (hubb)"])
ax.scatter(x, y)
ax.errorbar(x, y, yerr=dy, fmt="none")
ax.plot([0, 40], [0, 40], "-")
ax.set(
    xlabel="Projected separation",
    ylabel="True separation",
    xlim=[0, 40],
    ylim=[0, 40],
)
ax.set_aspect("equal")
...;

fig, ax = plt.subplots()
x = df["R_proj"]
y = unumpy.nominal_values(df["R_s (const)"])
dy = unumpy.std_devs(df["R_s (const)"])
ax.scatter(x, y)
ax.errorbar(x, y, yerr=dy, fmt="none")
ax.plot([0, 40], [0, 40], "-")
ax.set(
    xlabel="Projected separation",
    ylabel="True separation",
    xlim=[0, 40],
    ylim=[0, 40],
)
ax.set_aspect("equal")
...;


