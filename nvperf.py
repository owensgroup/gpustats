#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import requests
import re
import os
import json
from io import StringIO
from joblib import Parallel, delayed
import altair as alt
from collections import Counter
import vl_convert as vlc
import vegafusion as vf

data = {
    "NVIDIA": {
        "url": "https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units"
    },
    "AMD": {
        "url": "https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units",
    },
    "Intel": {
        "url": "https://en.wikipedia.org/wiki/Intel_Xe",
    },
}

referencesAtEnd = r"(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$"


def merge(df, dst, src, replaceNoWithNaN=False, delete=True):
    df[src] = df[src].replace("\u2014", np.nan)  # em-dash
    if replaceNoWithNaN:
        df[src] = df[src].replace("No", np.nan)
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


for vendor in ["NVIDIA", "AMD", "Intel"]:
    # requests.get handles https
    html = requests.get(data[vendor]["url"]).text
    # oddly, some dates look like:
    # <td><span class="sortkey" style="display:none;speak:none">000000002010-02-25-0000</span><span style="white-space:nowrap">Feb 25, 2010</span></td>
    html = re.sub(r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', "", html)
    html = re.sub(r"<span[^>]*>([^<]+)</span>", r"\1", html)
    # pretty sure I don't need this next one (filter out <a href></a>)
    # html = re.sub(r"<a href=[^>]*>([^<]+)</a>", r"\1", html)
    # someone writes "1234" as "1&nbsp;234", sigh
    html = re.sub(r"(\d)&#160;(\d)", r"\1\2", html)
    html = re.sub(r"&thinsp;", "", html)  # delete thin space (thousands sep)
    html = re.sub(r"&#8201;", "", html)  # delete thin space (thousands sep)
    html = re.sub("\xa0", " ", html)  # non-breaking space -> ' '
    html = re.sub(r"&#160;", " ", html)  # non-breaking space -> ' '
    html = re.sub(r"&nbsp;", " ", html)  # non-breaking space -> ' '
    html = re.sub(r"<br />", " ", html)  # breaking space -> ' '
    html = re.sub("\u2012", "-", html)  # figure dash -> '-'
    html = re.sub("\u2013", "-", html)  # en-dash -> '-'
    html = re.sub("\u2014", "", html)  # delete em-dash (indicates empty cell)
    html = re.sub(r"mm<sup>2</sup>", "mm2", html)  # mm^2 -> mm2
    html = re.sub("\u00d710<sup>6</sup>", "\u00d7106", html)  # 10^6 -> 106
    html = re.sub("\u00d710<sup>9</sup>", "\u00d7109", html)  # 10^9 -> 109
    html = re.sub(r"<sup>\d+</sup>", "", html)  # delete footnotes
    # with open('/tmp/%s.html' % vendor, 'wb') as f:
    #     f.write(html.encode('utf8'))

    dfs = pd.read_html(
        StringIO(html),
        match=re.compile("Launch|Release Date & Price"),
        parse_dates=True,
    )
    for idx, df in enumerate(dfs):
        # Multi-index to index

        # column names that are duplicated should be unduplicated
        # 'Launch Launch' -> 'Launch'
        # TODO do this
        # ' '.join(a for a, b in itertools.zip_longest(my_list, my_list[1:]) if a != b)`
        # print(df.columns.values)
        df.columns = [
            " ".join(
                a
                for a, b in itertools.zip_longest(col, col[1:])
                if (a != b and not a.startswith("Unnamed: "))
            ).strip()
            for col in df.columns.values
        ]
        # df.columns = [' '.join(col).strip() for col in df.columns.values]

        # Intel Xe has "Arc X" rows that get translated into column names, delete
        df.columns = [re.sub(r" Arc [\w]*$", "", col) for col in df.columns.values]
        # this is only to make sure the column names are not ending in #s
        df = df.rename(
            columns={
                "GFLOPS FP32": "GFLOPS FP32_",
                "MFLOPS FP32": "MFLOPS FP32_",
            }
        )

        # TODO this next one disappears
        # Get rid of 'Unnamed' column names
        # df.columns = [re.sub(' Unnamed: [0-9]+_level_[0-9]+', '', col)
        # for col in df.columns.values]
        # If a column-name word ends in a number or number,comma,number, delete
        # it
        df.columns = [
            " ".join([re.sub(r"[\d,]+$", "", word) for word in col.split()])
            for col in df.columns.values
        ]
        # If a column-name word ends with one or more '[x]',
        # where 'x' is an upper- or lower-case letter or number, delete it
        df.columns = [
            " ".join(
                [re.sub(r"(?:\[[a-zA-Z0-9]+\])+$", "", word) for word in col.split()]
            )
            for col in df.columns.values
        ]
        # Get rid of hyphenation in column names
        df.columns = [col.replace("- ", "") for col in df.columns.values]
        # Get rid of space after slash in column names
        df.columns = [col.replace("/ ", "/") for col in df.columns.values]
        # Get rid of trailing space in column names
        df.columns = df.columns.str.strip()

        # These columns are causing JSON problems, just delete them rather than fix it
        # undefined:17
        # "API compliance (version) OpenCL": NaN,
        #                                    ^
        # SyntaxError: Unexpected token N in JSON at position 442
        #   at JSON.parse (<anonymous>)
        #   at /Users/jowens/Documents/working/vega-lite/bin/vl2vg:14:42

        df = df.drop(
            columns=[
                "Supported API version OpenGL",
                "Fillrate Pixel (GP/s)",
                "Fillrate Texture (GT/s)",
                "API support (version) OpenGL",
                "API compliance (version) OpenGL",
            ],
            errors="ignore",
        )

        df["Vendor"] = vendor

        if ("Launch" not in df.columns.values) and (
            "Release Date & Price" in df.columns.values
        ):
            # take everything up to ####
            df["Launch"] = df["Release Date & Price"].str.extract(
                r"^(.*\d\d\d\d)", expand=False
            )
            # leaving this in as a string for now, will parse later
            df["Release Price (USD)"] = df["Release Date & Price"].str.extract(
                r"(\$[\d,]+)", expand=False
            )

        if "Launch" not in df.columns.values:
            print("Launch not in following df:\n", df)
        # make sure Launch is a string (dtype=object) before parsing it
        df["Launch"] = df["Launch"].apply(lambda x: str(x))
        df["Launch"] = df["Launch"].str.replace(referencesAtEnd, "", regex=True)
        # kill everything beyond the year
        df["Launch"] = df["Launch"].str.extract(r"^(.*?[\d]{4})", expand=False)
        df["Launch"] = df["Launch"].apply(lambda x: pd.to_datetime(x, errors="coerce"))
        # if we have duplicate column names, we will see them here
        # pd.concat will fail if so
        if [c for c in Counter(df.columns).items() if c[1] > 1]:
            # so remove them
            # https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
            df = df.loc[:, ~df.columns.duplicated()]
        # this assignment is because I don't have a way to do the above line
        # in-place
        dfs[idx] = df

    data[vendor]["dfs"] = dfs

df = pd.concat(
    data["NVIDIA"]["dfs"] + data["AMD"]["dfs"] + data["Intel"]["dfs"],
    sort=False,
    ignore_index=True,
)

# print all columns
# print(list(df))


# merge related columns
df = merge(
    df, "Processing power (GFLOPS) Single precision", "Processing power (GFLOPS)"
)
df = merge(df, "Processing power (GFLOPS) Single precision", "GFLOPS FP32_")
df = merge(
    df,
    "Processing power (GFLOPS) Single precision",
    "Processing power (GFLOPS) Single precision (MAD or FMA)",
    replaceNoWithNaN=True,
)
df = merge(
    df,
    "Processing power (GFLOPS) Double precision",
    "Processing power (GFLOPS) Double precision (FMA)",
    replaceNoWithNaN=True,
)
df = merge(df, "Memory Bandwidth (GB/s)", "Memory configuration Bandwidth (GB/s)")
df = merge(df, "TDP (Watts)", "TDP (Watts) Max.")
df = merge(df, "TDP (Watts)", "TDP (Watts) Max")
df = merge(df, "TDP (Watts)", "TBP (W)")
df = merge(df, "TDP (Watts)", "TDP (W)")
df = merge(df, "TDP (Watts)", "Combined TDP Max. (W)")
df = merge(df, "TDP (Watts)", "TDP /idle (Watts)")
# get only the number out of TBP
# TODO this doesn't work - these numbers don't appear
df["TBP"] = df["TBP"].str.extract(r"([\d]+)", expand=False)
df = merge(df, "TDP (Watts)", "TBP")
# fix up watts?
# df['TDP (Watts)'] = df['TDP (Watts)'].str.extract(r'<([\d\.]+)', expand=False)
df = merge(df, "Model", "Model (Codename)")
df = merge(df, "Model", "Model (Code name)")
df = merge(df, "Model", "Model (codename)")
df = merge(df, "Model", "Code name (console model)")
# df = merge(df, 'Model', 'Chip (Device)')
# replace when AMD page updated
df = merge(df, "Model", "Model: Mobility Radeon")
df = merge(df, "Core clock (MHz)", "Shaders Base clock (MHz)")
df = merge(df, "Core clock (MHz)", "Shader clock (MHz)")
df = merge(df, "Core clock (MHz)", "Clock rate Base (MHz)")
df = merge(df, "Core clock (MHz)", "Clock rate (MHz)")
df = merge(df, "Core clock (MHz)", "Clock speeds Base core clock (MHz)")
df = merge(df, "Core clock (MHz)", "Core Clock (MHz)")
df = merge(df, "Core clock (MHz)", "Clock rate Core (MHz)")
df = merge(df, "Core clock (MHz)", "Clock speed Core (MHz)")
df = merge(df, "Core clock (MHz)", "Clock speed Average (MHz)")
df = merge(df, "Core clock (MHz)", "Core Clock rate (MHz)")
df = merge(df, "Core clock (MHz)", "Clock rate (MHz) Core (MHz)")
df = merge(df, "Core clock (MHz)", "Clock speed Shader (MHz)")
df = merge(df, "Core clock (MHz)", "Clock speeds  Base core (MHz)")
df = merge(df, "Core clock (MHz)", "Core Clock (MHz) Base")
df = merge(df, "Core config", "Core Config")
df = merge(df, "Transistors Die Size", "Transistors & Die Size")
df = merge(df, "Transistors Die Size", "Transistors & die size")
df = merge(df, "Memory Bus type", "Memory RAM type")
df = merge(df, "Memory Bus type", "Memory Type")
df = merge(df, "Memory Bus type", "Memory configuration DRAM type")
df = merge(df, "Memory Bus width (bit)", "Memory configuration Bus width (bit)")
df = merge(df, "Release Price (USD)", "Release price (USD)")
df = merge(df, "Release Price (USD)", "Release price (USD) MSRP")
df["Release Price (USD)"] = df["Release Price (USD)"].str.extract(
    r"\$?([\d,]+)", expand=False
)

# filter out {Chips, Code name, Core config}: '^[2-9]\u00d7'
df = df[~df["Chips"].str.contains(r"^[2-9]\u00d7", re.UNICODE, na=False)]
df = df[~df["Code name"].str.contains(r"^[2-9]\u00d7", re.UNICODE, na=False)]
df = df[~df["Core config"].str.contains(r"^[2-9]\u00d7", re.UNICODE, na=False)]
# filter out if Model ends in [xX]2
df = df[~df["Model"].str.contains("[xX]2$", na=False)]
# filter out {transistors, die size} that end in x2
df = df[
    ~df["Transistors (million)"].str.contains(r"\u00d7[2-9]$", re.UNICODE, na=False)
]
df = df[~df["Die size (mm2)"].str.contains(r"\u00d7[2-9]$", re.UNICODE, na=False)]

for prec in ["Single", "Double", "Half"]:
    for col in [
        f"Processing power (TFLOPS) {prec} precision",
        f"Processing power (TFLOPS) {prec} precision (base)",
        f"Processing power (TFLOPS) {prec} precision (boost)",
    ]:
        if col in df.columns.values:
            destcol = f"Processing power (GFLOPS) {prec} precision"
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(",", "")  # get rid of commas
            df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)
            df[col] = pd.to_numeric(df[col]) * 1000.0  # change to GFLOPS
            df = merge(df, destcol, col)

for col in ["MFLOPS FP32_"]:
    if col in df.columns.values:
        destcol = f"Processing power (GFLOPS) Single precision"
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(",", "")  # get rid of commas
        df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)
        df[col] = pd.to_numeric(df[col]) / 1000.0  # change to GFLOPS
        df = merge(df, destcol, col)

# merge GFLOPS columns with "Boost" column headers and rename
for prec in ["Single", "Double", "Half"]:  # single before others for '1/16 SP'
    col = "Processing power (GFLOPS) %s precision" % prec
    spcol = "%s-precision GFLOPS" % "Single"
    # if prec != 'Half':
    #   df = merge(
    #   df, col, 'Processing power (GFLOPS) %s precision Base Core (Base Boost) (Max Boost 2.0)' % prec)
    for (
        srccol
    ) in [  # 'Processing power (GFLOPS) %s precision Base Core (Base Boost) (Max Boost 3.0)',
        # 'Processing power (GFLOPS) %s precision R/F.E Base Core Reference (Base Boost) F.E. (Base Boost) R/F.E. (Max Boost 4.0)',
        "Processing power (GFLOPS) %s"
    ]:
        df = merge(df, col, srccol % prec)

    # handle weird references to single-precision column
    if prec != "Single":
        df.loc[df[col] == "1/16 SP", col] = pd.to_numeric(df[spcol]) / 16
        df.loc[df[col] == "2x SP", col] = pd.to_numeric(df[spcol]) * 2
    # pick the first number we see as the actual number
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(",", "")  # get rid of commas
    df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)

    # convert TFLOPS to GFLOPS
    # tomerge = 'Processing power (TFLOPS) %s Prec.' % prec
    # df[col] = df[col].fillna(
    #     pd.to_numeric(df[tomerge].str.split(' ').str[0], errors='coerce') * 1000.0)
    # df.drop(tomerge, axis=1, inplace=True)

    df = df.rename(columns={col: "%s-precision GFLOPS" % prec})

# split out 'transistors die size'
# example: u'292\u00d7106 59 mm2'
for exponent in ["\u00d7106", "\u00d7109", "B"]:
    dftds = df["Transistors Die Size"].str.extract(
        r"^([\d\.]+)%s (\d+) mm2" % exponent, expand=True
    )
    if exponent == "\u00d7106":
        df["Transistors (million)"] = df["Transistors (million)"].fillna(
            pd.to_numeric(dftds[0], errors="coerce")
        )
    if exponent == "\u00d7109" or exponent == "B":
        df["Transistors (billion)"] = df["Transistors (billion)"].fillna(
            pd.to_numeric(dftds[0], errors="coerce")
        )
    df["Die size (mm2)"] = df["Die size (mm2)"].fillna(
        pd.to_numeric(dftds[1], errors="coerce")
    )

# some AMD chips have core/boost in same entry, take first number
# df["Core clock (MHz)"] = df["Core clock (MHz)"].astype(str).str.split(" ").str[0]
df["Core clock (MHz)"] = df["Core clock (MHz)"].str.extract(r"(\d+)")

df["Memory Bus width (bit)"] = (
    df["Memory Bus width (bit)"].astype(str).str.split(" ").str[0]
)
df["Memory Bus width (bit)"] = (
    df["Memory Bus width (bit)"].astype(str).str.split("/").str[0]
)
df["Memory Bus width (bit)"] = (
    df["Memory Bus width (bit)"].astype(str).str.split(",").str[0]
)
# strip out bit width from combined column
df = merge(df, "Memory Bus type & width (bit)", "Memory Bus type & width")
df["bus"] = df["Memory Bus type & width (bit)"].str.extract(r"(\d+)-bit", expand=False)
df["bus"] = df["bus"].fillna(pd.to_numeric(df["bus"], errors="coerce"))
df = merge(df, "Memory Bus width (bit)", "bus", delete=False)
# collate memory bus type and take first word only, removing chud as
# appropriate
df = merge(df, "Memory Bus type", "Memory Bus type & width (bit)", delete=False)
df["Memory Bus type"] = df["Memory Bus type"].str.split(" ").str[0]
df["Memory Bus type"] = df["Memory Bus type"].str.split(",").str[0]
df["Memory Bus type"] = df["Memory Bus type"].str.split("/").str[0]
df["Memory Bus type"] = df["Memory Bus type"].str.split("[").str[0]
df.loc[df["Memory Bus type"] == "EDO", "Memory Bus type"] = "EDO VRAM"


# merge and numerify transistor counts
df["Transistors (billion)"] = df["Transistors (billion)"].fillna(
    pd.to_numeric(df["Transistors (million)"], errors="coerce") / 1000.0
)
df["Transistors (billion)"] = pd.to_numeric(
    df["Transistors (billion)"], errors="coerce"
)

# extract shader (processor) counts
# df = merge(df, 'Core config',
#            'Core config (SM/SMP/Streaming Multiprocessor)',
#            delete=False)
df["Pixel/unified shader count"] = df["Core config"].str.split(":").str[0]
# this converts core configs like "120(24x5)" to "120"
df["Pixel/unified shader count"] = (
    df["Pixel/unified shader count"].str.split("(").str[0]
)
# now convert text to numbers
df["Pixel/unified shader count"] = pd.to_numeric(
    df["Pixel/unified shader count"], downcast="integer", errors="coerce"
)
df = merge(df, "Pixel/unified shader count", "Stream processors")
df = merge(df, "Pixel/unified shader count", "Shaders CUDA cores (total)")
df = merge(df, "Pixel/unified shader count", "Shading units")  # Intel
# note there might be zeroes

df["SM count (extracted)"] = df["Core config"].str.extract(
    r"\((\d+ SM[MX])\)", expand=False
)
df = merge(df, "SM count", "SM count (extracted)")
# AMD has CUs
df["SM count (extracted)"] = df["Core config"].str.extract(r"(\d+) CU", expand=False)
df = merge(df, "SM count", "SM count (extracted)")
df["SM count (extracted)"] = df["Core config"].str.extract(r"\((\d+)\)", expand=False)
df = merge(df, "SM count", "SM count (extracted)")
df = merge(df, "SM count", "SMX count")
df = merge(df, "SM count", "Execution units")  # Intel


# merge in AMD fab stats
df["Architecture (Fab) (extracted)"] = df["Architecture (Fab)"].str.extract(
    r"\((\d+) nm\)", expand=False
)
df = merge(df, "Fab (nm)", "Architecture (Fab) (extracted)")
df["Architecture (Fab) (extracted)"] = df["Architecture & Fab"].str.extract(
    r"(\d+) nm", expand=False
)
df = merge(df, "Fab (nm)", "Architecture (Fab) (extracted)")
for fab in [
    "TSMC",
    "GloFo",
    "Samsung/GloFo",
    "Samsung",
    "SGS",
    "SGS/TSMC",
    "IBM",
    "UMC",
    "TSMC/UMC",
]:
    df["Architecture (Fab) (extracted)"] = df["Architecture & Fab"].str.extract(
        r"%s (\d+)" % fab, expand=False
    )
    df = merge(df, "Fab (nm)", "Architecture (Fab) (extracted)")
    df["Fab (nm)"] = df["Fab (nm)"].str.replace(fab, "")
    df["Fab (nm)"] = df["Fab (nm)"].str.replace("nm$", "", regex=True)

# NVIDIA: just grab number
# for fab in ['TSMC', 'Samsung']:
#     df.loc[df['Fab (nm)'].str.match(
#         '^' + fab), 'Fab (nm)'] = df['Fab (nm)'].str.extract('^' + fab + r'(\d+)', expand=False)
df["Architecture (Fab) (extracted)"] = df["Process"].str.extract(r"(\d+)", expand=False)
df = merge(df, "Fab (nm)", "Architecture (Fab) (extracted)")

# take first number from "release price" after deleting $ and ,
df["Release Price (USD)"] = (
    df["Release Price (USD)"]
    .str.replace(r"[,\$]", "", regex=True)
    .str.split(" ")
    .str[0]
)

# this cleans up columns to make sure they're not mixed float/text
# also was useful to make sure columns can be converted to Arrow without errors
for col in [
    "Memory Bandwidth (GB/s)",
    "TDP (Watts)",
    "Fab (nm)",
    "Release Price (USD)",
    "Core clock (MHz)",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].astype(float)


# compute new columns
df["Arithmetic intensity (FLOP/B)"] = pd.to_numeric(
    df["Single-precision GFLOPS"], errors="coerce"
) / pd.to_numeric(df["Memory Bandwidth (GB/s)"], errors="coerce")
df["Single precision GFLOPS/Watt"] = pd.to_numeric(
    df["Single-precision GFLOPS"], errors="coerce"
) / pd.to_numeric(df["TDP (Watts)"], errors="coerce")
df["Single precision GFLOPS/USD"] = (
    pd.to_numeric(df["Single-precision GFLOPS"], errors="coerce")
    / df["Release Price (USD)"]
)
df["Watts/mm2"] = pd.to_numeric(df["TDP (Watts)"], errors="coerce") / pd.to_numeric(
    df["Die size (mm2)"], errors="coerce"
)
df["Transistor Density (B/mm2)"] = pd.to_numeric(
    df["Transistors (billion)"], errors="coerce"
) / pd.to_numeric(df["Die size (mm2)"], errors="coerce")
df["Single-precision GFLOPS/mm2"] = pd.to_numeric(
    df["Single-precision GFLOPS"], errors="coerce"
) / pd.to_numeric(df["Die size (mm2)"], errors="coerce")

# remove references from end of model/transistor names
for col in ["Model", "Transistors (million)"]:
    df[col] = df[col].str.replace(referencesAtEnd, "", regex=True)
    # then take 'em out of the middle too
    df[col] = df[col].str.replace(r"\[\d+\]", "", regex=True)

# mark mobile processors
df["GPU Type"] = np.where(
    df["Model"].str.contains(r" [\d]+M[X]?|\(Notebook\)"), "Mobile", "Desktop"
)

# we end up with a lot of "nan" that cause problems during JSON serialization
# df = df.fillna(value="")
# df = df.replace("nan", "")

# print(df.columns.values)

# alt.values=c("amd"="#ff0000",
#   "nvidia"="#76b900",
#   "intel"="#0860a8",

colormap = alt.Scale(
    domain=["AMD", "NVIDIA", "Intel"], range=["#ff0000", "#76b900", "#0071c5"]
)

# ahmed:
bw_selection = alt.selection_point(fields=["Memory Bus type"])
bw_color = alt.condition(
    bw_selection, alt.Color("Memory Bus type:N"), alt.value("lightgray")
)
##
bw = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Memory Bandwidth (GB/s):Q",
            scale=alt.Scale(type="log"),
        ),
        # color='Memory Bus type',
        color=bw_color,
        shape="Vendor",
        tooltip=["Model", "Memory Bus type", "Memory Bandwidth (GB/s)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(bw_selection)
)
# ahmed:
# Legend
# bw |= Chart(df).mark_point().encode(
#    y= Y('Memory Bus type:N', axis= Axis(orient='right')),
#    color=bw_color
# ).add_params(
#    bw_selection
# )
##


# ahmed:
bus_selection = alt.selection_point(fields=["Memory Bus type"])
bus_color = alt.condition(
    bus_selection, alt.Color("Memory Bus type:N"), alt.value("lightgray")
)
##
bus = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Memory Bus width (bit):Q",
            scale=alt.Scale(type="log"),
        ),
        # color='Memory Bus type',
        color=bus_color,
        shape="Vendor",
        tooltip=["Model", "Memory Bus type", "Memory Bus width (bit)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(bus_selection)
)

# ahmed:
pr_selection = alt.selection_point(fields=["Datatype"])
pr_color = alt.condition(pr_selection, alt.Color("Datatype:N"), alt.value("lightgray"))
##
pr = (
    alt.Chart(
        pd.melt(
            df,
            id_vars=["Launch", "Model", "GPU Type", "Vendor"],
            value_vars=[
                "Single-precision GFLOPS",
                "Double-precision GFLOPS",
                "Half-precision GFLOPS",
            ],
            var_name="Datatype",
            value_name="Processing power (GFLOPS)",
        )
    )
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Processing power (GFLOPS):Q",
            scale=alt.Scale(type="log"),
        ),
        shape="Vendor",
        # color='Datatype',
        color=pr_color,
        tooltip=["Model", "Datatype", "Processing power (GFLOPS)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(pr_selection)
)


# ahmed:
sm_selection = alt.selection_point(fields=["Vendor"])
sm_color = alt.condition(
    sm_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
sm = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y="SM count:Q",
        # color=alt.Color('Vendor',scale=colormap,),
        color=sm_color,
        tooltip=["Model", "SM count"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(sm_selection)
)


# ahmed:
die_selection = alt.selection_point(fields=["Vendor"])
die_color = alt.condition(
    die_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
die = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Die size (mm2):Q",
            scale=alt.Scale(type="log"),
        ),
        # color=alt.Color('Vendor',scale=colormap,),
        color=die_color,
        shape="GPU Type",
        tooltip=["Model", "Die size (mm2)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(die_selection)
)


# ahmed:
xt_selection = alt.selection_point(fields=["Vendor"])
xt_color = alt.condition(
    xt_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
xt = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Transistors (billion):Q",
            scale=alt.Scale(type="log"),
        ),
        # color=alt.Color('Vendor',scale=colormap,),
        color=xt_color,
        shape="GPU Type",
        tooltip=["Model", "Transistors (billion)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(xt_selection)
)


# ahmed:
fab_selection = alt.selection_point(fields=["Vendor"])
fab_color = alt.condition(
    fab_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
fab = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Fab (nm):Q",
            scale=alt.Scale(type="log"),
        ),
        # color=alt.Color('Vendor',scale=colormap,),
        color=fab_color,
        tooltip=["Model", "Fab (nm)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(fab_selection)
)


# ahmed:
ai_selection = alt.selection_point(fields=["Vendor"])
ai_color = alt.condition(
    ai_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
ai = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y="Arithmetic intensity (FLOP/B):Q",
        shape="GPU Type",
        # color=alt.Color('Vendor',scale=colormap,),
        color=ai_color,
        tooltip=["Model", "Arithmetic intensity (FLOP/B)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(ai_selection)
)


# ahmed:
fpw_selection = alt.selection_point(fields=["Vendor"])
fpw_color = alt.condition(
    fpw_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
fpw = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y="Single precision GFLOPS/Watt:Q",
        shape="GPU Type",
        # color=alt.Color('Vendor', scale=colormap,),
        color=fpw_color,
        tooltip=["Model", "Single precision GFLOPS/Watt"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(fpw_selection)
)


# ahmed:
clk_selection = alt.selection_point(fields=["Vendor"])
clk_color = alt.condition(
    clk_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
clk = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y="Core clock (MHz):Q",
        shape="GPU Type",
        # color=alt.Color('Vendor', scale=colormap,),
        color=clk_color,
        tooltip=["Model", "Core clock (MHz)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(clk_selection)
)


# ahmed:
cost_selection = alt.selection_point(fields=["Vendor"])
cost_color = alt.condition(
    cost_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
cost = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y("Release Price (USD):Q", scale=alt.Scale(type="log")),
        # color=alt.Color('Vendor', scale=colormap,),
        color=cost_color,
        shape="GPU Type",
        tooltip=["Model", "Release Price (USD)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(cost_selection)
)


# ahmed:
fperdollar_selection = alt.selection_point(fields=["Vendor"])
fperdollar_color = alt.condition(
    fperdollar_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
fperdollar = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="Launch:T",
        y="Single precision GFLOPS/USD:Q",
        # color=alt.Color('Vendor',scale=colormap,),
        color=fperdollar_color,
        shape="GPU Type",
        tooltip=["Model", "Single precision GFLOPS/USD"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(fperdollar_selection)
)


# ahmed:
fpwsp_selection = alt.selection_point(fields=["Fab (nm)"])
fpwsp_color = alt.condition(
    fpwsp_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
# only plot chips with actual feature sizes
fpwsp = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x=alt.X(
            "Single-precision GFLOPS:Q",
            scale=alt.Scale(type="log"),
        ),
        y="Single precision GFLOPS/Watt:Q",
        shape="Vendor",
        # color='Fab (nm):N',
        color=fpwsp_color,
        tooltip=["Model", "Fab (nm)", "Single precision GFLOPS/Watt"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(fpwsp_selection)
)


# ahmed:
fpwbw_selection = alt.selection_point(fields=["Fab (nm)"])
fpwbw_color = alt.condition(
    fpwbw_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
fpwbw = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x=alt.X(
            "Memory Bandwidth (GB/s):Q",
            scale=alt.Scale(type="log"),
        ),
        y="Single precision GFLOPS/Watt:Q",
        shape="Vendor",
        # color='Fab (nm):N',
        color=fpwbw_color,
        tooltip=["Model", "Fab (nm)", "Memory Bandwidth (GB/s)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(fpwbw_selection)
)


# ahmed:
aisp_selection = alt.selection_point(fields=["Fab (nm)"])
aisp_color = alt.condition(
    aisp_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
aisp = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x=alt.X(
            "Single-precision GFLOPS:Q",
            scale=alt.Scale(type="log"),
        ),
        y="Arithmetic intensity (FLOP/B):Q",
        shape="Vendor",
        # color='Fab (nm):N',
        color=aisp_color,
        tooltip=[
            "Model",
            "Fab (nm)",
            "Single-precision GFLOPS",
            "Memory Bandwidth (GB/s)",
        ],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(aisp_selection)
)


# ahmed:
aibw_selection = alt.selection_point(fields=["Fab (nm)"])
aibw_color = alt.condition(
    aibw_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
aibw = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x=alt.X(
            "Memory Bandwidth (GB/s):Q",
            scale=alt.Scale(type="log"),
        ),
        y="Arithmetic intensity (FLOP/B):Q",
        shape="Vendor",
        # color='Fab (nm):N',
        color=aibw_color,
        tooltip=[
            "Model",
            "Fab (nm)",
            "Single-precision GFLOPS",
            "Memory Bandwidth (GB/s)",
        ],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(aibw_selection)
)


# ahmed:
sh_selection = alt.selection_point(fields=["Vendor"])
sh_color = alt.condition(
    sh_selection,
    alt.Color(
        "Vendor:N",
        scale=colormap,
    ),
    alt.value("lightgray"),
)
##
# need != 0 because we're taking a log
sh = (
    alt.Chart(df[df["Pixel/unified shader count"] != 0])
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Pixel/unified shader count:Q",
            scale=alt.Scale(type="log"),
        ),
        shape="GPU Type",
        # color=alt.Color('Vendor', scale=colormap,),
        color=sh_color,
        tooltip=["Model", "Pixel/unified shader count"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(sh_selection)
)


# ahmed:
pwr_selection = alt.selection_point(fields=["Fab (nm)"])
pwr_color = alt.condition(
    pwr_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
pwr = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x="Launch:T",
        y="TDP (Watts)",
        shape="Vendor",
        # color='Fab (nm):N',
        color=pwr_color,
        tooltip=["Model", "Fab (nm)", "TDP (Watts)"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(pwr_selection)
)


# ahmed:
pwrdens_selection = alt.selection_point(fields=["Fab (nm)"])
pwrdens_color = alt.condition(
    pwrdens_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
pwrdens = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x="Launch:T",
        y="Watts/mm2",
        shape="Vendor",
        # color='Fab (nm):N',
        color=pwrdens_color,
        tooltip=["Model", "Fab (nm)", "TDP (Watts)", "Die size (mm2)", "Watts/mm2"],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(pwrdens_selection)
)

# ahmed:
transdens_selection = alt.selection_point(fields=["Fab (nm)"])
transdens_color = alt.condition(
    transdens_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
transdens = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x="Launch:T",
        y=alt.Y(
            "Transistor Density (B/mm2)",
            scale=alt.Scale(type="log"),
        ),
        shape="Vendor",
        # color='Fab (nm):N',
        color=transdens_color,
        tooltip=[
            "Model",
            "Fab (nm)",
            "Transistors (billion)",
            "Die size (mm2)",
            "Transistor Density (B/mm2)",
        ],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(transdens_selection)
)

# ahmed:
tdpvsdens_selection = alt.selection_point(fields=["Fab (nm)"])
tdpvsdens_color = alt.condition(
    tdpvsdens_selection, alt.Color("Fab (nm):N"), alt.value("lightgray")
)
##
tdpvsdens = (
    alt.Chart(df[df["Fab (nm)"].notnull()])
    .mark_point()
    .encode(
        x=alt.X(
            "Single precision GFLOPS/Watt:Q",
            scale=alt.Scale(type="log"),
        ),
        y=alt.Y(
            "Single-precision GFLOPS/mm2:Q",
            scale=alt.Scale(type="log"),
        ),
        shape="Vendor",
        # color='Fab (nm):N',
        color=tdpvsdens_color,
        tooltip=[
            "Model",
            "Fab (nm)",
            "Die size (mm2)",
            "Transistor Density (B/mm2)",
            "Single precision GFLOPS/Watt",
            "Single-precision GFLOPS/mm2",
        ],
    )
    .properties(width=1213, height=750)
    .interactive()
    .add_params(tdpvsdens_selection)
)

# df.to_csv("/tmp/gpu.csv", encoding="utf-8")

template = """<!DOCTYPE html>
<html>
<head>
  <!-- Import vega & vega-Lite (does not have to be from CDN) -->
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <!-- Import vega-embed -->
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <title>{title}</title>

  <style media="screen">
    /* Add space between vega-embed links */
    /* http://vega.github.io/vega-tutorials/airports/ */
    .vega-embed .vega-actions a {{
      margin-left: 1em;
      visibility: hidden;
    }}
    .vega-embed:hover .vega-actions a {{
      visibility: visible;
    }}
  </style>
</head>
<body>

<div id="vis"></div>

<script type="text/javascript">
  var opt = {{
    "mode": "vega-lite",
  }};
  vegaEmbed('#vis', {spec}, opt).catch(console.warn);
</script>
</body>
</html>"""

readme = "# GPU Statistics\n\nData sourced from [Wikipedia's NVIDIA GPUs page](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units), [Wikipedia's AMD GPUs page](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units), and [Wikipedia's Intel Xe page](https://en.wikipedia.org/wiki/Intel_Xe).\n\n"

# outputdir = "/Users/jowens/Documents/working/owensgroup/proj/gpustats/plots"
# ahmed: Get absoulte folder path https://stackoverflow.com/a/32973383
script_dir = os.path.dirname(os.path.realpath("__file__"))
rel_path = "plots"
outputdir = os.path.join(script_dir, rel_path)

plots = [
    (bw, "Memory Bandwidth over Time"),
    (bus, "Memory Bus Width over Time"),
    (pr, "Processing Power over Time"),
    (sm, "SM count over Time"),
    (sh, "Shader count over Time"),
    (die, "Die Size over Time"),
    (xt, "Transistor Count over Time"),
    (fab, "Feature size over Time"),
    (clk, "Clock rate over Time"),
    (cost, "Release price over Time"),
    (fperdollar, "GFLOPS per Dollar over Time"),
    (fpw, "GFLOPS per Watt over Time"),
    (fpwsp, "GFLOPS per Watt vs. Peak Processing Power"),
    (fpwbw, "GFLOPS per Watt vs. Memory Bandwidth"),
    (ai, "Arithmetic Intensity over Time"),
    (aisp, "Arithmetic Intensity vs. Peak Processing Power"),
    (aibw, "Arithmetic Intensity vs. Memory Bandwidth"),
    (pwr, "Power over Time"),
    (pwrdens, "Power density over Time"),
    (transdens, "Transistor density over Time"),
    (tdpvsdens, "GFLOPS per Area vs. GFLOPS per Watt"),
]


def saveHTMLAndPDF(chart, title):
    chart.save(os.path.join(outputdir, title + ".pdf"), engine="vl-convert")
    vf.save_html(chart, os.path.join(outputdir, title + ".html"))


# Parallel(n_jobs=4)(delayed(saveHTMLAndPDF)(chart, title) for (chart, title) in plots)

# if we're looking for a particular value, here's how we find it:
# search_val = "1066"
# for col in df.columns:
#     try:
#         if df[col].dtype.kind == "O":
#             idx = df[col].tolist().index(search_val)
#             print(f"val: {search_val}, col: {col}, index: {idx}")
#     except valueError:
#         pass

for chart, title in plots:
    saveHTMLAndPDF(chart, title)
    readme += f"- {title} [[html](plots/{title}.html), [pdf](plots/{title}.pdf)]\n"
with open(os.path.join(outputdir, "../README.md"), "w") as f:
    f.write(readme)
