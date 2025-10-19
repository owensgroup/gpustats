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
import altair_transform  # https://stackoverflow.com/questions/67808483/get-altair-regression-parameters (installed 0.3.0dev0)
from collections import Counter
import vl_convert as vlc
import vegafusion as vf

pd.set_option("display.max_columns", None)

data = {
    "NVIDIA": {
        "urls": [
            "https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units"
        ]
    },
    "AMD": {
        "urls": [
            "https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units",
        ]
    },
    "Intel": {
        "urls": [
            "https://en.wikipedia.org/wiki/Intel_Xe",
            "https://en.wikipedia.org/wiki/Intel_Arc",
        ]
    },
}

# this matches a number at the end
referencesAtEnd = r"(?:\s*\[[a-z\d]+\])+(?:\d+,)?(?:\d+)?$"
# this does not
referencesOnlyAtEnd = r"(?:\s*\[[a-z\d]+\])+$"


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
    html = ""
    for url in data[vendor]["urls"]:
        print(url)
        html = (
            html
            + requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
                },
            ).text
        )
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
    html = re.sub(r"<sup>[\d\*]+</sup>", "", html)  # delete footnotes (num or *)
    # with open("/tmp/%s.html" % vendor, "wb") as f:
    #     f.write(html.encode("utf8"))

    dfs = pd.read_html(
        StringIO(html),
        match=re.compile("Launch|Release Date & Price|Release date"),
        parse_dates=True,
    )
    # purge tables with <= 2 columns, because they're not real/helpful
    dfs = [df for df in dfs if len(df.columns.values) > 2]
    dfs = [
        df for df in dfs if df.columns[0] != "GeForce RTX"
    ]  ### XXX fix this is a weird transposed table
    #### dfs = [df.transpose(copy=True) if not df.empty and df.columns[0] == 'GeForce RTX' else df for df in dfs]

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

        # Combine neighboring columns that are named X and X.1
        deleteCombine = set()
        for left, right in zip(df.columns, df.columns[1:]):
            if (f"{left}.1" == right) or (f"{left} {left}.1" == right):
                df[left] = df[left] + " " + df[right]
                deleteCombine.add(right)
        for col in deleteCombine:
            df.drop(col, axis=1, inplace=True)

        # Intel Xe has "Arc X" rows that get translated into column names, delete
        df.columns = [re.sub(r" Arc [\w]*$", "", col) for col in df.columns.values]
        # TODO this next one disappears
        # Get rid of 'Unnamed' column names
        # df.columns = [re.sub(' Unnamed: [0-9]+_level_[0-9]+', '', col)
        # for col in df.columns.values]
        # If a column-name word ends in a number or number,comma,number, delete
        # it
        df = df.rename(columns={"L2 Cache (MiB)": "L2_ Cache (MiB)"})
        df.columns = [
            " ".join([re.sub(r"[\d,]+$", "", word) for word in col.split()])
            for col in df.columns.values
        ]
        # now put back the ones we broke
        df = df.rename(columns={"L2_ Cache (MiB)": "L2 Cache (MiB)"})
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

        # ignore errors
        # syntax: from: to
        df = df.rename(
            columns={
                "Release date & price": "Release Date & Price",
                "Release date": "Launch",
                "Architecture & fab": "Architecture & Fab",
                "Architecture Fab": "Architecture & Fab",
                "Architecture (Fab)": "Architecture & Fab",
                "Processing power (TFLOPS) Bfloat16": "Processing power (TFLOPS) Half precision",
            }
        )

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
        # handle Q3 2023
        df["Launch"] = df["Launch"].str.replace(r"(Q\d) (\d+)", r"\2-\1", regex=True)
        df["Launch"] = df["Launch"].apply(lambda x: pd.to_datetime(x, errors="coerce"))
        # knock out any invalid launch dates
        df = df[df["Launch"].notna()]

        # if we have duplicate column names, we will see them here
        # pd.concat will fail if so
        if [c for c in Counter(df.columns).items() if c[1] > 1]:
            # so remove them
            # https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
            df = df.loc[:, ~df.columns.duplicated()]
        # this assignment is because I don't have a way to do the above line
        # in-place
        dfs[idx] = df

        # df.to_csv(f"/tmp/{vendor}-{idx}.csv", encoding="utf-8")

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
df = merge(
    df, "Processing power (GFLOPS) Single precision", "Performance (GFLOPS FP32)"
)
df = merge(df, "Memory Bandwidth (GB/s)", "Memory configuration Bandwidth (GB/s)")
df = merge(df, "TDP (Watts)", "TDP (Watts) Max.")
df = merge(df, "TDP (Watts)", "TDP (W)")
df = merge(df, "TDP (Watts)", "Combined TDP Max. (W)")
df = merge(df, "TDP (Watts)", "TDP /idle (Watts)")
df = df.copy()  # this unfragments the df, better performance
# get only the number out of {TBP,TDP}
for col in ["TBP", "TDP"]:
    df[f"{col} (extracted)"] = pd.to_numeric(
        df[col].str.extract(r"(\d+)", expand=False)
    )
    df = merge(df, "TDP (Watts)", f"{col} (extracted)")

df = merge(df, "Model", "Model (Codename)")
df = merge(df, "Model", "Model (Code name)")
df = merge(df, "Model", "Model name")
df = merge(df, "Model", "Code name (console model)")
df = merge(df, "Model", "Branding and Model")
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
df = merge(df, "Core clock (MHz)", "Core Clock (MHz) Base")
df = merge(df, "Core config", "Core Config")
df = merge(df, "Transistors Die Size", "Transistors & die size")
df = merge(df, "Memory Size (MB)", "Memory Size (MiB)")
df = merge(df, "Memory Size (GB)", "Memory Size (GiB)")
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
for col in ["Code name", "Core config"]:
    df = df[~df[col].str.contains(r"^[2-9]\u00d7", re.UNICODE, na=False)]
# filter out if Model ends in [xX]2 or is 2<times>
df = df[~df["Model"].str.contains("[xX]2$", na=False)]
df = df[~df["Model"].str.contains(r"^2\u00d7$", na=False)]
# filter out fields that end in <times>2 or start with 2x
for col in ["Transistors (million)", "Die size (mm2)", "Core config"]:
    df = df[~df[col].str.contains(r"\u00d7[2-9]$", re.UNICODE, na=False)]
    df = df[~df[col].str.contains(r"^2x", re.UNICODE, na=False)]

for prec in ["Single", "Double", "Half"]:
    for col in [
        f"Processing power TFLOPS {prec}",
        f"Processing power (TFLOPS) {prec}",
        f"Processing power (TFLOPS) {prec} precision",
        f"Processing power (TFLOPS) {prec} precision (base)",
        f"Processing power (TFLOPS) {prec} precision (boost)",
        f"Processing power (GFLOPS) {prec} precision (MAD or FMA)",
        f"Processing power (GFLOPS) {prec} precision (FMA)",
    ]:
        if col in df.columns.values:
            destcol = f"Processing power (GFLOPS) {prec} precision"
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(",", "")  # get rid of commas
            df[col] = df[col].str.extract(r"^([\d\.]+)", expand=False)
            df[col] = pd.to_numeric(df[col]) * 1000.0  # change to GFLOPS
            df = merge(df, destcol, col)

for col in ["Performance (MFLOPS FP32)"]:
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

# remove references from end of model/transistor names
for col in ["Model", "Transistors (million)"]:
    df[col] = df[col].str.replace(referencesAtEnd, "", regex=True)
    # then take 'em out of the middle too
    df[col] = df[col].str.replace(r"\[\d+\]", "", regex=True)

# Simple treatment of multiple columns: just grab the first number
for col in [
    "Core clock (MHz)",
    "Memory Bus width (bit)",
    "Memory Size (KiB)",
    "Memory Size (MB)",
    "Memory Size (GB)",
]:
    df[col] = df[col].apply(lambda x: str(x))
    df[col] = df[col].str.extract(r"(\d*\.\d+|\d+)")

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


# merge and numerify {transistor counts, memory size}
for to, frm in [
    ("Transistors (billion)", "Transistors (million)"),
    ("Memory Size (MB)", "Memory Size (KiB)"),
    ("Memory Size (GB)", "Memory Size (MB)"),
]:
    df[to] = df[to].fillna(pd.to_numeric(df[frm], errors="coerce") / 1000.0)
    df[to] = pd.to_numeric(df[to], errors="coerce")

# extract shader (processor) counts
# df = merge(df, 'Core config',
#            'Core config (SM/SMP/Streaming Multiprocessor)',
#            delete=False)
# Intel-specific: extract then get rid of "N Xe cores"
df["Xe cores"] = df["Core config"].str.extract(r"^(\d)+ Xe cores", expand=False)
df["Core config"] = df["Core config"].str.replace(r"^\d+ Xe cores ", "", regex=True)
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
df = merge(df, "Pixel/unified shader count", "Shading units")  # Intel
# note there might be zeroes

df = merge(df, "SM count", "Xe cores")
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

# Simple treatment of fab: just grab the first number
df = merge(df, "Fab (nm)", "Process")
df["Fab (nm)"] = df["Fab (nm)"].apply(lambda x: str(x))
df["Fab (nm)"] = df["Fab (nm)"].str.replace(referencesOnlyAtEnd, "", regex=True)
df["Fab (nm)"] = df["Fab (nm)"].str.extract(r"(\d+)")
# merge in AMD fab stats
for pat in [r"(\d+) nm", r"(?:TSMC N|GloFo |GF |TSMC CLN|Samsung )(\d+)"]:
    df["Fab (extracted)"] = df["Architecture & Fab"].str.extract(pat, expand=False)
    df = merge(df, "Fab (nm)", "Fab (extracted)")

# take first number from "release price" after deleting $ and ,
df["Release Price (USD)"] = (
    df["Release Price (USD)"]
    .str.replace(r"[,\$]", "", regex=True)
    .str.split(" ")
    .str[0]
)

# patch up weird Radeon Pro V series data
# works around Pandas bug https://github.com/pandas-dev/pandas/issues/58461
df.loc[df["Model"] == "Radeon Pro V520 (Navi 12)", "TDP (Watts)"] = 225
df.loc[df["Model"] == "Radeon Pro V620 (Navi 21)", "TDP (Watts)"] = 300

# this cleans up columns to make sure they're not mixed float/text
# also was useful to make sure columns can be converted to Arrow without errors
for col in [
    "Memory Bandwidth (GB/s)",
    "Memory Size (GB)",
    "L2 Cache (MiB)",
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
df["Memory Bandwidth (GB/s)/USD"] = (
    pd.to_numeric(df["Memory Bandwidth (GB/s)"], errors="coerce")
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
df["Memory Bandwidth per Pin (GB/s)"] = pd.to_numeric(
    df["Memory Bandwidth (GB/s)"], errors="coerce"
) / pd.to_numeric(df["Memory Bus width (bit)"], errors="coerce")
df["L2 to DRAM Ratio"] = (
    0.001
    * pd.to_numeric(df["L2 Cache (MiB)"], errors="coerce")
    / pd.to_numeric(df["Memory Size (GB)"], errors="coerce")
)

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

dfpr = pd.DataFrame(
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
    ),
)

config_default = {
    "df": df,
    "x": "Launch:T",
    "xscale": "linear",
    "yscale": "log",
    "shape": "Vendor",
    "selection_color_fields": [],
    "selection_shape_fields": [],
    "tooltip": ["Launch:T", "Vendor:N", "Model:N"],
}

vendor_colormap = alt.Color(
    "Vendor:N",
    scale=colormap,
)

# if we need to knock out a set of null rows:
# "df": df[df["Fab (nm)"].notnull()],

config = {
    "bw": {
        "df": df[df["Memory Bandwidth (GB/s)"].notnull()],
        "title": "Memory Bandwidth over Time",
        "y": "Memory Bandwidth (GB/s):Q",
        "color": "Memory Bus type:N",
        "selection_color_fields": ["Memory Bus type"],
        "selection_shape_fields": ["Vendor"],
        "tooltip": [
            "Memory Bus type",
            "Memory Bandwidth (GB/s)",
            "Memory Bus width (bit)",
        ],
    },
    "bus": {
        "title": "Memory Bus Width over Time",
        "y": "Memory Bus width (bit):Q",
        "color": "Memory Bus type:N",
        "tooltip": ["Memory Bus type", "Memory Bus width (bit)"],
    },
    "memsz": {
        "title": "Memory Capacity over Time",
        "y": "Memory Size (GB):Q",
        "color": "Memory Bus type:N",
        "tooltip": ["Memory Size (GB)", "Memory Bus type"],
    },
    "l2": {
        "df": df[df["L2 Cache (MiB)"].notnull()],
        "title": "L2 Cache Size over Time",
        "y": "L2 Cache (MiB):Q",
        "color": "Memory Bus type:N",
        "tooltip": ["L2 Cache (MiB)"],
    },
    "l2dram": {
        "df": df[df["L2 to DRAM Ratio"].notnull()],
        "title": "L2 to DRAM Ratio",
        "y": "L2 to DRAM Ratio:Q",
        "color": "Memory Bus type:N",
        "tooltip": ["L2 Cache (MiB)", "Memory Size (GB)", "L2 to DRAM Ratio"],
    },
    "bwpin": {
        "df": df[df["Memory Bandwidth per Pin (GB/s)"].notnull()],
        "title": "Memory Bandwidth per Pin over Time",
        "y": "Memory Bandwidth per Pin (GB/s):Q",
        "color": "Memory Bus type:N",
        "tooltip": [
            "Memory Bus type",
            "Memory Bandwidth (GB/s)",
            "Memory Bus width (bit)",
        ],
    },
    "pr": {
        "title": "Processing Power over Time",
        "df": dfpr[dfpr["Processing power (GFLOPS)"].notnull()],
        "y": "Processing power (GFLOPS):Q",
        "color": "Datatype:N",
        "tooltip": ["Datatype", "Processing power (GFLOPS)"],
    },
    "sm": {
        "df": df[df["SM count"].notnull()],
        "title": "SM count over Time",
        "y": "SM count:Q",
        "shape": "GPU Type",
        "color": vendor_colormap,
        "tooltip": ["SM count"],
    },
    "sh": {
        "title": "Shader count over Time",
        "df": df[
            df["Pixel/unified shader count"].notnull()
            & (df["Pixel/unified shader count"] != 0)
        ],
        "y": "Pixel/unified shader count:Q",
        "shape": "GPU Type",
        "color": vendor_colormap,
        "tooltip": ["Pixel/unified shader count"],
    },
    "die": {
        "df": df[df["Die size (mm2)"].notnull()],
        "title": "Die Size over Time",
        "y": "Die size (mm2):Q",
        "shape": "GPU Type",
        "color": vendor_colormap,
        "tooltip": ["Die size (mm2)"],
    },
    "xt": {
        "df": df[df["Transistors (billion)"].notnull()],
        "title": "Transistor Count over Time",
        "y": "Transistors (billion):Q",
        "shape": "GPU Type",
        "color": vendor_colormap,
        "tooltip": ["Transistors (billion)"],
    },
    "fab": {
        "title": "Feature size over Time",
        "y": "Fab (nm):Q",
        "color": vendor_colormap,
        "tooltip": ["Fab (nm)"],
    },
    "clk": {
        "title": "Clock rate over Time",
        "df": df[df["Core clock (MHz)"].notnull()],
        "y": "Core clock (MHz):Q",
        "yscale": "linear",
        "shape": "GPU Type",
        "color": vendor_colormap,
        "tooltip": ["Core clock (MHz)", "GPU Type"],
    },
    "cost": {
        "df": df[df["Release Price (USD)"].notnull()],
        "title": "Release price over Time",
        "y": "Release Price (USD):Q",
        "color": vendor_colormap,
        "shape": "GPU Type",
        "tooltip": ["Release Price (USD)"],
    },
    "fperdollar": {
        "df": df[df["Single precision GFLOPS/USD"].notnull()],
        "title": "GFLOPS per Dollar over Time",
        "y": "Single precision GFLOPS/USD:Q",
        "yscale": "linear",
        "color": vendor_colormap,
        "shape": "GPU Type",
        "tooltip": [
            "Single-precision GFLOPS",
            "Release Price (USD)",
            "Single precision GFLOPS/USD",
        ],
    },
    "bwperdollar": {
        "df": df[df["Memory Bandwidth (GB/s)/USD"].notnull()],
        "title": "Memory Bandwidth per Dollar over Time",
        "y": "Memory Bandwidth (GB/s)/USD:Q",
        "yscale": "linear",
        "color": vendor_colormap,
        "shape": "GPU Type",
        "tooltip": [            
            "Release Price (USD)",
            "Memory Bandwidth (GB/s)",
            "Single-precision GFLOPS",
        ],
    },
    "fpw": {
        "df": df[df["Single precision GFLOPS/Watt"].notnull()],
        "title": "GFLOPS per Watt over Time",
        "color": "Fab (nm):N",
        "y": "Single precision GFLOPS/Watt:Q",
        "shape": "Vendor",
        "tooltip": [
            "Fab (nm)",
            "Single-precision GFLOPS",
            "TDP (Watts)",
            "Single precision GFLOPS/Watt",
        ],
    },
    "fpwsp": {
        "title": "GFLOPS per Watt vs. Peak Processing Power",
        "color": "Fab (nm):N",
        "x": "Single-precision GFLOPS:Q",
        "xscale": "log",
        "y": "Single precision GFLOPS/Watt:Q",
        "shape": "Vendor",
        "tooltip": [
            "Fab (nm)",
            "Single-precision GFLOPS",
            "TDP (Watts)",
            "Single precision GFLOPS/Watt",
        ],
    },
    "fpwbw": {
        "title": "GFLOPS per Watt vs. Memory Bandwidth",
        "color": "Fab (nm):N",
        "x": "Memory Bandwidth (GB/s):Q",
        "xscale": "log",
        "y": "Single precision GFLOPS/Watt:Q",
        "shape": "Vendor",
        "tooltip": ["Fab (nm)", "Memory Bandwidth (GB/s)"],
    },
    "ai": {
        "title": "Arithmetic Intensity over Time",
        "df": df[df["Arithmetic intensity (FLOP/B)"].notnull()],
        "color": vendor_colormap,
        "y": "Arithmetic intensity (FLOP/B):Q",
        "shape": "GPU Type",
        "tooltip": ["Arithmetic intensity (FLOP/B)"],
    },
    "aisp": {
        "title": "Arithmetic Intensity vs. Peak Processing Power",
        "color": "Fab (nm):N",
        "x": "Single-precision GFLOPS:Q",
        "xscale": "log",
        "y": "Arithmetic intensity (FLOP/B):Q",
        "shape": "Vendor",
        "tooltip": [
            "Fab (nm)",
            "Single-precision GFLOPS",
            "Memory Bandwidth (GB/s)",
        ],
    },
    "aibw": {
        "title": "Arithmetic Intensity vs. Memory Bandwidth",
        "color": "Fab (nm):N",
        "x": "Memory Bandwidth (GB/s):Q",
        "xscale": "log",
        "y": "Arithmetic intensity (FLOP/B):Q",
        "shape": "Vendor",
        "tooltip": [
            "Fab (nm)",
            "Single-precision GFLOPS",
            "Memory Bandwidth (GB/s)",
        ],
    },
    "pwr": {
        "df": df[df["TDP (Watts)"].notnull()],
        "title": "Power over Time",
        "y": "TDP (Watts)",
        "shape": "Vendor:N",
        "color": "Fab (nm):N",
        "tooltip": ["Fab (nm)", "TDP (Watts)"],
    },
    "pwrdens": {
        "df": df[df["Watts/mm2"].notnull()],
        "title": "Power density over Time",
        "y": "Watts/mm2",
        "yscale": "linear",
        "shape": "Vendor:N",
        "color": "Fab (nm):N",
        "tooltip": ["Fab (nm)", "TDP (Watts)", "Die size (mm2)", "Watts/mm2"],
    },
    "transdens": {
        "df": df[df["Transistor Density (B/mm2)"].notnull()],
        "title": "Transistor density over Time",
        "y": "Transistor Density (B/mm2)",
        "shape": "Vendor:N",
        "color": "Fab (nm):N",
        "tooltip": [
            "Fab (nm)",
            "Transistor Density (B/mm2)",
            "Die size (mm2)",
        ],
    },
    "tdpvsdens": {
        "title": "GFLOPS per Area vs. GFLOPS per Watt",
        "x": "Single precision GFLOPS/Watt:Q",
        "xscale": "log",
        "y": "Single-precision GFLOPS/mm2:Q",
        "color": "Fab (nm):N",
        "shape": "Vendor:N",
        "tooltip": [
            "Fab (nm)",
            "Die size (mm2)",
            "Transistor Density (B/mm2)",
            "Single precision GFLOPS/Watt",
            "Single-precision GFLOPS/mm2",
        ],
    },
}

# merge in defaults
# first prepend default for appropriate fields
for key in config:
    for field in ["tooltip"]:
        if field in config[key] and field in config_default:
            config[key][field] = config_default[field] + config[key][field]
for key in config:
    # d | other: The values of other take priority when d and other share keys.
    config[key] = config_default | config[key]

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

script_dir = os.path.dirname(os.path.realpath("__file__"))
rel_path = "plots"
outputdir = os.path.join(script_dir, rel_path)


def stripShorthand(str):
    if str[-2] == ":" and str[-1].isupper():
        str = str[:-2]
    if str[0] == "[" and str[-1] == "]":
        str = str[1:-1]
    return str


for key in config:
    color_selection = alt.selection_point(encodings=["color"], bind="legend")
    shape_selection = alt.selection_point(encodings=["shape"], bind="legend")
    combined_selection = alt.PredicateComposition(
        {"and": [color_selection, shape_selection]}
    )

    # selection = alt.selection_point(encodings=["color", "shape"], bind="legend")
    # selection_shape = alt.selection_point(
    #     fields=config[key]["selection_shape_fields"], bind="legend"
    # )
    c = config[key]
    title = c["title"]
    base = (
        alt.Chart(c["df"])
        .encode(
            x=alt.X(
                c["x"],
                scale=alt.Scale(type=c["xscale"]),
            ),
            y=alt.Y(
                c["y"],
                scale=alt.Scale(type=c["yscale"]),
            ),
        )
        .properties(width=1213, height=750, title=title)
    )
    chart = (
        base.mark_point()
        .encode(
            color=c["color"],
            shape=c["shape"],
            opacity=alt.condition(combined_selection, alt.value(1.0), alt.value(0.1)),
            tooltip=c["tooltip"],
        )
        .interactive()
        .add_params(color_selection)
        .add_params(shape_selection)
    )

    # indexed by [xscale][yscale]
    # there is no model for log-log
    regression_method = {
        "linear": {"linear": "linear", "log": "exp"},
        "log": {"linear": "log", "log": None},
    }

    if False and regression_method[c["xscale"]][c["yscale"]]:
        # we set lchart to layer (chart, base)
        line_selection = alt.selection_point(encodings=["color"], bind="legend")
        # bind_checkbox = alt.binding_checkbox(name="Toggle best-fit line")
        # param_checkbox = alt.param(bind=bind_checkbox)
        chart = (
            alt.layer(
                chart,
                base.transform_regression(
                    on=stripShorthand(c["x"]),
                    regression=stripShorthand(c["y"]),
                    # groupby=[stripShorthand(c["color"])],
                    method=regression_method[c["xscale"]][c["yscale"]],
                )
                .mark_line(opacity=0.5)
                .transform_calculate(Fit='"LinReg"')
                .encode(stroke="Fit:N"),
                base.transform_regression(
                    on=stripShorthand(c["x"]),
                    regression=stripShorthand(c["y"]),
                    # groupby=[stripShorthand(c["color"])],
                    method=regression_method[c["xscale"]][c["yscale"]],
                    params=True,
                )
                .mark_text(align="left", lineBreak="\n")
                .encode(
                    x=alt.value(150),  # pixels from left
                    y=alt.value(250),  # pixels from top
                    text="params:N",
                )
                .transform_calculate(
                    params='"r² = " + round(datum.rSquared * 100)/100 + \
    "      y = " + round(datum.coef[0] * 10)/10 + " + e ^ (" + \
    round(datum.coef[1] * 10000)/10000 + "x" + ")" + \n + " "'
                ),
            )
            .encode(
                # opacity=alt.condition(param_checkbox, alt.value(1.0), alt.value(0.05)),
                # alt.Color("Regression:N"),
            )
            .interactive()
            .add_params(line_selection)
            .resolve_scale(color="independent")
        )
        # b = base.transform_regression(
        #     on=stripShorthand(c["x"]),
        #     regression=stripShorthand(c["y"]),
        #     method=regression_method[c["xscale"]][c["yscale"]],
        #     params=True,
        # ).mark_line()
        # print(title, altair_transform.extract_data(b))

    # chart = (
    #     chart.interactive()
    #     .add_params(color_selection)
    #     .add_params(shape_selection)
    #     # .add_params(param_checkbox)
    # )

    chart.save(os.path.join(outputdir, title + ".pdf"), engine="vl-convert")
    # vf.save_html(chart, os.path.join(outputdir, title + ".html"))
    chart.save(os.path.join(outputdir, title + ".html"), inline=True)
    readme += f"- {title} [[html](plots/{title}.html), [pdf](plots/{title}.pdf)]\n"

with open(os.path.join(outputdir, "../README.md"), "w") as f:
    f.write(readme)

# print particular row
# pd.set_option("display.max_columns", None)
# row = df[df["Model"] == "NAME OF MODEL"]
# row = row[row.columns[~row.isnull().all()]]
# print(row)

# df.to_csv("/tmp/gpu.csv", encoding="utf-8")

# if we're looking for a particular value, here's how we find it:
# search_val = "1066"
# for col in df.columns:
#     try:
#         if df[col].dtype.kind == "O":
#             idx = df[col].tolist().index(search_val)
#             print(f"val: {search_val}, col: {col}, index: {idx}")
#     except valueError:
#         pass
