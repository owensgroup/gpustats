#!/bin/sh
''''exec "$(dirname "$0")/.venv/bin/python" "$0" "$@" 2>/dev/null || exec python3 "$0" "$@" #'''
# ^ sh/Python polyglot shebang: sh execs ./.venv/bin/python if present, else python3.
# Python sees the second line as a triple-quoted string literal (a no-op statement).

import warnings
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

warnings.filterwarnings(
    "ignore",
    message="Downcasting object dtype arrays",
    category=FutureWarning,
)
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


def merge(df, dst, src, replaceNoWithNaN=False, delete=True, silentlySkip=True):
    if silentlySkip and src not in df.columns:
        return df
    df[src] = df[src].replace("\u2014", np.nan)  # em-dash
    if replaceNoWithNaN:
        df[src] = df[src].replace("No", np.nan)
    if dst not in df.columns:
        df[dst] = df[src]
    else:
        df[dst] = df[dst].fillna(df[src]).infer_objects(copy=False)
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


# Map of unit suffix -> multiplier to reach the "base" unit. The base unit is
# context-dependent (MiB for cache, GFLOPS for compute) \u2014 callers pick a unit
# table compatible with their column.
UNIT_MIB = {  # base = MiB (binary)
    "KB": 1.0 / 1024.0, "KiB": 1.0 / 1024.0,
    "MB": 1.0,          "MiB": 1.0,
    "GB": 1024.0,       "GiB": 1024.0,
}


def to_number(series, scale=1.0, unit_pattern=None, unit_table=None,
              anchored=True):
    """Extract a numeric value from each cell of `series` and return a float Series.

    Strips commas, then either:
      - extracts the leading number and applies `scale` (default), or
      - extracts (number, unit) and multiplies by unit_table[unit] when
        `unit_pattern` is given (e.g. "KB|MB|MiB|GB|GiB").

    `anchored` controls whether the number must be at the start of the cell
    (True) or can appear anywhere (False \u2014 first match wins)."""
    s = series.astype(str).str.replace(",", "")
    num_re = r"(\d*\.\d+|\d+)"
    if anchored:
        num_re = "^" + num_re
    if unit_pattern:
        m = s.str.extract(rf"{num_re}\s*({unit_pattern})", expand=True)
        return pd.to_numeric(m[0], errors="coerce") * m[1].map(unit_table)
    return (
        pd.to_numeric(s.str.extract(num_re, expand=False), errors="coerce")
        * scale
    )


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
    html = re.sub("<span[^>]*>\u00d7</span>", "\u00d7", html)  # unwrap × from span styling
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
        # it. Negative lookbehind preserves cache-level identifiers like
        # "L1"/"L2"/"L3" so columns such as "Cache L2 (MiB)" survive intact.
        df.columns = [
            " ".join([re.sub(r"(?<!L)[\d,]+$", "", word) for word in col.split()])
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

        # ignore errors
        # syntax: from: to
        df = df.rename(
            columns={
                "Release date & price": "Release Date & Price",
                "Release date": "Launch",
                "Launch Date": "Launch",
                "Architecture & fab": "Architecture & Fab",
                "Architecture Fab": "Architecture & Fab",
                "Architecture (Fab)": "Architecture & Fab",
                "Processing power (TFLOPS) Bfloat16": "Processing power (TFLOPS) Half precision",
                "Die Size (mm2)": "Die size (mm2)",
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
merge_map = {
    "Processing power (GFLOPS) Single precision": [
        "Processing power (GFLOPS)",
        "Performance (GFLOPS FP32)",
    ],
    "Memory Bandwidth (GB/s)": [
        "Memory configuration Bandwidth (GB/s)",
    ],
    "TDP (Watts)": [
        "TDP (Watts) Max.",
        "TDP (W)",
        "Combined TDP Max. (W)",
        "TDP /idle (Watts)",
    ],
    "Model": [
        "Model (Codename)",
        "Model (Code name)",
        "Model name",
        "Model name (Architecture)",
        "Code name (console model)",
        "Branding and Model",
        # "Chip (Device)",
        "Model: Mobility Radeon",  # replace when AMD page updated
    ],
    "Core clock (MHz)": [
        "Shaders Base clock (MHz)",
        "Shader clock (MHz)",
        "Clock rate Base (MHz)",
        "Clock rate (MHz)",
        "Clock speeds Base core clock (MHz)",
        "Core Clock (MHz)",
        "Clock rate Core (MHz)",
        "Clock speed Core (MHz)",
        "Clock speed Average (MHz)",
        "Core Clock rate (MHz)",
        "Clock rate (MHz) Core (MHz)",
        "Clock speed Shader (MHz)",
        "Core Clock (MHz) Base",
    ],
    "Core config": [
        "Core Config",
        "Shaders Core config",
    ],
    "Transistors Die Size": [
        "Transistors & die size",
    ],
    "Memory Size (MB)": [
        "Memory Size (MiB)",
    ],
    "Memory Size (GB)": [
        "Memory Size (GiB)",
    ],
    "Memory Bus type": [
        "Memory RAM type",
        "Memory Type",
        "Memory configuration DRAM type",
    ],
    "Memory Bus width (bit)": [
        "Memory configuration Bus width (bit)",
    ],
    "Release Price (USD)": [
        "Release price (USD)",
        "Release price (USD) MSRP",
    ],
}
for dst, srcs in merge_map.items():
    for src in srcs:
        df = merge(df, dst, src)

df = df.copy()  # this unfragments the df, better performance
# get only the number out of {TBP,TDP}
for col in ["TBP", "TDP"]:
    df[f"{col} (extracted)"] = pd.to_numeric(
        df[col].str.extract(r"(\d+)", expand=False)
    )
    df = merge(df, "TDP (Watts)", f"{col} (extracted)")

# Normalize L2 cache columns into "L2 Cache (MiB)". Wikipedia uses many
# spellings; NVIDIA puts units in the column header, Intel puts them in cells.
# MB is treated as MiB (within ~5%, matches existing convention).
for src, scale in [
    ("Cache L2 (MiB)", 1.0),
    ("L2 Cache(MiB)", 1.0),
    ("Cache L2 (MB)", 1.0),
    ("Cache L2 (KB)", 1.0 / 1024.0),
]:
    if src in df.columns:
        df[src] = to_number(df[src], scale=scale)
        df = merge(df, "L2 Cache (MiB)", src)
for src in ["L2 cache", "Cache L2"]:
    if src in df.columns:
        df[src] = to_number(
            df[src], unit_pattern="KB|KiB|MB|MiB|GB|GiB",
            unit_table=UNIT_MIB, anchored=False,
        )
        df = merge(df, "L2 Cache (MiB)", src)

df["Release Price (USD)"] = df["Release Price (USD)"].str.extract(
    r"^\$?([\d,]+)", expand=False
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

# Normalize per-precision GFLOPS columns. Multiple Wikipedia source-column
# shapes need merging into "Processing power (GFLOPS) {prec} precision",
# each with a known unit scale. Order across precisions is preserved (matches
# original code): all-precs TFLOPS pass, then MFLOPS-only-Single, then Boost
# pass with cleanup and rename.
flops_tflops_sources = [
    # (column-name template, scale to GFLOPS)
    ("Processing power TFLOPS {prec}",                          1000),
    ("Processing power (TFLOPS) {prec}",                        1000),
    ("Processing power (TFLOPS) {prec} precision",              1000),
    ("Processing power (TFLOPS) {prec} precision (base)",       1000),
    ("Processing power (TFLOPS) {prec} precision (boost)",      1000),
    # XXX preserved-behavior: these say (GFLOPS) but are scaled by 1000.
    # Possibly a bug in the original — investigate whether source cells are
    # actually in TFLOPS despite the column name claiming GFLOPS.
    ("Processing power (GFLOPS) {prec} precision (MAD or FMA)", 1000),
    ("Processing power (GFLOPS) {prec} precision (FMA)",        1000),
]
for prec in ["Single", "Double", "Half"]:
    dst = f"Processing power (GFLOPS) {prec} precision"
    for tmpl, scale in flops_tflops_sources:
        src = tmpl.format(prec=prec)
        if src in df.columns:
            df[src] = to_number(df[src], scale=scale)
            df = merge(df, dst, src)

# MFLOPS only applies to single precision
src = "Performance (MFLOPS FP32)"
if src in df.columns:
    df[src] = to_number(df[src], scale=1.0 / 1000.0)
    df = merge(df, "Processing power (GFLOPS) Single precision", src)

# Boost: "Processing power (GFLOPS) {prec}" merge, plus cross-precision
# refs ("1/16 SP", "2x SP"), final cleanup, and rename. Single precision is
# processed first so spcol is set before Half/Double consult it.
for prec in ["Single", "Double", "Half"]:
    dst = f"Processing power (GFLOPS) {prec} precision"
    spcol = "Single-precision GFLOPS"
    src = f"Processing power (GFLOPS) {prec}"
    if src in df.columns:
        df[src] = to_number(df[src], scale=1.0)
        df = merge(df, dst, src)

    if prec != "Single":
        df.loc[df[dst] == "1/16 SP", dst] = pd.to_numeric(df[spcol]) / 16
        df.loc[df[dst] == "2x SP", dst] = pd.to_numeric(df[spcol]) * 2

    # final cleanup: any remaining string values in dst (e.g. from the
    # earlier merge_map pass) get normalized
    df[dst] = to_number(df[dst], scale=1.0)
    df = df.rename(columns={dst: f"{prec}-precision GFLOPS"})

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
    df[col] = to_number(df[col], anchored=False)

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

# Extract Xe core counts from Intel Xe tables
for col, pattern in [
    ("Core config", r"(\d+) Xe cores"),  # Alchemist (Xe)
    ("Core Core Config", r"(\d+) Xe2-cores"),  # Battlemage (Xe2)
]:
    if col in df.columns:
        df["SM count (extracted)"] = df[col].str.extract(pattern, expand=False)
        df = merge(df, "SM count", "SM count (extracted)")
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
        "tooltip": [
            "Arithmetic intensity (FLOP/B)",
            "Single-precision GFLOPS",
            "Memory Bandwidth (GB/s)",
        ],
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
            "Single-precision GFLOPS",
            "TDP (Watts)",
            "Die size (mm2)",
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
