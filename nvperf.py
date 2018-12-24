#!/usr/bin/env python3

import pandas as pd
import numpy as np
import requests
import re
import os
import json

from altair import *

from fileops import save

from collections import Counter

data = {
    'NVIDIA': {
        'url': 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
    },
    'AMD': {
        'url': 'https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units',
    }
}

referencesAtEnd = r'(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$'

for vendor in ['NVIDIA', 'AMD']:
    # requests.get handles https
    html = requests.get(data[vendor]['url']).text
    # oddly, some dates look like:
    # <td><span class="sortkey" style="display:none;speak:none">000000002010-02-25-0000</span><span style="white-space:nowrap">Feb 25, 2010</span></td>
    html = re.sub(
        r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', '', html)
    html = re.sub(r'<span[^>]*>([^<]+)</span>', r'\1', html)
    html = re.sub('\xa0', ' ', html)  # non-breaking space -> ' '
    html = re.sub(r'&#160;', ' ', html)  # non-breaking space -> ' '
    html = re.sub(r'<br />', ' ', html)  # breaking space -> ' '
    html = re.sub('\u2012', '-', html)  # figure dash -> '-'
    html = re.sub('\u2013', '-', html)  # en-dash -> '-'
    html = re.sub(r'mm<sup>2</sup>', 'mm2', html)  # mm^2 -> mm2
    html = re.sub('\u00d710<sup>6</sup>', '\u00d7106', html)  # 10^6 -> 106
    html = re.sub('\u00d710<sup>9</sup>', '\u00d7109', html)  # 10^9 -> 109
    html = re.sub(r'<sup>\d+</sup>', '', html)  # delete footnotes
    # with open('/tmp/%s.html' % vendor, 'wb') as f:
    #     f.write(html.encode('utf8'))

    dfs = pd.read_html(html, match='Launch', parse_dates=True)
    for idx, df in enumerate(dfs):
        # Multi-index to index
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        # Get rid of 'Unnamed' column names
        df.columns = [re.sub(' Unnamed: [0-9]+_level_[0-9]+', '', col)
                      for col in df.columns.values]
        # If a column-name word ends in a number or number,comma,number, delete
        # it
        df.columns = [' '.join([re.sub('[\d,]+$', '', word) for word in col.split()])
                      for col in df.columns.values]
        # If a column-name word ends with one or more '[x]',
        # where 'x' is a lower-case letter or number, delete it
        df.columns = [' '.join([re.sub(r'(?:\[[a-z0-9]+\])+$', '', word) for word in col.split()])
                      for col in df.columns.values]
        # Get rid of hyphenation in column names
        df.columns = [col.replace('- ', '') for col in df.columns.values]
        # Get rid of space after slash in column names
        df.columns = [col.replace('/ ', '/') for col in df.columns.values]
        # Get rid of trailing space in column names
        df.columns = df.columns.str.strip()

        df['Vendor'] = vendor

        # make sure Launch is a string (dtype=object) before parsing it
        df['Launch'] = df['Launch'].apply(lambda x: str(x))
        df['Launch'] = df['Launch'].str.replace(referencesAtEnd, '')
        df['Launch'] = df['Launch'].apply(
            lambda x: pd.to_datetime(x,
                                     infer_datetime_format=True,
                                     errors='coerce'))
        # if we have duplicate column names, we will see them here
        # pd.concat will fail if so
        if ([c for c in Counter(df.columns).items() if c[1] > 1]):
            # so remove them
            # https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
            df = df.loc[:, ~df.columns.duplicated()]
        # this assignment is because I don't have a way to do the above line
        # in-place
        dfs[idx] = df

    data[vendor]['dfs'] = dfs

df = pd.concat(data['NVIDIA']['dfs'] + data['AMD']['dfs'], ignore_index=True)

# print all columns
# print(list(df))


def merge(df, dst, src, replaceNoWithNaN=False, delete=True):
    if replaceNoWithNaN:
        df[src] = df[src].replace('No', np.nan)
    df[dst] = df[dst].fillna(df[src])
    if delete:
        df.drop(src, axis=1, inplace=True)
    return df


# merge related columns
df = merge(df, 'Model', 'Model Units')
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS)')
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS) Single precision (MAD+MUL)',
           replaceNoWithNaN=True)
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS) Single precision (MAD or FMA)',
           replaceNoWithNaN=True)
df = merge(df, 'Processing power (GFLOPS) Double precision',
           'Processing power (GFLOPS) Double precision (FMA)',
           replaceNoWithNaN=True)
df = merge(df, 'Memory Bandwidth (GB/s)',
           'Memory configuration Bandwidth (GB/s)')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) Max.')
df = merge(df, 'TDP (Watts)', 'TBP (W)')
# fix up watts?
# df['TDP (Watts)'] = df['TDP (Watts)'].str.extract(r'<([\d\.]+)', expand=False)
df = merge(df, 'Model', 'Model (Codename)')
# df = merge(df, 'Model', 'Chip (Device)')
# replace when AMD page updated
df = merge(df, 'Model', 'Model: Mobility Radeon')
df = merge(df, 'Core clock (MHz)', 'Clock rate Base (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speeds Base core clock (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speeds R/F.E Base core clock (MHz)')
df = merge(df, 'Core clock (MHz)', 'Core Clock (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock rate Core (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speed Core (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speed Average (MHz)')
df = merge(df, 'Core clock (MHz)', 'Core Clock rate (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock rate (MHz) Core (MHz)')
df = merge(df, 'Core config', 'Core Config')
df = merge(df, 'Memory Bus type', 'Memory RAM type')
df = merge(df, 'Memory Bus type', 'Memory Type')
df = merge(df, 'Memory Bus type', 'Memory configuration DRAM type')
df = merge(df, 'Memory Bus width (bit)',
           'Memory configuration Bus width (bit)')
df = merge(df, 'Release Price (USD)', 'Release price (USD) MSRP')

# filter out {Chips, Code name, Core config}: '^[2-9]\u00d7'
df = df[~df['Chips'].str.contains(r'^[2-9]\u00d7', re.UNICODE, na=False)]
df = df[~df['Code name'].str.contains(r'^[2-9]\u00d7', re.UNICODE, na=False)]
df = df[~df['Core config'].str.contains(
    r'^[2-9]\u00d7', re.UNICODE, na=False)]
# filter out if Model ends in [xX]2
df = df[~df['Model'].str.contains('[xX]2$', na=False)]
# filter out {transistors, die size} that end in x2
df = df[~df['Transistors (million)'].str.contains(
    r'\u00d7[2-9]$', re.UNICODE, na=False)]
df = df[~df['Die size (mm2)'].str.contains(
    r'\u00d7[2-9]$', re.UNICODE, na=False)]

# merge GFLOPS columns with "Boost" column headers and rename
for prec in ['Double', 'Single', 'Half']:
    col = 'Processing power (GFLOPS) %s precision' % prec
    if prec != 'Half':
        df = merge(
            df, col, 'Processing power (GFLOPS) %s precision Base Core (Base Boost) (Max Boost 2.0)' % prec)
    for srccol in ['Processing power (GFLOPS) %s precision Base Core (Base Boost) (Max Boost 3.0)',
                   'Processing power (GFLOPS) %s precision R/F.E Base Core Reference (Base Boost) F.E. (Base Boost) R/F.E. (Max Boost 4.0)',
                   'Processing power (GFLOPS) %s'
                   ]:
        df = merge(df, col, srccol % prec)

    # pick the first number we see as the actual number
    df[col] = df[col].astype(str)
    df[col] = df[col].str.extract(r'^([\d\.]+)', expand=False)

    # convert TFLOPS to GFLOPS
    # tomerge = 'Processing power (TFLOPS) %s Prec.' % prec
    # df[col] = df[col].fillna(
    #     pd.to_numeric(df[tomerge].str.split(' ').str[0], errors='coerce') * 1000.0)
    # df.drop(tomerge, axis=1, inplace=True)

    df = df.rename(columns={col: '%s-precision GFLOPS' % prec})

# split out 'transistors die size'
# example: u'292\u00d7106 59 mm2'
for exponent in ['\u00d7106', '\u00d7109', 'B']:
    dftds = df['Transistors Die Size'].str.extract('^([\d\.]+)%s (\d+) mm2' % exponent,
                                                   expand=True)
    if exponent == '\u00d7106':
        df['Transistors (million)'] = df['Transistors (million)'].fillna(
            pd.to_numeric(dftds[0], errors='coerce'))
    if exponent == '\u00d7109' or exponent == 'B':
        df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
            pd.to_numeric(dftds[0], errors='coerce'))
    df['Die size (mm2)'] = df['Die size (mm2)'].fillna(
        pd.to_numeric(dftds[1], errors='coerce'))

# some AMD chips have core/boost in same entry, take first number
df['Core clock (MHz)'] = df['Core clock (MHz)'].astype(
    str).str.split(' ').str[0]

df['Memory Bus width (bit)'] = df['Memory Bus width (bit)'].str.split(
    ' ').str[0]
df['Memory Bus width (bit)'] = df['Memory Bus width (bit)'].str.split(
    '/').str[0]
df['Memory Bus width (bit)'] = df['Memory Bus width (bit)'].str.split(
    ',').str[0]
# strip out bit width from combined column
df['bus'] = df['Memory Bus type & width (bit)'].str.extract(
    '(\d+)-bit', expand=False)
df['bus'] = df['bus'].fillna(pd.to_numeric(df['bus'], errors='coerce'))
df = merge(df, 'Memory Bus width (bit)', 'bus', delete=False)
# collate memory bus type and take first word only, removing chud as
# appropriate
df = merge(df, 'Memory Bus type',
           'Memory Bus type & width (bit)', delete=False)
df['Memory Bus type'] = df['Memory Bus type'].str.split(' ').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split(',').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split('/').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split('[').str[0]
df.loc[df['Memory Bus type'] == 'EDO', 'Memory Bus type'] = 'EDO VRAM'


# merge transistor counts
df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
    pd.to_numeric(df['Transistors (million)'], errors='coerce') / 1000.0)

# extract shader (processor) counts
df = merge(df, 'Core config',
           'Core config (SM/SMP/Streaming Multiprocessor)',
           delete=False)
df = merge(df, 'Core config',
           'Core config CUDA cores (SM/SMP/Streaming Multiprocessor)',
           delete=False)
df['Pixel/unified shader count'] = df['Core config'].str.split(':').str[0]
# this converts core configs like "120(24x5)" to "120"
df['Pixel/unified shader count'] = df['Pixel/unified shader count'].str.split(
    '(').str[0]
# now convert text to numbers
df['Pixel/unified shader count'] = pd.to_numeric(
    df['Pixel/unified shader count'], downcast='integer', errors='coerce')
df = merge(df, 'Pixel/unified shader count', 'Stream processors')
df = merge(df, 'Pixel/unified shader count', 'Shaders Cuda cores (total)')
# note there might be zeroes

df['SM count (extracted)'] = df['Core config'].str.extract(r'\((\d+ SM[MX])\)',
                                                           expand=False)
df = merge(df, 'SM count', 'SM count (extracted)')
for smcount in [
        # GF 10xx SM counts
        'Core config (SM/SMP/Streaming Multiprocessor)',
        # Volta series
        'Core config CUDA cores (SM/SMP/Streaming Multiprocessor)',
]:
    df['SM count (extracted)'] = df[smcount].str.extract(r'\((\d+)\)',
                                                         expand=False)
    df = merge(df, 'SM count', 'SM count (extracted)')
df = merge(df, 'SM count', 'SMX count')


# merge in AMD fab stats
df['Architecture (Fab) (extracted)'] = df[
    'Architecture (Fab)'].str.extract(r'\((\d+) nm\)', expand=False)
df = merge(df, 'Fab (nm)', 'Architecture (Fab) (extracted)')

# NVIDIA has more complicated names of some newer fabs
# TODO haven't figured this out yet
# for fab in ['TSMC', 'Samsung']:
#     df.loc[df['Fab (nm)'].str.match(
#         '^' + fab), 'Fab (nm)'] = df['Fab (nm)'].str.extract('^' + fab + r'(\d+)', expand=False)


# take first number from "release price" after deleting $ and ,
df['Release Price (USD)'] = df['Release Price (USD)'].str.replace(
    r'[,\$]', '').str.split(' ').str[0]

for col in ['Memory Bandwidth (GB/s)',
            'TDP (Watts)',
            'Fab (nm)',
            'Release Price (USD)',
            ]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# compute arithmetic intensity and FLOPS/Watt
df['Arithmetic intensity (FLOP/B)'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['Memory Bandwidth (GB/s)'], errors='coerce')
df['Single precision FLOPS/Watt'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['TDP (Watts)'], errors='coerce')
df['Single precision FLOPS/USD'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / df['Release Price (USD)']

# remove references from end of model names
df['Model'] = df['Model'].str.replace(referencesAtEnd, '')
# then take 'em out of the middle too
df['Model'] = df['Model'].str.replace(r'\[\d+\]', '')

# mark mobile processors
df['GPU Type'] = np.where(
    df['Model'].str.contains(r' [\d]+M[X]?|\(Notebook\)'), 'Mobile', 'Desktop')

# values=c("amd"="#ff0000",
#   "nvidia"="#76b900",
#   "intel"="#0860a8",

colormap = Scale(domain=['AMD', 'NVIDIA'],
                 range=['#ff0000', '#76b900'])

bw = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    color='Memory Bus type',
    shape='Vendor',
    tooltip=['Model', 'Memory Bandwidth (GB/s)'],
).properties(
    width=1213,
    height=750
).interactive()

bus = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Memory Bus width (bit):Q',
        scale=Scale(type='log'),
        ),
    color='Memory Bus type',
    shape='Vendor',
    tooltip=['Model', 'Memory Bus width (bit)'],
).properties(
    width=1213,
    height=750
).interactive()

pr = Chart(pd.melt(df,
                   id_vars=['Launch', 'Model', 'GPU Type', 'Vendor'],
                   value_vars=['Single-precision GFLOPS',
                               'Double-precision GFLOPS',
                               'Half-precision GFLOPS'],
                   var_name='Datatype',
                   value_name='Processing power (GFLOPS)')).mark_point().encode(
    x='Launch:T',
    y=Y('Processing power (GFLOPS):Q',
        scale=Scale(type='log'),
        ),
    shape='Vendor',
    color='Datatype',
    tooltip=['Model', 'Processing power (GFLOPS)'],
).properties(
    width=1213,
    height=750
).interactive()


sm = Chart(df).mark_point().encode(
    x='Launch:T',
    y='SM count:Q',
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'SM count'],
).properties(
    width=1213,
    height=750
).interactive()


die = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Die size (mm2):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
    tooltip=['Model', 'Die size (mm2)'],
).properties(
    width=1213,
    height=750
).interactive()


xt = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Transistors (billion):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
    tooltip=['Model', 'Transistors (billion)'],
).properties(
    width=1213,
    height=750
).interactive()


fab = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Fab (nm):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'Fab (nm)'],
).properties(
    width=1213,
    height=750
).interactive()


ai = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Arithmetic intensity (FLOP/B):Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'Arithmetic intensity (FLOP/B)'],
).properties(
    width=1213,
    height=750
).interactive()


fpw = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Single precision FLOPS/Watt:Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'Single precision FLOPS/Watt'],
).properties(
    width=1213,
    height=750
).interactive()


clk = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Core clock (MHz):Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'Core clock (MHz)'],
).properties(
    width=1213,
    height=750
).interactive()


cost = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Release Price (USD):Q',
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
    tooltip=['Model', 'Release Price (USD)'],
).properties(
    width=1213,
    height=750
).interactive()


fperdollar = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Single precision FLOPS/USD:Q',
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
    tooltip=['Model', 'Single precision FLOPS/USD'],
).properties(
    width=1213,
    height=750
).interactive()


# only plot chips with actual feature sizes
fpwsp = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Single-precision GFLOPS:Q',
        scale=Scale(type='log'),
        ),
    y='Single precision FLOPS/Watt:Q',
    shape='Vendor',
    color='Fab (nm):N',
    tooltip=['Model', 'Fab (nm)', 'Single precision FLOPS/Watt'],
).properties(
    width=1213,
    height=750
).interactive()


fpwbw = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    y='Single precision FLOPS/Watt:Q',
    shape='Vendor',
    color='Fab (nm):N',
    tooltip=['Model', 'Fab (nm)', 'Memory Bandwidth (GB/s)'],
).properties(
    width=1213,
    height=750
).interactive()

aisp = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Single-precision GFLOPS:Q',
        scale=Scale(type='log'),
        ),
    y='Arithmetic intensity (FLOP/B):Q',
    shape='Vendor',
    color='Fab (nm):N',
    tooltip=['Model', 'Fab (nm)', 'Single-precision GFLOPS'],
).properties(
    width=1213,
    height=750
).interactive()

aibw = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    y='Arithmetic intensity (FLOP/B):Q',
    shape='Vendor',
    color='Fab (nm):N',
    tooltip=['Model', 'Fab (nm)', 'Memory Bandwidth (GB/s)'],
).properties(
    width=1213,
    height=750
).interactive()

# need != 0 because we're taking a log
sh = Chart(df[df['Pixel/unified shader count'] != 0]).mark_point().encode(
    x='Launch:T',
    y=Y('Pixel/unified shader count:Q',
        scale=Scale(type='log'),
        ),
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
    tooltip=['Model', 'Pixel/unified shader count'],
).properties(
    width=1213,
    height=750
).interactive()

# df.to_csv("/tmp/gpu.csv", encoding="utf-8")

template = """<!DOCTYPE html>
<html>
<head>
  <!-- Import Vega 3 & Vega-Lite 2 (does not have to be from CDN) -->
  <script src="https://cdn.jsdelivr.net/npm/vega@3"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@2"></script>
  <!-- Import vega-embed -->
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@3"></script>
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

readme = "# GPU Statistics\n\nData sourced from [Wikipedia's NVIDIA GPUs page](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units) and [Wikipedia's AMD GPUs page](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units).\n\n"

outputdir = "/Users/jowens/Documents/working/owensgroup/proj/gpustats/plots"
for (chart, title) in [(bw, "Memory Bandwidth over Time"),
                       (bus, "Memory Bus Width over Time"),
                       (pr, "Processing Power over Time"),
                       (sm, "SM count over Time"),
                       (sh, "Shader count over Time"),
                       (die, "Die Size over Time"),
                       (xt, "Transistor Count over Time"),
                       (fab, "Feature size over Time"),
                       (clk, "Clock rate over Time"),
                       (cost, "Release price over Time"),
                       (fperdollar, "FLOPS per Dollar over Time"),
                       (fpw, "FLOPS per Watt over Time"),
                       (fpwsp, "FLOPS per Watt vs. Peak Processing Power"),
                       (fpwbw, "FLOPS per Watt vs. Memory Bandwidth"),
                       (ai, "Arithmetic Intensity over Time"),
                       (aisp, "Arithmetic Intensity vs. Peak Processing Power"),
                       (aibw, "Arithmetic Intensity vs. Memory Bandwidth"),
                       ]:
    # save html
    # print chart.to_dict()
    with open(os.path.join(outputdir, title + '.html'), 'w') as f:
        spec = chart.to_dict()
        # spec['height'] = 750
        # spec['width'] = 1213
        # spec['encoding']['tooltip'] = {"field": "Model", "type": "nominal"}
        # this chart.to_dict -> json.dumps can probably be simplified
        f.write(template.format(spec=json.dumps(spec), title=title))
        # f.write(chart.from_json(spec_str).to_html(
        # title=title, template=template))
        save(chart, df=pd.DataFrame(), plotname=title, outputdir=outputdir,
             formats=['pdf'])
    readme += "- [%s](plots/%s.html)\n" % (title, title)

with open(os.path.join(outputdir, '../README.md'), 'w') as f:
    f.write(readme)
