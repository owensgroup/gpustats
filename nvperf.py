#!/usr/bin/env python

import pandas as pd
import numpy as np
import requests
import re
import os
import json

from altair import *

from fileops import save

data = {
    'NVIDIA': {'url': 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units',
               },
    'AMD': {'url': 'https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units',
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
    html = re.sub(u'\xa0', ' ', html)  # non-breaking space -> ' '
    html = re.sub(r'&#160;', ' ', html)  # non-breaking space -> ' '
    html = re.sub(u'\u2012', '-', html)  # figure dash -> '-'
    html = re.sub(u'\u2013', '-', html)  # en-dash -> '-'
    html = re.sub(r'mm<sup>2</sup>', 'mm2', html)  # mm^2 -> mm2
    html = re.sub(u'\u00d710<sup>6</sup>', u'\u00d7106', html)  # 10^6 -> 106
    html = re.sub(u'\u00d710<sup>9</sup>', u'\u00d7109', html)  # 10^9 -> 109
    html = re.sub(r'<sup>\d+</sup>', '', html)  # delete footnotes
    # with open('/tmp/%s.html' % vendor, 'w') as f:
    #     f.write(html.encode('utf8'))

    dfs = pd.read_html(html, match='Launch', parse_dates=True)
    for df in dfs:
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

        df['Vendor'] = vendor

        # make sure Launch is a string (dtype=object) before parsing it
        df['Launch'] = df['Launch'].apply(lambda x: str(x))
        df['Launch'] = df['Launch'].str.replace(referencesAtEnd, '')
        df['Launch'] = df['Launch'].apply(
            lambda x: pd.to_datetime(x,
                                     infer_datetime_format=True,
                                     errors='coerce'))

    data[vendor]['dfs'] = dfs

df = pd.concat(data['NVIDIA']['dfs'] + data['AMD']['dfs'], ignore_index=True)


def merge(df, dst, src, replaceNoWithNaN=False):
    if replaceNoWithNaN:
        df[src] = df[src].replace('No', np.nan)
    df[dst] = df[dst].fillna(df[src])
    df.drop(src, axis=1, inplace=True)
    return df

# merge related columns
df = merge(df, 'Model', 'Model Units')
df = merge(df, 'SM count', 'SMM count')
df = merge(df, 'SM count', 'SMX count')
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
df = merge(df, 'TDP (Watts)', 'TDP (Watts) GPU only')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) (GPU only)')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) Max.')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) W')
df = merge(df, 'TDP (Watts)', 'TBP (W)')
df = merge(df, 'Model', 'Model (Codename)')
df = merge(df, 'Model', 'Model: Mobility Radeon')
df = merge(df, 'Core clock (MHz)', 'Clock speeds Base core clock (MHz)')
df = merge(df, 'Core clock (MHz)', 'Core Clock (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speed Core (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock speed Average (MHz)')
df = merge(df, 'Core clock (MHz)', 'Clock rate Base (MHz)')
df = merge(df, 'Core clock (MHz)', 'Core Clock rate (MHz)')
df = merge(df, 'Core config', 'Core Config')
df = merge(df, 'Memory Bus type', 'Memory RAM type')
df = merge(df, 'Memory Bus type', 'Memory Type')
df = merge(df, 'Memory Bus type', 'Memory configuration DRAM type')

# filter out {Chips, Code name, Core config}: '^[2-9]\u00d7'
df = df[~df['Chips'].str.contains(ur'^[2-9]\u00d7', re.UNICODE, na=False)]
df = df[~df['Code name'].str.contains(ur'^[2-9]\u00d7', re.UNICODE, na=False)]
df = df[~df['Core config'].str.contains(
    ur'^[2-9]\u00d7', re.UNICODE, na=False)]
# filter out if Model ends in [xX]2
df = df[~df['Model'].str.contains('[xX]2$', na=False)]
# filter out {transistors, die size} that end in x2
df = df[~df['Transistors (million)'].str.contains(
    ur'\u00d7[2-9]$', re.UNICODE, na=False)]
df = df[~df['Die size (mm2)'].str.contains(
    ur'\u00d7[2-9]$', re.UNICODE, na=False)]

# merge GFLOPS columns with "Boost" column headers and rename
for prec in ['Double', 'Single', 'Half']:
    col = 'Processing power (GFLOPS) %s precision' % prec
    df = merge(df, col, 'Processing power (GFLOPS) %s precision (Boost)' % prec)
    df = merge(df, col, 'Processing power (GFLOPS) %s' % prec)

    # pick the first number we see as the actual number
    df[col] = df[col].astype(str)
    df[col] = df[col].str.extract(r'^([\d\.]+)', expand=False)

    # convert TFLOPS to GFLOPS
    tomerge = 'Processing power (TFLOPS) %s' % prec
    df[col] = df[col].fillna(
        pd.to_numeric(df[tomerge].str.split(' ').str[0], errors='coerce') * 1000.0)
    df.drop(tomerge, axis=1, inplace=True)

    df = df.rename(columns={col: '%s-precision GFLOPS' % prec})

# split out 'transistors die size'
# example: u'292\u00d7106 59 mm2'
for exponent in [u'\u00d7106', u'\u00d7109', 'B']:
    dftds = df['Transistors Die Size'].str.extract(u'^([\d\.]+)%s (\d+) mm2' % exponent,
                                                   expand=True)
    if exponent == u'\u00d7106':
        df['Transistors (million)'] = df['Transistors (million)'].fillna(
            pd.to_numeric(dftds[0], errors='coerce'))
    if exponent == u'\u00d7109' or exponent == 'B':
        df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
            pd.to_numeric(dftds[0], errors='coerce'))
    df['Die size (mm2)'] = df['Die size (mm2)'].fillna(
        pd.to_numeric(dftds[1], errors='coerce'))

# some AMD chips have core/boost in same entry, take first number
df['Core clock (MHz)'] = df['Core clock (MHz)'].str.split(' ').str[0]

# collate memory bus type and take first word only, removing chud as
# appropriate
df = merge(df, 'Memory Bus type', 'Memory Bus type & width (bit)')
df['Memory Bus type'] = df['Memory Bus type'].str.split(' ').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split(',').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split('/').str[0]
df['Memory Bus type'] = df['Memory Bus type'].str.split('[').str[0]
df.loc[df['Memory Bus type'] == 'EDO', 'Memory Bus type'] = 'EDO VRAM'

# merge transistor counts
df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
    pd.to_numeric(df['Transistors (million)'], errors='coerce') / 1000.0)

# extract shader (processor) counts
df['Pixel/unified shader count'] = df['Core config'].str.split(':').str[0]
df = merge(df, 'Pixel/unified shader count', 'Stream processors')
df = merge(df, 'Pixel/unified shader count', 'Shaders Cuda cores (total)')

# merge in AMD fab stats
df['Architecture (Fab) (extracted)'] = df[
    'Architecture (Fab)'].str.extract(r'\((\d+) nm\)', expand=False)
df = merge(df, 'Fab (nm)', 'Architecture (Fab) (extracted)')

for col in ['Memory Bandwidth (GB/s)', 'TDP (Watts)', 'Fab (nm)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# compute arithmetic intensity and FLOPS/Watt
df['Arithmetic intensity (FLOP/B)'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['Memory Bandwidth (GB/s)'], errors='coerce')
df['Single precision FLOPS/Watt'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['TDP (Watts)'], errors='coerce')

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
)
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
)

sm = Chart(df).mark_point().encode(
    x='Launch:T',
    y='SM count:Q',
    color=Color('Vendor',
                scale=colormap,
                ),
)
die = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Die size (mm2):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
)
xt = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Transistors (billion):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
    shape='GPU Type',
)
fab = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Fab (nm):Q',
        scale=Scale(type='log'),
        ),
    color=Color('Vendor',
                scale=colormap,
                ),
)
ai = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Arithmetic intensity (FLOP/B):Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
)

fpw = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Single precision FLOPS/Watt:Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
)

clk = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Core clock (MHz):Q',
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
)

# only plot chips with actual feature sizes
fpwsp = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Single-precision GFLOPS:Q',
        scale=Scale(type='log'),
        ),
    y='Single precision FLOPS/Watt:Q',
    shape='Vendor',
    color='Fab (nm):N',
)

fpwbw = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    y='Single precision FLOPS/Watt:Q',
    shape='Vendor',
    color='Fab (nm):N',
)

aisp = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Single-precision GFLOPS:Q',
        scale=Scale(type='log'),
        ),
    y='Arithmetic intensity (FLOP/B):Q',
    shape='Vendor',
    color='Fab (nm):N',
)

aibw = Chart(df[df['Fab (nm)'].notnull()]).mark_point().encode(
    x=X('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    y='Arithmetic intensity (FLOP/B):Q',
    shape='Vendor',
    color='Fab (nm):N',
)

sh = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Pixel/unified shader count:Q',
        scale=Scale(type='log'),
        ),
    shape='GPU Type',
    color=Color('Vendor',
                scale=colormap,
                ),
)

# df.to_csv("/tmp/gpu.csv", encoding="utf-8")

template = """<!DOCTYPE html>
<html>
<head>
  <!-- Import Vega 3 & Vega-Lite 2 js (does not have to be from cdn) -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega/3.0.2/vega.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-lite/2.0.0-beta.13/vega-lite.js"></script>
  <!-- Import vega-embed -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-embed/3.0.0-beta.20/vega-embed.js"></script>
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
  vega.embed('#vis', {spec}, opt);
</script>
</body>
</html>"""

readme = "# GPU Statistics\n\nData sourced from [Wikipedia's NVIDIA GPUs page](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units) and [Wikipedia's AMD GPUs page](https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units).\n\n"

outputdir = "/Users/jowens/Documents/working/owensgroup/proj/gpustats/plots"
for (chart, title) in [(bw, "Memory Bandwidth over Time"),
                       (pr, "Processing Power over Time"),
                       (sm, "SM count over Time"),
                       (sh, "Shader count over Time"),
                       (die, "Die Size over Time"),
                       (xt, "Transistor Count over Time"),
                       (fab, "Feature size over Time"),
                       (clk, "Clock rate over Time"),
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
        spec['height'] = 750
        spec['width'] = 1213
        spec['encoding']['tooltip'] = {"field": "Model", "type": "nominal"}
        f.write(template.format(spec=json.dumps(spec), title=title))
        # f.write(chart.from_json(spec_str).to_html(
        # title=title, template=template))
        save(chart, df=pd.DataFrame(), plotname=title, outputdir=outputdir,
             formats=['pdf'])
    readme += "- [%s](plots/%s.html)\n" % (title, title)

with open(os.path.join(outputdir, '../README.md'), 'w') as f:
    f.write(readme)
