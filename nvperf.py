#!/usr/bin/env python

import pandas as pd
import numpy as np
import requests
import re
import os
import json

from altair import *

data = {
    'NVIDIA': {'url': 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units',
               },
    'AMD': {'url': 'https://en.wikipedia.org/wiki/List_of_AMD_graphics_processing_units',
            }
}

for vendor in ['NVIDIA', 'AMD']:
    # requests.get handles https
    html = requests.get(data[vendor]['url']).text
    # oddly, some dates look like:
    # <td><span class="sortkey" style="display:none;speak:none">000000002010-02-25-0000</span><span style="white-space:nowrap">Feb 25, 2010</span></td>
    html = re.sub(
        r'<span [^>]*style="display:none[^>]*>([^<]+)</span>', '', html)
    html = re.sub(r'<span[^>]*>([^<]+)</span>', r'\1', html)
    with open('/tmp/%s.html' % vendor, 'w') as f:
        f.write(html.encode('utf8'))

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
        df['Vendor'] = vendor
        df['Launch'] = df['Launch'].apply(
            lambda x: pd.to_datetime(x,
                                     infer_datetime_format=True,
                                     errors='coerce'))

    data[vendor]['dfs'] = dfs

df = pd.concat(data['NVIDIA']['dfs'] + data['AMD']['dfs'], ignore_index=True)


def merge(df, dst, src):
    df[dst] = df[dst].fillna(df[src])
    df.drop(src, axis=1, inplace=True)
    return df

# merge related columns
df = merge(df, 'SM count', 'SMM count')
df = merge(df, 'SM count', 'SMX count')
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS)')
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS) Single')
df = merge(df, 'Processing power (GFLOPS) Double precision',
           'Processing power (GFLOPS) Double')
df = merge(df, 'Memory Bandwidth (GB/s)',
           'Memory configuration Bandwidth (GB/s)')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) GPU only')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) (GPU only)')
df = merge(df, 'TDP (Watts)', 'TDP (Watts) Max.')
df = merge(df, 'Model', 'Model (Codename)')
df = merge(df, 'Model', 'Model: Mobility Radeon')

# merge GFLOPS columns with "Boost" column headers and rename
for prec in ['Double', 'Single', 'Half']:
    tomerge = 'Processing power (GFLOPS) %s precision (Boost)' % prec
    df['Processing power (GFLOPS) %s precision' % prec] = df['Processing power (GFLOPS) %s precision' % prec].fillna(
        df[tomerge].str.split(' ').str[0])
    df.drop(tomerge, axis=1, inplace=True)

    tomerge = 'Processing power (GFLOPS) %s (Boost)' % prec
    df['Processing power (GFLOPS) %s precision' % prec] = df['Processing power (GFLOPS) %s precision' % prec].fillna(
        df[tomerge].str.split(' ').str[0])
    df.drop(tomerge, axis=1, inplace=True)

    if prec == 'Single':
        tomerge = 'Processing power (GFLOPS) (Boost)'
        df['Processing power (GFLOPS) %s precision' % prec] = df['Processing power (GFLOPS) %s precision' % prec].fillna(
            df[tomerge].str.split(' ').str[0])
        df.drop(tomerge, axis=1, inplace=True)

    tomerge = 'Processing power (TFLOPS) %s (Boost)' % prec
    df['Processing power (GFLOPS) %s precision' % prec] = df['Processing power (GFLOPS) %s precision' % prec].fillna(
        pd.to_numeric(df[tomerge].str.split(' ').str[0], errors='coerce') * 1000.0)
    df.drop(tomerge, axis=1, inplace=True)

    df = df.rename(columns={'Processing power (GFLOPS) %s precision' % prec:
                            '%s-precision GFLOPS' % prec})

# merge transistor counts
df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
    pd.to_numeric(df['Transistors (million)'], errors='coerce') / 1000.0)

# extract shader (processor) counts
df['Pixel/unified shader count'] = df['Core config'].str.split(':').str[0]
df = merge(df, 'Pixel/unified shader count', 'Stream processors')

for col in ['Memory Bandwidth (GB/s)', 'TDP (Watts)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# compute arithmetic intensity and FLOPS/Watt
df['Arithmetic intensity (FLOP/B)'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['Memory Bandwidth (GB/s)'], errors='coerce')
df['FLOPS/Watt'] = pd.to_numeric(df[
    'Single-precision GFLOPS'], errors='coerce') / pd.to_numeric(df['TDP (Watts)'], errors='coerce')

# remove references from end of model names
df['Model'] = df['Model'].str.replace(r'(?:\s*\[\d+\])+(?:\d+,)?(?:\d+)?$', '')

# mark mobile processors
df['GPU Type'] = np.where(
    df['Model'].str.contains(r' [\d]+M[X]?|\(Notebook\)'), 'Mobile', 'Desktop')

mb = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        ),
    color='GPU Type',
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
    color='Datatype',
    shape='Vendor',
)

sm = Chart(df).mark_point().encode(
    x='Launch:T',
    y='SM count:Q',
    shape='Vendor',
)
die = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Die size (mm2):Q',
        scale=Scale(type='log'),
        ),
    color='Vendor',
)
xt = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Transistors (billion):Q',
        scale=Scale(type='log'),
        ),
    color='Vendor',
)
fab = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Fab (nm):Q',
        scale=Scale(type='log'),
        ),
    color='Vendor',
)
ai = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Arithmetic intensity (FLOP/B):Q',
    color='GPU Type',
    shape='Vendor',
)

fpw = Chart(df).mark_point().encode(
    x='Launch:T',
    y='FLOPS/Watt:Q',
    color='GPU Type',
    shape='Vendor',
)

sh = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Pixel/unified shader count:Q',
        scale=Scale(type='log'),
        ),
    color='GPU Type',
    shape='Vendor',
)

df.to_csv("/tmp/gpu.csv", encoding="utf-8")

template = """<!DOCTYPE html>
<html>
<head>
  <!-- Import Vega 3 & Vega-Lite 2 js (does not have to be from cdn) -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega/3.0.0-rc4/vega.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-lite/2.0.0-beta.10/vega-lite.js"></script>
  <!-- Import vega-embed -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/vega-embed/3.0.0-beta.19/vega-embed.js"></script>
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

readme = "# GPU Statistics\n\nData sourced from [Wikipedia's NVIDIA GPUs page](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units).\n\n"

outputdir = "/Users/jowens/Documents/working/owensgroup/proj/gpustats/plots"
for (chart, title) in [(mb, "Memory Bandwidth over Time"),
                       (pr, "Processing Power over Time"),
                       (sm, "SM count over Time"),
                       (die, "Die Size over Time"),
                       (xt, "Transistor Count over Time",),
                       (fab, "Feature size over Time"),
                       (ai, "Arithmetic Intensity over Time"),
                       (fpw, "FLOPS per Watt over Time"),
                       (sh, "Shader count over Time")]:
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
    readme += "- [%s](plots/%s.html)\n" % (title, title)

with open(os.path.join(outputdir, '../README.md'), 'w') as f:
    f.write(readme)
