#!/usr/bin/env python

import pandas as pd
import requests
import re
import os
import json

from altair import *
url = 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
f = requests.get(url).text      # handles https
dfs = pd.read_html(f, match='Launch', parse_dates=True)
for df in dfs:
    # Multi-index to index
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    # Get rid of 'Unnamed' column names
    df.columns = [re.sub(' Unnamed: [0-9]+_level_[0-9]+', '', col)
                  for col in df.columns.values]
    # If a word ends in a number or number,comma,number, delete it
    df.columns = [' '.join([re.sub('[\d,]+$', '', word) for word in col.split()])
                  for col in df.columns.values]

df = pd.concat(dfs, ignore_index=True)


def merge(df, dst, src):
    df[dst] = df[dst].fillna(df[src])
    df.drop(src, axis=1, inplace=True)
    return df

# merge related columns
df = merge(df, 'SM count', 'SMM count')
df = merge(df, 'SM count', 'SMX count')
df = merge(df, 'Processing power (GFLOPS) Single precision',
           'Processing power (GFLOPS)')

# merge GFLOPS columns with "Boost" column headers
for prec in ['Double', 'Single', 'Half']:
    df['Processing power (GFLOPS) %s precision' % prec] = df['Processing power (GFLOPS) %s precision' % prec].fillna(
        df['Processing power (GFLOPS) %s precision (Boost)' % prec].str.split(' ').str[0])

# merge transistor counts
df['Transistors (billion)'] = df['Transistors (billion)'].fillna(
    pd.to_numeric(df['Transistors (million)'], errors='coerce') / 1000.0)

# extract shader (processor) counts
df['Pixel/unified shader count'] = df['Core config'].str.split(':').str[0]

# compute arithmetic intensity and FLOPS/Watt
df['Arithmetic intensity (FLOP/B)'] = pd.to_numeric(df[
    'Processing power (GFLOPS) Single precision'], errors='coerce') / pd.to_numeric(df['Memory Bandwidth (GB/s)'], errors='coerce')
df['FLOPS/Watt'] = pd.to_numeric(df[
    'Processing power (GFLOPS) Single precision'], errors='coerce') / pd.to_numeric(df['TDP (watts)'], errors='coerce')

df['Launch'] = df['Launch'].apply(
    lambda x: pd.to_datetime(x,
                             infer_datetime_format=True,
                             errors='coerce'))

mb = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Memory Bandwidth (GB/s):Q',
        scale=Scale(type='log'),
        )
)
pr = Chart(pd.melt(df,
                   id_vars=['Launch'],
                   value_vars=['Processing power (GFLOPS) Single precision',
                               'Processing power (GFLOPS) Double precision',
                               'Processing power (GFLOPS) Half precision'],
                   var_name='Datatype',
                   value_name='Processing power (GFLOPS)')).mark_point().encode(
    x='Launch:T',
    y=Y('Processing power (GFLOPS):Q',
        scale=Scale(type='log'),
        ),
    color='Datatype',
)

sm = Chart(df).mark_point().encode(
    x='Launch:T',
    y='SM count:Q',
)
die = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Die size (mm2):Q',
        scale=Scale(type='log'),
        )
)
xt = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Transistors (billion):Q',
        scale=Scale(type='log'),
        )
)
fab = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Fab (nm):Q',
        scale=Scale(type='log'),
        )
)
ai = Chart(df).mark_point().encode(
    x='Launch:T',
    y='Arithmetic intensity (FLOP/B):Q',
)

fpw = Chart(df).mark_point().encode(
    x='Launch:T',
    y='FLOPS/Watt:Q',
)

sh = Chart(df).mark_point().encode(
    x='Launch:T',
    y=Y('Pixel/unified shader count:Q',
        scale=Scale(type='log'),
        )
)

# df.to_csv("/tmp/nv.csv", encoding="utf-8")

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
  var spec = "https://owensgroup.github.io/gpustats/plots/{title}.json";
  var opt = {{
    "mode": "vega-lite",
  }};
  vega.embed('#vis', spec, opt);
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
    with open(os.path.join(outputdir, title + '.json'), 'w') as f:
        d = chart.to_dict()
        d['height'] = 750
        d['width'] = 1213
        d['encoding']['tooltip'] = {"field": "Model", "type": "nominal"}
        j = json.dump(d, f)
    with open(os.path.join(outputdir, title + '.html'), 'w') as f:
        f.write(chart.to_html(title=title, template=template))
    readme += "- [%s](plots/%s.html)\n" % (title, title)

with open(os.path.join(outputdir, '../README.md'), 'w') as f:
    f.write(readme)
