#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import requests
import re
import os
import json
from html_template import html_template

import altair as alt

from fileops import save

# https://productapi.intel.com/intel-product-catalogue-service/reference

products_url = "https://productapi.intel.com/api/products/get-products"
products_info_url = "https://productapi.intel.com/api/products/get-products-info"
products_codename_url = "https://productapi.intel.com/api/products/get-codename"

with open("intel_credentials.json") as intel_credentials_file:
    intel_credentials = json.load(intel_credentials_file)
intel_auth = (intel_credentials["username"], intel_credentials["password"])

# The goal of querying "products" is a list (and count) of product IDs
products_params = {
    "client_id": intel_credentials["client_id"],
    "category_id": '["873"]',  # processors, per https://productapi.intel.com/intel-product-catalogue-service/reference/product-details/get-product-list
    "locale_geo_id": "en-US",
    "per_page": 10000,
}
r = requests.get(products_url, auth=intel_auth, params=products_params)
content = json.loads(r.content)
df_product_ids = pd.json_normalize(content["result"])
num_products = content["total_count"]
product_ids = df_product_ids["product_id"].tolist()
# now product_ids is a list of the product IDs we need to fetch

dfs = []
explanations = {}  # column names -> what they mean
product_id_start = 0
product_id_end = 40
products_info_params = {
    "client_id": intel_credentials["client_id"],
    "product_id": "",  # this will get replaced on each call with a list <= 40
    # "product_id": '["120475"]',
    "locale_geo_id": "en-US",
}
while True:
    # format the list like Intel wants it
    products_info_params["product_id"] = (
        "["
        + ",".join(
            [('"' + str(i) + '"') for i in product_ids[product_id_start:product_id_end]]
        )
        + "]"
    )

    r = requests.get(products_info_url, auth=intel_auth, params=products_info_params)
    products_info_json = json.loads(r.content)["result"]
    # this is a list of dicts.
    # within each list, tech_spec is another list of dicts
    # explode its contents
    for idx, item in enumerate(products_info_json):
        for d in products_info_json[idx]["tech_spec"]:
            if d["raw_value"] == "TRUE":
                d["raw_value"] = True
            if d["raw_value"] == "FALSE":
                d["raw_value"] = False
            if d["raw_value"] == "Yes":
                d["raw_value"] = True
            if d["raw_value"] == "No":
                d["raw_value"] = False
            if d["highlight_key"] not in explanations.keys():
                explanations[d["highlight_key"]] = d["label"]
            # (key, value) is (highlight_key, raw_value)
            products_info_json[idx][d["highlight_key"]] = d["raw_value"]
    # then normalize it so it's flat
    df = pd.json_normalize(products_info_json)
    dfs.append(df)

    if product_id_end >= num_products:
        break
    product_id_start += 40
    product_id_end = min(product_id_end + 40, num_products)

# dfs is a list of dataframes, each of which has <= 40 products
df = pd.concat(dfs, ignore_index=True)

# fix up the dates
for datecol in ["product_on_market_date", "created_date", "updated_date"]:
    # this is kludgey -- I couldn't make to_datetime give me back a
    # date with the "actual" format so I'm manually cutting off the
    # parts after T and that works
    df[datecol] = df[datecol].str.extract(r"^([^T]*)", expand=False)
    # df[datecol] = pd.to_datetime(df[datecol], format="%Y-%m-%dT%H:%M:%S:%f%z")
    df[datecol] = pd.to_datetime(df[datecol], format="%Y-%m-%d")

datatypes = {
    "product_on_market_date": "temporal",
    "CoreCount": "quantitative",
    "Lithography": "quantitative",
    "MaxTDP": "quantitative",
    "ClockSpeed": "quantitative",
    "Cache": "quantitative",
    "product_manufacturer": "nominal",
}

script_dir = os.path.dirname(os.path.realpath("__file__"))
rel_path = "plots"
outputdir = os.path.join(script_dir, rel_path)

chart = {}

my = {
    "Cores over Time (Intel)": {
        "mark": "point",
        "x": ("product_on_market_date", "Date", "time"),
        "y": ("CoreCount", "Cores", "linear"),
        "shape": ("product_manufacturer", "Vendor"),
        "color": ("product_manufacturer", "Vendor"),
    },
    "Feature size over Time (Intel)": {
        "mark": "point",
        "x": ("product_on_market_date", "Date", "time"),
        "y": ("Lithography", "Fab (nm)", "linear"),
        "shape": ("product_manufacturer", "Vendor"),
        "color": ("product_manufacturer", "Vendor"),
    },
    "Power over Time (Intel)": {
        "mark": "point",
        "x": ("product_on_market_date", "Date", "time"),
        "y": ("MaxTDP", "TDP (Watts)", "linear"),
        "shape": ("product_manufacturer", "Vendor"),
        "color": ("product_manufacturer", "Vendor"),
    },
    "Clock rate over Time (Intel)": {
        "mark": "point",
        "x": ("product_on_market_date", "Date", "time"),
        "y": ("ClockSpeed", "Core clock (MHz)", "linear"),
        "shape": ("product_manufacturer", "Vendor"),
        "color": ("product_manufacturer", "Vendor"),
    },
    "Cache size over Time (Intel)": {
        "mark": "point",
        "x": ("product_on_market_date", "Date", "time"),
        "y": ("Cache", "Cache Size (MB)", "linear"),
        "shape": ("product_manufacturer", "Vendor"),
        "color": ("product_manufacturer", "Vendor"),
    },
}

vendor_colormap = alt.Scale(
    # domain=["AMD", "NVIDIA", "Intel"], range=["#ff0000", "#76b900", "#0071c5"]
    domain=["Intel"],
    range=["#0071c5"],
)


for plot in my.keys():
    print(f"*** Processing {plot} ***")

    if "filter" in my[plot].keys():
        dfx = my[plot]["filter"](df)
    else:
        dfx = df

    my_selection = alt.selection_multi(fields=[my[plot]["color"][0]])
    my_color = alt.condition(
        my_selection,
        alt.Color(
            my[plot]["color"][0],
            type=datatypes[my[plot]["color"][0]],
            legend=alt.Legend(title=my[plot]["color"][1]),
            scale=vendor_colormap,
        ),
        alt.value("lightgray"),
    )

    chart[plot] = (
        alt.Chart(dfx, mark=my[plot]["mark"])
        .encode(
            x=alt.X(
                my[plot]["x"][0],
                type=datatypes[my[plot]["x"][0]],
                axis=alt.Axis(title=my[plot]["x"][1],),
                scale=alt.Scale(type=my[plot]["x"][2]),
            ),
            y=alt.Y(
                my[plot]["y"][0],
                type=datatypes[my[plot]["y"][0]],
                axis=alt.Axis(title=my[plot]["y"][1],),
                scale=alt.Scale(type=my[plot]["y"][2]),
            ),
            tooltip=[f"{my[plot]['x'][0]}", f"{my[plot]['y'][0]}", "product_name"],
            color=my_color,
        )
        .properties(width=1213, height=750)
        .interactive()
        .add_selection(my_selection)
    )

    if "shape" in my[plot]:
        shape = my[plot]["shape"][0]
        chart[plot] = chart[plot].encode(
            shape=alt.Shape(
                shape,
                type=datatypes[shape],
                legend=alt.Legend(title=my[plot]["shape"][1]),
            )
        )
    if "column" in my[plot]:
        chart[plot] = chart[plot].encode(
            column=alt.Column(
                my[plot]["column"][0],
                type=datatypes[my[plot]["column"][0]],
                header=alt.Header(title=my[plot]["column"][1]),
            )
        )
    if "row" in my[plot]:
        chart[plot] = chart[plot].encode(
            row=alt.Row(
                my[plot]["row"][0],
                type=datatypes[my[plot]["row"][0]],
                header=alt.Header(title=my[plot]["row"][1]),
            )
        )

    with open(os.path.join(outputdir, plot + ".html"), "w") as f:
        spec = chart[plot].to_dict()
        f.write(html_template.format(spec=json.dumps(spec), title=plot))
        save(
            chart[plot],
            df=pd.DataFrame(),
            plotname=plot,
            outputdir=outputdir,
            formats=["pdf"],
        )


####

# if I wanted to get codenames, here's how I'd do it
# products_codename_params = {
#     "client_id": intel_credentials["client_id"],
#     "locale_geo_id": "en-US",
# }
# r = requests.get(products_codename_url, auth=intel_auth, params=products_codename_params)
# print(r.content)


# helpful links on FLOPS/core/cycle
# https://en.wikichip.org/wiki/flops
# https://software.intel.com/en-us/forums/software-tuning-performance-optimization-platform-monitoring/topic/761046
