#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import requests
import re
import os
import json

import altair as alt

from fileops import save

products_url = "https://productapi.intel.com/api/products/get-products"
products_info_url = "https://productapi.intel.com/api/products/get-products-info"
products_codename_url = "https://productapi.intel.com/api/products/get-codename"

with open("intel_credentials.json") as intel_credentials_file:
    intel_credentials = json.load(intel_credentials_file)
intel_auth = (intel_credentials["username"], intel_credentials["password"])

products_params = {
    "client_id": intel_credentials["client_id"],
    "category_id": '["873"]',
    "locale_geo_id": "en-US",
    "per_page": 10000,
}
r = requests.get(products_url, auth=intel_auth, params=products_params)
content = json.loads(r.content)
df_product_ids = pd.json_normalize(content["result"])
num_products = content["total_count"]
product_ids = df_product_ids["product_id"].tolist()

# now product_ids is a list of the product IDs we need to fetch.

dfs = []
explanations = {}
product_id_start = 0
product_id_end = 40
products_info_params = {
    "client_id": intel_credentials["client_id"],
    "product_id": "",  # this will get replaced on each call
    # "product_id": '["120475"]',
    "locale_geo_id": "en-US",
}
while True:
    products_info_params["product_id"] = (
        "["
        + ",".join(
            [('"' + str(i) + '"') for i in product_ids[product_id_start:product_id_end]]
        )
        + "]"
    )

    # print(products_info_params["product_id"])

    r = requests.get(products_info_url, auth=intel_auth, params=products_info_params)
    products_info_json = json.loads(r.content)["result"]
    # this is a list of dicts. tech_spec is also a list of one dict
    # convert "tech_spec" to a plain dict by taking [0]
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
            products_info_json[idx][d["highlight_key"]] = d["raw_value"]
    # then normalize it so it's flat
    df = pd.json_normalize(products_info_json)
    dfs.append(df)

    if product_id_end >= num_products:
        break
    product_id_start += 40
    product_id_end = min(product_id_end + 40, num_products)

df = pd.concat(dfs, ignore_index=True)
for datecol in ["product_on_market_date", "created_date", "updated_date"]:
    # this is kludgey -- I couldn't make to_datetime give me back a date
    # so I'm manually cutting off the parts after T and that works
    df[datecol] = df[datecol].str.extract(r"^([^T]*)", expand=False)
    # df[datecol] = pd.to_datetime(df[datecol], format="%Y-%m-%dT%H:%M:%S:%f%z")
    df[datecol] = pd.to_datetime(df[datecol], format="%Y-%m-%d")
# df.drop(columns=["created_date", "updated_date"])
columnsOfInterest = ["product_on_market_date", "ThreadCount"]
df = df[columnsOfInterest]

# df.to_csv("intel.csv")
# print(explanations)

alt.Chart(df).mark_point().encode(x="product_on_market_date", y="ThreadCount:Q")

# products_codename_params = {
#     "client_id": intel_credentials["client_id"],
#     "locale_geo_id": "en-US",
# }

# r = requests.get(products_codename_url, auth=intel_auth, params=products_codename_params)
# print(r.content)
