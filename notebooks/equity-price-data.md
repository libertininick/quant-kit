---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python [conda env:neuralopt_env]
    language: python
    name: conda-env-neuralopt_env-py
---

```python
%load_ext autoreload
%autoreload 2

import glob
import json
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg as pg
from pandas import DataFrame, Series

from quant_kit_core import get_timediff

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

```python
engine = pg.connect(
    user='admin',
    password='quest',
    host='127.0.0.1',
    port='8812',
)
```

```python
st_dt = "2009-12-31"
end_dt = "2020-12-31"
index = "S&P 500"

query = f"""
SELECT * 
FROM index_membership
WHERE Index = '{index}'
AND (Date BETWEEN '{st_dt}' AND '{end_dt}') 
"""

members = pd.pivot_table(
    data=pd.read_sql(query, con=engine),
    index="Date",
    columns="Symbol",
    aggfunc="count",
    fill_value=0,
)

members.columns = members.columns.droplevel(level=0)

members.shape
```

```python
symbols = ",".join([f"'{sym}'" for sym in members.columns.values])
```

```python
query = f"""
SELECT * 
FROM equity_prices
WHERE Symbol in ({symbols})
AND (Date BETWEEN '{st_dt}' AND '{end_dt}') 
"""

prices = pd.read_sql(query, con=engine)
```

```python
def true_range(window: DataFrame, log: bool = False) -> float:
    """
    """
    if window.shape[0] == 2:
        c0 = window.Close.iloc[0]
        h1 = window.High.iloc[1]
        l1 = window.Low.iloc[1]
        c1 = window.Close.iloc[1]

        tr = max(c0, h1, c1) / min(c0, l1, c1)

        if log:
            tr = np.log(tr)
        else:
            tr -= 1
    else:
        tr = np.nan
    
    return tr
```

```python
dfs = []
for (sym, df) in prices.groupby("Symbol"):
    df["true_range"] = [true_range(window) for window in df.rolling(window=2, axis=0)]
    df["med_tr"] = df["true_range"].rolling(20).median()
    df = df.loc[:, ["Date", "med_tr"]]
    df = df.set_index("Date")
    df.columns = [sym]
    dfs.append(df)

df_merged = reduce(
    lambda left, right: pd.merge_ordered(left, right, on="Date", fill_method="ffill"),
    dfs
)
df_merged = df_merged.set_index("Date")
df_merged = df_merged.loc[:, members.columns]
```

```python
avg_vol = (np.where(np.isnan(df_merged.values), 0, df_merged.values) * members.values).sum(-1)/members.values.sum(-1)
```

```python
fig, ax = plt.subplots(figsize=(15,7))
ax.scatter(x=df_merged.index, y=avg_vol)
```

```python

```
