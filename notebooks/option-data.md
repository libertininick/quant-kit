---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

```python
%load_ext autoreload
%autoreload 2

import glob
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg as pg
from pandas import DataFrame


from quant_kit_core.utils import get_timediff

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

```python
files = glob.glob("/storage/hd1/datasets/options/*.csv")
len(files)
```

```python
def load_data(filename: str) -> DataFrame:
    df = (
        pd.read_csv(filename)
        .rename(
            {
                'UnderlyingSymbol': 'Symbol', 
                'DataDate': 'Date', 
                'UnderlyingPrice': 'Spot', 
                'HmHR': 'HR'
            }, 
            axis='columns'
        )
        .loc[:, 
            [
                "Symbol",
                "Date",
                "Expiration",
                "Strike",
                "Type",
                "Spot",
                "Last",
                "Bid",
                "Ask",
                "Volume",
                "OpenInterest"
            ]
        ]
    )

    df.Date = pd.to_datetime(df.Date)
    df.Expiration = pd.to_datetime(df.Expiration)
    df.Type = df.Type.apply(lambda x: "C" if x == "call" else "P")
    df["moneyness"] = df["Strike"] / df["Spot"]
    df["t"] = df.loc[:, ["Date", "Expiration"]].apply(lambda r: get_timediff(*r), axis=1)
    
    return df
```

```python
schema = [
    {"name": "Date", "type": "date", "pattern": "yyyy-MM-dd"},
    {"name": "Symbol", "type": "symbol"}, 
    {"name": "Index", "type": "symbol"}, 
]
with open("/storage/hd1/datasets/options/options_schema.json", "w") as fp:
    json.dump(schema, fp)
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
query = """
SELECT * 
FROM indexes
WHERE Date BETWEEN '2019-12-31' and '2020-02-01'
"""

indexes = pd.read_sql(query, con=engine)
```
