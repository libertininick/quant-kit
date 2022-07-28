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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame
from torch import Tensor

from quant_kit_core.utils import get_timediff

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

# Load data


## Scrape

```python
url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
data = pd.concat(
    [
        pd.read_csv(
            f"{url}/{yr}/all?type=daily_treasury_yield_curve&field_tdr_date_value={yr}&page&_format=csv"
        )
        for yr in range(1990, 2022 + 1)
    ],
    axis=0
)

data.to_csv("../data/daily-treasury-rates.csv", index=False)
```

## Load

```python
data = pd.read_csv("../data/daily-treasury-rates.csv")
cols = [
    'Date', 
    '1 Mo', 
    '2 Mo', 
    '3 Mo', 
    '6 Mo', 
    '1 Yr', 
    '2 Yr', 
    '3 Yr', 
    '5 Yr', 
    '7 Yr', 
    '10 Yr',
    '20 Yr',
    '30 Yr',
]
t = np.array([1,2,3,6,12,24,36,60,84,120,240,360])/120
col_to_maturity_map = {k: v for (k,v) in zip(cols[1:], t)}
data.Date = pd.to_datetime(data.Date, format='%m/%d/%Y')
data = data[cols].set_index("Date", drop=True)
data.info()
```

```python
d = data.iloc[-360:]
d = np.exp(d / 100)
d = d / d.loc[:, ["3 Mo"]].values
d = pd.melt(d, ignore_index=False).sort_index()
d["quote_delta"] = (d.index - d.index[-1]).days
d["term"] = d.variable.apply(col_to_maturity_map.get)
d = d[~d.isna().any(axis=1)]
```

```python
x = torch.tensor(d.loc[:, ["quote_delta", "term"]].values, dtype=torch.float32)
y = torch.tensor(d.value.values, dtype=torch.float32)
```

```python
n = len(x)
n_in = int(n*0.67)
n_out = n - n_in
idx_in = np.random.choice(np.arange(n), size=n_in, replace=False)
idx_out = np.setdiff1d(np.arange(n), idx_in)
len(idx_in), len(idx_out)
```

```python
x_in, x_out = x[idx_in], x[idx_out]
y_in, y_out = y[idx_in, None], y[idx_out, None]
```

```python
from torch.nn import MSELoss
from torch.optim import Adam
```

```python

```

```python
model = nn.Sequential(
    nn.BatchNorm1d(num_features=2),
    nn.Linear(2, 32),
    nn.GELU(),
    nn.Linear(32, 1),
    nn.ReLU(),
)

optimizer = Adam(params=model.parameters())

n_iter = 10000
loss_fxn = MSELoss()
errs = []

for i in range(n_iter):
    model.train()
    yh = model(x_in)
    err = loss_fxn(yh, y_in)
    err.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i >= 200:
        model.eval()
        with torch.no_grad():
            yh = model(x_out)
            errs.append(loss_fxn(yh, y_out).item())
        
        if (i + 1) % 100 == 0:
            e_a = np.mean(errs[-100:])
            e_b = np.mean(errs[-200:-100])
            e_r = e_a / e_b
            print(e_r)
            if e_r  > 0.95:
                break
```

```python
np.mean(errs[-100:]) / np.mean(errs[-200:-100])
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.plot(errs)
```

```python
errs_out
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(x[:, 9], bins=30)
```

```python
fig, ax = plt.subplots(figsize=(20,10))
for x_i in x[-360:]:
    ax.plot(t, x_i, "-", alpha=0.1, color="k")
```

```python

```
