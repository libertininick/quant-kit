---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python [conda env:quant_kit_env]
    language: python
    name: conda-env-quant_kit_env-py
---

```python
%load_ext autoreload
%autoreload 2

import glob
import json
from collections import defaultdict


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from quant_kit_core.time_series import get_rolling_windows

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

# Minimum "edit" distance

```python
n = 63
seq1 = np.random.randint(low=-10, high=11, size=n)
seq2 = np.random.randint(low=-10, high=11, size=n)
```

```python
from numpy.typing import NDArray
def calc_min_edit_distance(arr1: NDArray, arr2: NDArray, sample_frac: float = 1) -> float:
    """
    """
    n = len(arr1)
    idxs = np.arange(n)
    if sample_frac < 1:
        idxs = np.random.choice(idxs, int(n*sample_frac), replace=False)
    
    value_cost = (arr1[idxs, None] - arr2[None, idxs])**2
    
    
    transport_cost = np.abs(idxs[:, None] - idxs[None, :])**0.5
    
    total_cost = value_cost + transport_cost
    
    # Minimum cost
    row_idxs, col_idxs = linear_sum_assignment(total_cost)
    min_cost = total_cost[row_idxs, col_idxs].sum()
    
    return (min_cost / sample_frac) / n
```

# Load Data

```python
data = pd.read_csv("../data/world-indices/$SPX.csv")
data
```

```python
prices = data.Close.values
returns = np.log(prices[1:] / prices[:-1])
```

```python
norm_returns = (returns - returns.mean()) / np.mean(returns**2)**0.5
```

# Rolling returns

```python
rolling_returns = get_rolling_windows(
    norm_returns,
    window_size=252*5
)
rolling_returns.shape
```

```python
s1 = np.random.randint(low=0, high=23474)
s2 = np.random.randint(low=0, high=23474)

seq1 = rolling_returns[s1]
seq2 = rolling_returns[s2]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(seq1.cumsum())
ax.plot(seq2.cumsum())
```

```python
calc_min_edit_distance(seq1, seq2, sample_frac=1)
```

```python
x = [
    calc_min_edit_distance(seq1, seq2, sample_frac=0.05)
    for _ in range(1000)
]
np.quantile(x, q=[0, 0.25, 0.5, 0.75, 1])
```

```python
252*5*0.05
```

# Sample represenative set

```python
size = 32

n = len(rolling_returns)

dists = np.zeros((size,n))

seed = np.random.randint(low=0, high=n)

seq_i = rolling_returns[seed]

for i in range(size):
    for j in range(n):
        dists[i,j] = calc_min_edit_distance(seq_i, rolling_returns[j])
    break
```

```python
%%timeit
calc_min_edit_distance(seq_i, rolling_returns[j])
```

```python

```

```python
calc_min_edit_distance(seq_i, rolling_returns[j], 0.01)
```

```python
np.quantile(x, q=[0, 0.25, 0.5, 0.75, 1])
```

```python

```
