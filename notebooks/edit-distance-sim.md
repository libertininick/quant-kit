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
from scipy.signal import find_peaks, peak_prominences

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

def nomralize(arr: NDArray[np.float_]) -> NDArray[np.float_]:
    return (arr - arr.mean()) / np.mean(arr**2)**0.5


def get_peak_idxs(
    arr: NDArray[np.float_],
    distance: int | None = None,
    prominence: float | None = None,
) -> NDArray[np.int_]:
    
    p_peak_idxs, _ = find_peaks(
        arr,
        distance=distance,
        prominence=prominence,
    )
    n_peak_idxs, _ = find_peaks(
        -arr,
        distance=distance,
        prominence=prominence,
    )
    peak_idxs = np.sort(np.concatenate((p_peak_idxs, n_peak_idxs)))
    
    return peak_idxs
    
    
def calc_min_edit_distance(
    returns_a: NDArray,
    returns_b: NDArray,
    distance: int = 5,
    prominence: float = 2,
    transport_threshold = 0.1
) -> float:
    """
    """
    n1 = len(returns_a)
    n2 = len(returns_b)
    
    scaled_a = returns_a / np.mean(returns_a**2)**0.5
    scaled_b = returns_b / np.mean(returns_b**2)**0.5
    
    peak_idxs_a = get_peak_idxs(
        (scaled_a - scaled_a.mean()).cumsum(), distance, prominence
    )
    peak_idxs_b = get_peak_idxs(
        (scaled_b - scaled_b.mean()).cumsum(), distance, prominence
    )
    
    seq_a = scaled_a.cumsum()
    seq_b = scaled_b.cumsum()
    value_cost = (seq_a[peak_idxs_a, None] - seq_b[None, peak_idxs_b])**2
    
    transport_cost = ((peak_idxs_a[:, None] / n1 - peak_idxs_b[None, :] / n2) / transport_threshold)**2
    
    total_cost = value_cost + transport_cost
    
    # Minimum cost
    row_idxs, col_idxs = linear_sum_assignment(total_cost)
    min_cost = total_cost[row_idxs, col_idxs].mean()
    
    return min_cost / total_cost.mean()
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

# Rolling returns

```python
rolling_returns = get_rolling_windows(
    returns,
    window_size=252*5
)
rolling_returns.shape
```

```python
s1 = np.random.randint(low=0, high=23474)
s2 = np.random.randint(low=0, high=23474)

r1 = rolling_returns[s1]
r2 = rolling_returns[s2]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(r1.cumsum())
ax.plot(r2.cumsum())
```

```python
calc_min_edit_distance(r1, r2, distance=5, prominence=2)
```

```python
x = nomralize(r2)
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(x.cumsum())
```

```python
seq_a = nomralize(r1).cumsum()
seq_b = nomralize(r2).cumsum()
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(seq_a.cumsum())
ax.plot(seq_b.cumsum())
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
