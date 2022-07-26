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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_kit_core import get_rolling_windows

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

# Load data

```python
df = pd.read_csv("../data/world-indices/$SPX.csv")
```

```python
df
```

# Rolling returns and vol

```python
prices = df.Close.values
log_ret = np.log(prices[1:]/prices[:-1])
```

```python
window_size = 250
windows = get_rolling_windows(log_ret, window_size)
dts = df.Date.values[np.arange(window_size, len(df))]
ret_avgs, ret_vols = windows.mean(-1), np.median(np.abs(windows), axis=-1)
```

```python
fig, axs = plt.subplots(nrows=2, figsize=(15,10))
ticks = np.arange(0, len(ret_avgs), 250*5)

axs[0].plot(ret_avgs)
axs[0].set_xticks(ticks)
axs[0].set_xticklabels([dt[:4] for dt in dts[ticks]])
axs[0].set_title("Rolling 1-yr avg return")

axs[1].plot(ret_vols)
axs[1].set_xticks(ticks)
axs[1].set_xticklabels([dt[:4] for dt in dts[ticks]])
axs[1].set_title("Rolling 1-yr avg vol")
```

# Environment segmentation

```python
vol_threshold = np.median(ret_vols)
env_flags = np.where(ret_vols <= vol_threshold, 1, 0)
states = np.unique(env_flags)

for state in states:
    s = ret_avgs[env_flags == state] * 250
    print(f"{state}: {s.mean():>10.4f} {s.std():>10.4f}")
```

```python
fig, axs = plt.subplots(nrows=2, figsize=(10,10), sharex=True)

bins = 30
for ax, state in zip(axs, states):
    _, bins, _ = ax.hist(ret_avgs[env_flags == state], bins=bins, edgecolor="black")
    ax.set_title(f"{state}")
```

# Transition matrix

```python
n = len(env_flags)
n_remainder = n % window_size
start = n_remainder // 2
stop = n - (n_remainder - start)
n_splits = (stop - start) // window_size
non_overlapping_idxs = np.arange(start, stop)
non_overlapping_idxs = np.array(np.split(non_overlapping_idxs, n_splits))
non_overlapping_envs = env_flags[non_overlapping_idxs]
non_overlapping_envs.shape
```

```python
obs_a = non_overlapping_envs[:-1]
obs_b = non_overlapping_envs[1:]

transition_matrix = np.array([
    [
        np.logical_and(obs_a == state_a, obs_b == state_b).sum().astype(float)
        for state_b in states
    ]
    for state_a in states
])

transition_matrix /= transition_matrix.sum(-1, keepdims=True)
transition_matrix
```

```python
state_0[:,1].sum()
```

```python

```
