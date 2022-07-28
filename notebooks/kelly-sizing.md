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
import scipy.stats as stats

from numpy import ndarray
from quant_kit_core.time_series import (
    get_robust_trend_coef, 
    get_rolling_windows,
)


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
window_size = 60
windows = get_rolling_windows(log_ret, window_size)


# Rolling stats
rolling_stats = dict(
    dates = df.Date.values[np.arange(window_size, len(df))],
    avgs = windows.mean(-1),
    vols = np.median(np.abs(windows), axis=-1),
    trends = np.apply_along_axis(get_robust_trend_coef, axis=-1, arr=windows.cumsum(axis=-1)),
)
```

```python
fig, axs = plt.subplots(nrows=3, figsize=(15,15))
ticks = np.arange(0, len(rolling_stats["dates"]), 250*5)

axs[0].plot(rolling_stats["avgs"])
axs[0].set_xticks(ticks)
axs[0].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[0].set_title("Rolling avg return")

axs[1].plot(rolling_stats["vols"])
axs[1].set_xticks(ticks)
axs[1].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[1].set_title("Rolling vol")

axs[2].plot(rolling_stats["trends"])
axs[2].set_xticks(ticks)
axs[2].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[2].set_title("Rolling trends")
```

# Environment segmentation

```python
vol_threshold = np.median(rolling_stats["vols"])
vol_flags = np.where(rolling_stats["vols"] <= vol_threshold, 1, 0)
```

```python
trend_flags = np.digitize(rolling_stats["trends"], bins=[-1.0, -0.25, 0.25, 1.0])
```

```python
np.unique(z, return_counts=True)
```

```python

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

## Fit distribution to each env

```python
dist = stats.distributions.johnsonsu

env_dists = dict()
fig, axs = plt.subplots(nrows=2, figsize=(10,10), sharex=True)
bins = 30
for ax, state in zip(axs, states):
    env_dists[state] = dist(*dist.fit(ret_avgs[env_flags == state]))
    _, bins, _ = ax.hist(
        env_dists[state].rvs(10000),
        bins=bins,
        edgecolor="black"
    )
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
def get_next_state(state: int, transition_matrix: ndarray) -> int:
    p = transition_matrix[state]
    return np.random.choice(np.arange(len(p)), size=1, p=p)[0]
```

```python
n_sims = 1000
n_steps = 30*4
sim_states = []
for _ in range(n_sims):
    
    state = np.random.choice(np.arange(len(transition_matrix)), size=1)[0]
    states = [state]
    for _ in range(n_steps - 1):
        state = get_next_state(state, transition_matrix)
        states.append(state)
    sim_states.append(states)
sim_states = np.array(sim_states)
```

```python
sim_rets = np.vectorize(lambda x: env_dists.get(x).rvs(1))(sim_states)
```

```python
tot_rets = np.exp((sim_rets * window_size).sum(-1))
fig, ax = plt.subplots(figsize=(10,5), sharex=True)
_ = ax.hist(
    tot_rets,
    bins=30,
    edgecolor="black",
)
```

```python
np.quantile(tot_rets, q=[0.05, 0.25, 0.5, 0.75, 0.95])
```

```python

```
