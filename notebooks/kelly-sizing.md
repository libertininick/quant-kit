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
df = pd.read_csv("../data/world-indices/$N225.csv")
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
    trends = np.apply_along_axis(get_robust_trend_coef, axis=-1, arr=windows),
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
vol_flags = np.where(rolling_stats["vols"] <= vol_threshold, 0, 1)
vol_states, vol_states_ct = np.unique(vol_flags, return_counts=True)
vol_states, vol_states_ct
```

```python
trend_flags = np.digitize(rolling_stats["trends"], bins=[-1.0, -0.25, 0.25, 1.0]) - 1
trend_states, trend_states_ct = np.unique(trend_flags, return_counts=True)
trend_states, trend_states_ct
```

```python
env_states_shape = (len(vol_states), len(trend_states))
env_states = np.arange(np.prod(env_states_shape)).reshape(env_states_shape)
env_flags = env_states[vol_flags, trend_flags]
_, env_states_ct = np.unique(env_flags, return_counts=True)
env_states, (env_states_ct.reshape(env_states_shape) / env_states_ct.sum()).round(2)
```

```python
for state in (states := env_states.flatten()):
    s = rolling_stats["avgs"][env_flags == state] * 250
    print(f"{state}: {s.mean():>10.4f} {s.std():>10.4f}")
```

```python
fig, axs = plt.subplots(nrows=len(states), figsize=(10,5*len(states)), sharex=True)

bins = np.linspace(
    rolling_stats["avgs"].min(),
    rolling_stats["avgs"].max(),
    50,
)
for ax, state in zip(axs, states):
    _, bins, _ = ax.hist(rolling_stats["avgs"][env_flags == state], bins=bins, edgecolor="black")
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
n_states = env_states.size
transition_states = np.arange(n_states**2).reshape(n_states, n_states)

_, transition_counts = np.unique(
    transition_states[non_overlapping_envs[:-1], non_overlapping_envs[1:]],
    return_counts=True
)
transition_matrix = transition_counts.reshape(n_states, n_states).astype(np.float64)
transition_matrix /= transition_matrix.sum(-1, keepdims=True)
transition_matrix.round(2)
```

```python
baseline_freqs = (env_states_ct.reshape(env_states_shape) / env_states_ct.sum())
tm4d = transition_matrix.reshape(env_states_shape+env_states_shape)
```

```python
vol_state = 1
trend_state = 0

print("Baseline")
print(baseline_freqs.round(2))
print()
print("Transition Probs")
print(tm4d[vol_state, trend_state].round(2))
print()
print("Rel to baseline")
print((tm4d[vol_state, trend_state] - baseline_freqs).round(2))
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
