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
from collections import defaultdict
from functools import reduce, cached_property
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

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
data = []
for fp in Path("./data/sector-etfs").glob("*.csv"):
    d = pd.read_csv(fp)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date", drop=True)
    p = d.Close.values
    d["log_ret"] = (log_ret := np.append([0],np.log(p[1:]/p[:-1])))
    d["vol_20d"] = (vol := d["log_ret"].rolling(20).apply(lambda x: np.median(x**2)**0.5))
    d["norm_ret"] = np.append([np.nan], log_ret[1:] / vol[:-1])
    data.append(d.loc[:, ["norm_ret"]].rename({"norm_ret": fp.stem}, axis="columns"))
data = pd.concat(data, axis=1).dropna(axis=0, how="any")
```

```python
class RollingDataset(Dataset):
    def __init__(
        self,
        data: DataFrame, 
        n_lookback_months: int, 
        n_forecast_months: int,
        p_noise: float = 0,
        idxs: ArrayLike = None,
        blackout_idxs: ArrayLike = None,
    ):
        lookback_len = 2**int(np.log(n_lookback_months*20)/np.log(2) + 0.5)
        forecast_len = n_forecast_months*20
        
        self.data = data.values
        self.names = data.columns.values
        
        self.p_noise = p_noise
        self.noise_dist = stats.t(*stats.t.fit(data.values.flatten()))
        
        self.input_slices, self.target_slices = [],[]
        
        if idxs is None:
            idxs = np.arange(lookback_len + forecast_len, len(data) + 1)
            
        if blackout_idxs is None:
            blackout_idxs = np.array([])
        
        for forecast_end in idxs:
            input_st = forecast_end - (lookback_len + forecast_len)
            input_end = forecast_end - forecast_len
            if len(np.intersect1d(blackout_idxs, np.arange(input_st, forecast_end))) == 0:
                self.input_slices.append((input_st, input_end))
                self.target_slices.append((input_end, forecast_end))
    
    @cached_property
    def baseline_error(self) -> float:
        """
        Returns
        -------
        Tensor, shape=(n_sectors,)
        """
        all_targets = torch.stack(
            [self.get_target(idx) for idx in range(len(self))],
            dim=0
        )
        return (all_targets - 0).pow(2).mean().pow(0.5).item()
            
    def __len__(self):
        return len(self.input_slices)

    def __getitem__(self, idx):
        return self.get_inputs(idx), self.get_target(idx)
    
    def get_inputs(self, idx: int) -> Tensor:
        """
        Returns
        -------
        Tensor, shape=(n_sectors, lookback_len)
        """
        st, end = self.input_slices[idx]
        inputs = self.data[st:end].T
        inputs = inputs * (1 - self.p_noise) + self.noise_dist.rvs(size=inputs.shape) * self.p_noise
        return torch.tensor(inputs, dtype=torch.float32)
    
    def get_target(self, idx: int) -> Tensor:
        """
        Returns
        -------
        Tensor, shape=(n_sectors,)
        """
        st, end = self.target_slices[idx]
        target = self.data[st:end]
        target = target * (1 - self.p_noise) + self.noise_dist.rvs(size=target.shape) * self.p_noise
        target = (target - target.mean(1, keepdims=True)).mean(0)
        return torch.tensor(target, dtype=torch.float32)
```

```python

```

```python
n_lookback_months = 24
n_forecast_months = 3
lookback_len = 2**int(np.log(n_lookback_months*20)/np.log(2) + 0.5)
forecast_len = n_forecast_months*20
n_cv_folds = 5

all_idxs = np.arange(lookback_len + forecast_len, len(data))
cv_splits = np.split(all_idxs[:len(all_idxs) // n_cv_folds * n_cv_folds], n_cv_folds)

```

```python
cv_idxs = cv_splits[0]

train_idxs = np.setdiff1d(all_idxs, cv_idxs)
valid_idxs = cv_idxs[:len(cv_idxs)//2]
test_idxs = cv_idxs[len(cv_idxs)//2:]

train_dataset = RollingDataset(
    data, 
    n_lookback_months=24, 
    n_forecast_months=1,
    p_noise=0.05,
    blackout_idxs=cv_idxs
)

valid_dataset = RollingDataset(
    data, 
    n_lookback_months=24, 
    n_forecast_months=1,
    p_noise=0.05,
    idxs=valid_idxs,
    blackout_idxs=test_idxs,
)

test_dataset = RollingDataset(
    data, 
    n_lookback_months=24, 
    n_forecast_months=1,
    p_noise=0,
    idxs=test_idxs,
)
len(train_dataset), len(valid_dataset), len(test_dataset)
```

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
)

vx = torch.stack([valid_dataset.get_inputs(i) for i in range(len(valid_dataset))])
vy = torch.stack([valid_dataset.get_target(i) for i in range(len(valid_dataset))])
```

```python
model = nn.Sequential(
    nn.Conv1d(8,14,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=14),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(14,24,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=24),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(24,42,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=42),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(42,72,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=72),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(72,126,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=126),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(126,220,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=220),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(220,384,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=384),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(384,672,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=672),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(672,1176,kernel_size=4,stride=2,padding=1),
    nn.BatchNorm1d(num_features=1176),
    nn.GELU(),
    nn.Dropout(p=0.25),
    nn.Conv1d(1176,8,kernel_size=1,stride=1,padding=0),
)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
```

```python
n_epochs=30
losses = []
for i in range(n_epochs):
    # Train
    model.train()
    train_losses = []
    for x, y in train_loader:
        yh = model(x).squeeze(-1)
        loss = (yh - y).pow(2).mean().pow(0.5)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
     # Train
    model.eval()
    with torch.no_grad():
        yh = model(vx).squeeze(-1)
        loss = (yh - vy).pow(2).mean().pow(0.5)
        losses.append(loss / valid_dataset.baseline_error)

    print(f"{i:<3} {np.mean(train_losses) / train_dataset.baseline_error:>10.2%} {losses[-1]:>10.2%}")
```

```python

```
