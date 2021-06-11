# Running HPF-SpecMatch for Target G 9-40


## Import packages


```python
from importlib import reload
import os
import glob2
import pandas as pd
import numpy as np
import sys
import hpfspec
import hpfspecmatch
```

## Read in Library data


```python
# download library if does not exist
hpfspecmatch.utils.get_library()
```

```python
# List of stellar library fits files
library_fitsfiles = hpfspecmatch.config.LIBRARY_FITSFILES

# Read in all files as a HPFSpecList object
HLS =  hpfspec.HPFSpecList(filelist=library_fitsfiles)
```

```python
# list library stars contained in HPFSpecList object
HLS.splist
```

```python
# More info on the targets
HLS.df
```

```python
# Read in required information on all of the
# This has the Teff, Fe/H, and logg for all of the stars
# OBJECT_ID is the HPF name of the star
df_lib = pd.read_csv(hpfspecmatch.config.PATH_LIBRARY_DB)
df_lib
```


## Read in example target


```python
# read in target G-9-40
filename = '../input/G_9-40/Slope-20190301T024821_R01.optimal.fits'
targetname = 'G_9-40'
```


```python
# Bounds to use for different HPF orders (used for knowing where to start/stop resampling the orders)
hpfspecmatch.BOUNDS
```


## Run hpfspecmatch


```python
# Running full hpfspecmatch for all orders above
# Results and plots will be saved to output directory
outputdir = '../output/G_9-40'
orders=['4', '5', '6', '14', '15', '16', '17']
hpfspecmatch.run_specmatch_for_orders(filename, targetname, outputdir, HLS=HLS, orders=orders)
```


## Summarizing orders for a given HPFSpecMatch Run

The following code explores the standard deviations in the stellar parameters between different orders to check how consistent they are across orders. For the formal parameter uncertainties, we recommend using the cross-validation technique.


```python
# collects pickle files for each order of SpecMatch analysis for given target
# contains teff, logg, feh, vsini, weights, and filenames for each order
targetname = 'G_9-40'
files = sorted(glob2.glob('../output/{}/*/*.pkl'.format(targetname)))
```

```python
# summarizes values from different orders from SpecMatch analysis
# saves two csv files to target output directory:
# 1) parameter values from all of the orders
# 2) median and the standard deviation from different orders

df_orders, df_orders_summary = hpfspecmatch.summarize_values_from_orders(files,targetname)
```

```python
df_orders_summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parameters</th>
      <th>median</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>teff</td>
      <td>3429.956155</td>
      <td>18.597423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>logg</td>
      <td>4.872851</td>
      <td>0.008384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>feh</td>
      <td>-0.058265</td>
      <td>0.023794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vsini</td>
      <td>1.835627</td>
      <td>0.288831</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_orders
```
