# Getting Started

## Using `iDiffuse` to calculate expected photometric precisions

### Import necessary packages


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

### Read in library

Make sure there is a folder called ../library/20201008_specmatch_nir/

Download data from here and unzip in the ../library/ folder

- *insert dropbox link*
It should be structured:

- ../library/20201008_specmatch_nir/20201008_specmatch_nir.csv
- ../library/20201008_specmatch_nir/FITS/

```python
# List of fitsfiles
LIBRARY_DIR = '../library/20201008_specmatch_nir/'

library_fitsfiles = glob2.glob(LIBRARY_DIR+'FITS/*/*.fits')
library_fitsfiles # should be many fitsfiles

```
```python
# Read in all files as a HPFSpecList object
HLS = hpfspec.HPFSpecList(filelist=library_fitsfiles)
```
```python
# Read in required information on all of the
# This has the Teff, Fe/H, and logg for all of the stars
# OBJECT_ID is the HPF name of the star
df_lib = pd.read_csv(LIBRARY_DIR+'20201008_specmatch_nir.csv')
df_lib
```
### Read in example target

```python
# Target data
targetfilename = '../input/toi1468/Slope-20200129T015117_R01.optimal.fits'
Htarget = hpfspec.HPFSpectrum(targetfilename,targetname='UCAC4 547-002110')

# Reference data
Hrefs   = HLS.splist
```