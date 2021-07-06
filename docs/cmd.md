# Command Line

`HPF-SpecMatch` can be run in terminal using the following commands

## HPF-SpecMatch

To run `HPF-SpecMatch` on a target spectrum:

```python
$ run_specmatch.py [-h] [--savefolder SAVEFOLDER] [--orders ORDERS [ORDERS ...]] [--vsinimax VSINIMAX] filename object
```

### Arguments

**filename**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name of the target spectrum file

**object**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name of object to reduce (SIMBAD or TIC Queryable)

#### Optional

**--savefolder**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output directory to save result files

**--orders**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Which HPF orders to run (orders 4, 5, 6, 14, 15, 16, and 17 recommended as they are the cleanest orders with minimal tellurics)

**--vsinimax**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Maximum v sin i to fit for

```python
$ filename = input/G_9-40/Slope-20190301T024821_R01.optimal.fits

$ run_specmatch.py --savefolder results --orders 5 17 --vsinimax 20 filename G_9-40
```

## Cross-Validation

To run the leave-one-out cross-validation process and assess the stellar library performance for a given order:

```python
$ run_crossval.py [-h] [--df_lib DF_LIB] [--HLS HLS] [--savefolder SAVEFOLDER] [--plot_results] order
```

### Arguments

**order**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Which HPF order to use for cross validation

#### Optional

**--df_lib**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to stellar library csv

**--HLS**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;List of stellar library fits files

**--savefolder**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output directory to save result files

**--plot_results**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Save cross validation summary plots

```python
$ run_crossval.py --df_lib hpfspecmatch.config.PATH_LIBRARY_DB  --HLS hpfspecmatch.config.LIBRARY_FITSFILES --savefolder results --plot_results True 17
```
