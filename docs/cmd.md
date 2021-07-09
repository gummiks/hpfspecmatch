# Command Line

`HPF-SpecMatch` can be run in terminal using the following commands

# HPF-SpecMatch

To run `HPF-SpecMatch` on a target spectrum:

```python
$ python run_hpfspecmatch.py [-h] [--savefolder SAVEFOLDER] [--orders ORDERS [ORDERS ...]] [--vsinimax VSINIMAX] filename object
```

### Required Arguments

**filename**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name of the target spectrum file (e.g. input/G_9-40/Slope-20190301T024821_R01.optimal.fits)

**object**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name of object to reduce; SIMBAD or TIC Queryable (e.g. G_9-40)

### Optional Arguments

**--savefolder**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output directory to save result files (e.g. --savefolder results)

**--orders**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Which HPF orders to run; orders 4, 5, 6, 14, 15, 16, and 17 recommended as they are the cleanest orders with minimal tellurics (e.g. --orders 5 17)

**--vsinimax**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Maximum v sin i to fit for in km/s (e.g. --vsinimax 20)

### Example code to run HPF-SpecMatch with G 9-40 spectrum

```python
$ python run_hpfspecmatch.py --savefolder results --orders 4 5 6 14 15 16 17 --vsinimax 20 input/G_9-40/Slope-20190301T024821_R01.optimal.fits G_9-40
```

# Cross-Validation

To run the leave-one-out cross-validation process and assess the stellar library performance for a given order:

```python
$ python run_crossval.py [-h] [--df_lib DF_LIB] [--HLS HLS] [--savefolder SAVEFOLDER] [--plot_results] order
```

### Required Arguments

**order**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Which HPF order to use for cross-validation; orders 4, 5, 6, 14, 15, 16, and 17 recommended as they are the cleanest orders with minimal tellurics (e.g. 5 17)

### Optional Arguments

**--df_lib**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Path to stellar library csv (defaults to installation library csv file)

**--HLS**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;List of stellar library fits files (defaults to installation library)

**--savefolder**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Output directory to save result files (e.g. --savefolder results)

**--plot_results**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Save cross validation summary plots; flag to plot results

### Example code to run stellar library cross-validation with HPF Order 17

```python
$ python run_crossval.py --savefolder results --plot_results 17
```
