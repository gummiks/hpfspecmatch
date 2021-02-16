# Installation

There are two main ways to install `HPF-SpecMatch`:


## From Git
`HPF-SpecMatch` can be installed from Git in the following way:
```
git clone git@github.com:gummiks/hpfspecmatch.git
cd hpfspecmatch
python setup.py install
```

## From pip
`HPF-SpecMatch` can be installed from pip with the following command:
```
pip install hpfspecmatch 
```

## Dependencies

- pyde, either (pip install pyde) or install from here: https://github.com/hpparvi/PyDE This package needs numba (try 'conda install numba' if problems).
- emcee (pip install emcee)
- crosscorr (git clone git@github.com:gummiks/crosscorr.git)
- hpfspec (git clone git@github.com:gummiks/hpfspec.git)
- astroquery (pip install astroquery)
- lmfit (pip install lmfit)
- barycorrpy (pip install barycorrpy)

