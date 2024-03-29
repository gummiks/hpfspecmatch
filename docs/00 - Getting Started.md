```python
from importlib import reload
import os
import glob2
import pandas as pd
import numpy as np
import sys
#sys.path.pop(0)
```

# Import our package 


```python
import hpfspec
import hpfspecmatch
hpfspecmatch
```




    <module 'hpfspecmatch' from '/home/sejones/anaconda3/lib/python3.8/site-packages/hpfspecmatch-0.1.0-py3.8.egg/hpfspecmatch/__init__.py'>



# Read in library 

Make sure there is a folder called ../library/20200128_specmatch_nir/

Download data from here and unzip in the ../library/ folder

- https://www.dropbox.com/s/8fcraxmpgqdq9w9/20200128_specmatch_nir.zip?dl=0

Should be structured:
-  ../library/20200128_specmatch_nir/20200128_specmatch_nir.csv
- ../library/20200128_specmatch_nir/FITS/


```python
# List of fitsfiles
LIBRARY_DIR = '../library/20200128_specmatch_nir/'

library_fitsfiles = glob2.glob(LIBRARY_DIR+'FITS/*/*.fits')
library_fitsfiles # should be many fitsfiles
```




    ['../library/20200128_specmatch_nir/FITS/265_GL109/Slope-20181221T061620_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/390_95735/Slope-20190224T102122_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/391_GJ176/Slope-20191208T040844_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/318_119850/Slope-20191222T122719_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/288_HIP15366/Slope-20191224T011605_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/401_HIP57087/Slope-20181221T095756_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/297_HIP47513/Slope-20191219T124524_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/169_87883/Slope-20191118T102934_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/107_24238/Slope-20191118T052618_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/305_HIP65016/Slope-20200114T095403_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/262_28343/Slope-20191119T101149_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/385_79210/Slope-20191130T090648_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/301_HIP57548/Slope-20190319T054134_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/273_GL625/Slope-20200124T122211_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/279_GL87/Slope-20191207T055134_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/248_HIP54810/Slope-20191125T120712_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/21_112914/Slope-20200108T094852_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/267_GL239/Slope-20191115T121100_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/259_232979/Slope-20191116T103941_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/241_HIP36551/Slope-20191201T115615_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/144_53927/Slope-20191125T122553_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/245_HIP47201/Slope-20191116T101405_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/272_GL514/Slope-20191216T122849_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/389_88230/Slope-20191119T103851_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/397_GL699/Slope-20190319T113007_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/18_110833/Slope-20191221T105911_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/15_110463/Slope-20191203T123535_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/285_HIP11048/Slope-20191208T013316_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/246_HIP48411/Slope-20191118T112544_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/271_GL393/Slope-20200114T104423_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/374_36395/Slope-20191213T062214_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/266_GL2066/Slope-20200107T065409_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/43_144872/Slope-20200112T124748_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/243_HIP40910/Slope-20191125T084533_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/303_HIP60093/Slope-20191123T122518_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/240_HIP25220/Slope-20191118T105426_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/278_GL83.1/Slope-20190915T111342_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/253_111631/Slope-20200113T110819_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/290_HIP21556/Slope-20191213T060232_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/254_1326B/Slope-20181224T033526_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/269_GL273/Slope-20181104T095012_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/123_35112/Slope-20191117T070203_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/260_245409/Slope-20191116T105902_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/250_HIP66222/Slope-20200114T110022_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/270_GL382/Slope-20191207T123306_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/261_265866/Slope-20200114T091114_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/341_173740/Slope-20190316T115749_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/393_GL338B/Slope-20191201T092225_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/264_GL105B/Slope-20200108T040109_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/30_128311/Slope-20200105T125333_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/176_96612/Slope-20191201T102853_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/236_HIP12493/Slope-20191116T070359_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/148_61606/Slope-20191118T101626_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/302_HIP59748/Slope-20191121T122807_R01.optimal.fits',
     '../library/20200128_specmatch_nir/FITS/125_37008/Slope-20191127T053426_R01.optimal.fits']




```python
# Read in all files as a HPFSpecList object
HLS = hpfspec.HPFSpecList(filelist=library_fitsfiles)
```

    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_109.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 30.50000km/s
    Second iteration: RVabs = 30.51588km/s, sigma=2.84619
    RVabs it #1: 30.50000+- 0.00000km/s
    RVabs it #2: 30.51588+- 0.00000km/s
    berv=-20.558423139315074,rv=30.515882303673667
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_411.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -84.66667km/s
    Second iteration: RVabs = -84.71157km/s, sigma=2.86970
    RVabs it #1: -84.66667+- 0.00000km/s
    RVabs it #2: -84.71157+- 0.00000km/s
    berv=-1.3179527415006078,rv=-84.7115659410015
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_176.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 26.00000km/s
    Second iteration: RVabs = 26.18685km/s, sigma=2.90934
    RVabs it #1: 26.00000+- 0.00000km/s
    RVabs it #2: 26.18685+- 0.00000km/s
    berv=-1.976092023437395,rv=26.186852199129387
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_526.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 15.83333km/s
    Second iteration: RVabs = 15.80178km/s, sigma=2.95392
    RVabs it #1: 15.83333+- 0.00000km/s
    RVabs it #2: 15.80178+- 0.00000km/s
    berv=26.462136029142705,rv=15.801779092302631
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_134.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -4.16667km/s
    Second iteration: RVabs = -4.23217km/s, sigma=2.87327
    RVabs it #1: -4.16667+- 0.00000km/s
    RVabs it #2: -4.23217+- 0.00000km/s
    berv=-15.999002411312265,rv=-4.232165813467776
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_436.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 9.50000km/s
    Second iteration: RVabs = 9.57729km/s, sigma=2.82349
    RVabs it #1:  9.50000+- 0.00000km/s
    RVabs it #2:  9.57729+- 0.00000km/s
    berv=27.257446021002547,rv=9.577288927823846
    Defaulting to fixed wavelength
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_361.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 11.50000km/s
    Second iteration: RVabs = 11.46329km/s, sigma=2.92564
    RVabs it #1: 11.50000+- 0.00000km/s
    RVabs it #2: 11.46329+- 0.00000km/s
    berv=24.870200714358766,rv=11.463285805782832
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_87883.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 10.00000km/s
    Second iteration: RVabs = 9.82649km/s, sigma=2.94098
    RVabs it #1: 10.00000+- 0.00000km/s
    RVabs it #2:  9.82649+- 0.00000km/s
    berv=28.21631999785097,rv=9.826486104784786
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_24238.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -112.33333km/s
    Second iteration: RVabs = -108.48812km/s, sigma=25.28712
    RVabs it #1: -112.33333+- 0.00000km/s
    RVabs it #2: -108.48812+- 0.00000km/s
    berv=6.08110408565654,rv=-108.4881221909557
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/NLTT_33716.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -11.66667km/s
    Second iteration: RVabs = -11.69989km/s, sigma=2.83920
    RVabs it #1: -11.66667+- 0.00000km/s
    RVabs it #2: -11.69989+- 0.00000km/s
    berv=22.587365952655855,rv=-11.699893837753025
    Defaulting to fixed wavelength
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_28343.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -35.00000km/s
    Second iteration: RVabs = -34.94685km/s, sigma=3.01175
    RVabs it #1: -35.00000+- 0.00000km/s
    RVabs it #2: -34.94685+- 0.00000km/s
    berv=5.922571435310169,rv=-34.94684756477991
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_338_A.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 12.50000km/s
    Second iteration: RVabs = 12.45851km/s, sigma=3.02997
    RVabs it #1: 12.50000+- 0.00000km/s
    RVabs it #2: 12.45851+- 0.00000km/s
    berv=20.58696596392642,rv=12.458512468018553
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_447.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -31.16667km/s
    Second iteration: RVabs = -31.14442km/s, sigma=2.84885
    RVabs it #1: -31.16667+- 0.00000km/s
    RVabs it #2: -31.14442+- 0.00000km/s
    berv=-0.007811766567551857,rv=-31.144416132676614
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_625.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -13.00000km/s
    Second iteration: RVabs = -13.06142km/s, sigma=2.93968
    RVabs it #1: -13.00000+- 0.00000km/s
    RVabs it #2: -13.06142+- 0.00000km/s
    berv=8.86289645684853,rv=-13.061424864293096
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_87.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -2.66667km/s
    Second iteration: RVabs = -2.58008km/s, sigma=2.97007
    RVabs it #1: -2.66667+- 0.00000km/s
    RVabs it #2: -2.58008+- 0.00000km/s
    berv=-20.543150317886102,rv=-2.5800832372702827
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_418.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 16.83333km/s
    Second iteration: RVabs = 17.02321km/s, sigma=3.11698
    RVabs it #1: 16.83333+- 0.00000km/s
    RVabs it #2: 17.02321+- 0.00000km/s
    berv=29.441934160271153,rv=17.023210170322407
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/NLTT_32537.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 27.33333km/s
    Second iteration: RVabs = 27.06454km/s, sigma=3.01868
    RVabs it #1: 27.33333+- 0.00000km/s
    RVabs it #2: 27.06454+- 0.00000km/s
    berv=20.17324465059718,rv=27.064544788502022
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_260655.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -58.16667km/s
    Second iteration: RVabs = -58.20504km/s, sigma=2.97358
    RVabs it #1: -58.16667+- 0.00000km/s
    RVabs it #2: -58.20504+- 0.00000km/s
    berv=21.19193429927581,rv=-58.20503843959427
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_232979.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 34.66667km/s
    Second iteration: RVabs = 34.41472km/s, sigma=3.08813
    RVabs it #1: 34.66667+- 0.00000km/s
    RVabs it #2: 34.41472+- 0.00000km/s
    berv=9.36796943464812,rv=34.41472390686219
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_276.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 66.50000km/s
    Second iteration: RVabs = 66.56320km/s, sigma=2.99957
    RVabs it #1: 66.50000+- 0.00000km/s
    RVabs it #2: 66.56320+- 0.00000km/s
    berv=20.20453134118459,rv=66.56320325567275
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_3428.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 20.50000km/s
    Second iteration: RVabs = 20.18247km/s, sigma=3.01958
    RVabs it #1: 20.50000+- 0.00000km/s
    RVabs it #2: 20.18247+- 0.00000km/s
    berv=19.639573108958004,rv=20.18247009606754
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/BD+23_2121.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -35.66667km/s
    Second iteration: RVabs = -35.74342km/s, sigma=2.90756
    RVabs it #1: -35.66667+- 0.00000km/s
    RVabs it #2: -35.74342+- 0.00000km/s
    berv=29.938223073008146,rv=-35.74342147344061
    Defaulting to fixed wavelength
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_514.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 14.66667km/s
    Second iteration: RVabs = 14.59507km/s, sigma=2.89693
    RVabs it #1: 14.66667+- 0.00000km/s
    RVabs it #2: 14.59507+- 0.00000km/s
    berv=26.756925258853183,rv=14.5950740551564
    Defaulting to fixed wavelength
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_88230.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -25.66667km/s
    Second iteration: RVabs = -25.73888km/s, sigma=3.00831
    RVabs it #1: -25.66667+- 0.00000km/s
    RVabs it #2: -25.73888+- 0.00000km/s
    berv=24.200600486956823,rv=-25.738878963937104
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_699.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -110.50000km/s
    Second iteration: RVabs = -110.57234km/s, sigma=2.98273
    RVabs it #1: -110.50000+- 0.00000km/s
    RVabs it #2: -110.57234+- 0.00000km/s
    berv=26.422657442994453,rv=-110.57233697001388
    Defaulting to fixed wavelength
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_483.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 10.50000km/s
    Second iteration: RVabs = 10.45719km/s, sigma=3.08731
    RVabs it #1: 10.50000+- 0.00000km/s
    RVabs it #2: 10.45719+- 0.00000km/s
    berv=18.65081822443051,rv=10.457190804162513
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_3743.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -9.33333km/s
    Second iteration: RVabs = -9.18861km/s, sigma=3.17345
    RVabs it #1: -9.33333+- 0.00000km/s
    RVabs it #2: -9.18861+- 0.00000km/s
    berv=18.261563112685042,rv=-9.188611202534236
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_96.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -37.83333km/s
    Second iteration: RVabs = -37.91472km/s, sigma=2.94734
    RVabs it #1: -37.83333+- 0.00000km/s
    RVabs it #2: -37.91472+- 0.00000km/s
    berv=-10.994345758256905,rv=-37.91471736271393
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_85488.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 22.00000km/s
    Second iteration: RVabs = 22.07867km/s, sigma=3.06076
    RVabs it #1: 22.00000+- 0.00000km/s
    RVabs it #2: 22.07867+- 0.00000km/s
    berv=29.871508202468732,rv=22.0786668026464
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_393.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 8.33333km/s
    Second iteration: RVabs = 8.29990km/s, sigma=2.93736
    RVabs it #1:  8.33333+- 0.00000km/s
    RVabs it #2:  8.29990+- 0.00000km/s
    berv=21.243650224722227,rv=8.299899971705932
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_205.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 8.83333km/s
    Second iteration: RVabs = 8.69036km/s, sigma=2.95880
    RVabs it #1:  8.83333+- 0.00000km/s
    RVabs it #2:  8.69036+- 0.00000km/s
    berv=0.5230519947041814,rv=8.690360251036259
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_2066.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 62.33333km/s
    Second iteration: RVabs = 62.18578km/s, sigma=2.92195
    RVabs it #1: 62.33333+- 0.00000km/s
    RVabs it #2: 62.18578+- 0.00000km/s
    berv=9.937567906060561,rv=62.185777722714434
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_612.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 24.00000km/s
    Second iteration: RVabs = 23.99270km/s, sigma=3.17296
    RVabs it #1: 24.00000+- 0.00000km/s
    RVabs it #2: 23.99270+- 0.00000km/s
    berv=14.892805953230663,rv=23.99270030342034
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_3494.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 7.83333km/s
    Second iteration: RVabs = 7.64419km/s, sigma=3.20490
    RVabs it #1:  7.83333+- 0.00000km/s
    RVabs it #2:  7.64419+- 0.00000km/s
    berv=26.519209957994395,rv=7.644186648222679
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/BD+29_2279.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -0.83333km/s
    Second iteration: RVabs = -0.64851km/s, sigma=2.93943
    RVabs it #1: -0.83333+- 0.00000km/s
    RVabs it #2: -0.64851+- 0.00000km/s
    berv=25.175974770530825,rv=-0.6485063364418017
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/*_111_Tau_B.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 38.50000km/s
    Second iteration: RVabs = 38.89993km/s, sigma=3.25551
    RVabs it #1: 38.50000+- 0.00000km/s
    RVabs it #2: 38.89993+- 0.00000km/s
    berv=12.474248238189897,rv=38.899926819025445
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_9066.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -28.50000km/s
    Second iteration: RVabs = -28.69814km/s, sigma=3.66684
    RVabs it #1: -28.50000+- 0.00000km/s
    RVabs it #2: -28.69814+- 0.00000km/s
    berv=18.633029064449087,rv=-28.69814496523769
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_488.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 5.16667km/s
    Second iteration: RVabs = 5.10396km/s, sigma=3.05968
    RVabs it #1:  5.16667+- 0.00000km/s
    RVabs it #2:  5.10396+- 0.00000km/s
    berv=29.832466158203054,rv=5.1039578093524085
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_173.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -7.00000km/s
    Second iteration: RVabs = -6.78727km/s, sigma=2.86088
    RVabs it #1: -7.00000+- 0.00000km/s
    RVabs it #2: -6.78727+- 0.00000km/s
    berv=-6.680746725273167,rv=-6.787271198493364
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_15_B.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 10.83333km/s
    Second iteration: RVabs = 10.96082km/s, sigma=2.89952
    RVabs it #1: 10.83333+- 0.00000km/s
    RVabs it #2: 10.96082+- 0.00000km/s
    berv=-22.251623938616127,rv=10.960822263741909
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_273.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 18.00000km/s
    Second iteration: RVabs = 18.14550km/s, sigma=2.80271
    RVabs it #1: 18.00000+- 0.00000km/s
    RVabs it #2: 18.14550+- 0.00000km/s
    berv=27.261888457060316,rv=18.14549743893623
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_35112.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 38.16667km/s
    Second iteration: RVabs = 38.57547km/s, sigma=3.34087
    RVabs it #1: 38.16667+- 0.00000km/s
    RVabs it #2: 38.57547+- 0.00000km/s
    berv=12.063036503004753,rv=38.57547240668419
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/V*_V2689_Ori.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 22.00000km/s
    Second iteration: RVabs = 22.09554km/s, sigma=3.36194
    RVabs it #1: 22.00000+- 0.00000km/s
    RVabs it #2: 22.09554+- 0.00000km/s
    berv=14.469820316946944,rv=22.095539353639264
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_1172.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 15.00000km/s
    Second iteration: RVabs = 15.00940km/s, sigma=2.93918
    RVabs it #1: 15.00000+- 0.00000km/s
    RVabs it #2: 15.00940+- 0.00000km/s
    berv=29.59114241034386,rv=15.009397507565565
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_382.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 8.00000km/s
    Second iteration: RVabs = 7.90709km/s, sigma=2.97982
    RVabs it #1:  8.00000+- 0.00000km/s
    RVabs it #2:  7.90709+- 0.00000km/s
    berv=28.95154263097634,rv=7.907094172607433
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_251.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 23.00000km/s
    Second iteration: RVabs = 22.89954km/s, sigma=2.81024
    RVabs it #1: 23.00000+- 0.00000km/s
    RVabs it #2: 22.89954+- 0.00000km/s
    berv=-6.209610930654121,rv=22.89954436399567
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_725_B.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -0.66667km/s
    Second iteration: RVabs = -0.71978km/s, sigma=2.87899
    RVabs it #1: -0.66667+- 0.00000km/s
    RVabs it #2: -0.71978+- 0.00000km/s
    berv=3.180900133062689,rv=-0.7197764335829121
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_338_B.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 12.50000km/s
    Second iteration: RVabs = 12.47525km/s, sigma=3.02655
    RVabs it #1: 12.50000+- 0.00000km/s
    RVabs it #2: 12.47525+- 0.00000km/s
    berv=20.32041091086356,rv=12.475246192835195
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_105_B.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 26.16667km/s
    Second iteration: RVabs = 26.16948km/s, sigma=2.85906
    RVabs it #1: 26.16667+- 0.00000km/s
    RVabs it #2: 26.16948+- 0.00000km/s
    berv=-28.021023312359198,rv=26.169475980482684
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_3860.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -8.50000km/s
    Second iteration: RVabs = -8.85452km/s, sigma=3.46894
    RVabs it #1: -8.50000+- 0.00000km/s
    RVabs it #2: -8.85452+- 0.00000km/s
    berv=26.417999937982277,rv=-8.854523084027555
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/NLTT_26380.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -35.66667km/s
    Second iteration: RVabs = -35.57566km/s, sigma=3.07700
    RVabs it #1: -35.66667+- 0.00000km/s
    RVabs it #2: -35.57566+- 0.00000km/s
    berv=26.167540901951085,rv=-35.575660266653664
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/BD+00_444.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 73.50000km/s
    Second iteration: RVabs = 73.39585km/s, sigma=3.02821
    RVabs it #1: 73.50000+- 0.00000km/s
    RVabs it #2: 73.39585+- 0.00000km/s
    berv=-8.15076585903598,rv=73.3958500957068
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/HD_61606.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -17.83333km/s
    Second iteration: RVabs = -17.72558km/s, sigma=3.10865
    RVabs it #1: -17.83333+- 0.00000km/s
    RVabs it #2: -17.72558+- 0.00000km/s
    berv=24.111441252189323,rv=-17.725577465794068
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/BD+49_2126.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -13.83333km/s
    Second iteration: RVabs = -13.92785km/s, sigma=3.08461
    RVabs it #1: -13.83333+- 0.00000km/s
    RVabs it #2: -13.92785+- 0.00000km/s
    berv=21.226893345330513,rv=-13.927846725631623
    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/GJ_3358.config
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = -45.50000km/s
    Second iteration: RVabs = -45.36884km/s, sigma=2.97354
    RVabs it #1: -45.50000+- 0.00000km/s
    RVabs it #2: -45.36884+- 0.00000km/s
    berv=9.770838896949163,rv=-45.36884488242152



```python
# List of all of the HPF spectra
HLS.splist
```




    [HPFSpec(GJ_109,sn18=501.3),
     HPFSpec(GJ_411,sn18=593.1),
     HPFSpec(GJ_176,sn18=261.8),
     HPFSpec(GJ_526,sn18=482.1),
     HPFSpec(GJ_134,sn18=480.1),
     HPFSpec(GJ_436,sn18=476.9),
     HPFSpec(GJ_361,sn18=242.6),
     HPFSpec(HD_87883,sn18=338.6),
     HPFSpec(HD_24238,sn18=469.6),
     HPFSpec(NLTT_33716,sn18=376.0),
     HPFSpec(HD_28343,sn18=430.9),
     HPFSpec(GJ_338_A,sn18=425.5),
     HPFSpec(GJ_447,sn18=438.6),
     HPFSpec(GJ_625,sn18=539.9),
     HPFSpec(GJ_87,sn18=381.9),
     HPFSpec(GJ_418,sn18=360.6),
     HPFSpec(NLTT_32537,sn18=330.3),
     HPFSpec(HD_260655,sn18=567.5),
     HPFSpec(HD_232979,sn18=386.0),
     HPFSpec(GJ_276,sn18=438.6),
     HPFSpec(GJ_3428,sn18=349.3),
     HPFSpec(BD+23_2121,sn18=263.1),
     HPFSpec(GJ_514,sn18=339.5),
     HPFSpec(HD_88230,sn18=536.4),
     HPFSpec(GJ_699,sn18=622.3),
     HPFSpec(GJ_483,sn18=438.9),
     HPFSpec(GJ_3743,sn18=514.1),
     HPFSpec(GJ_96,sn18=460.0),
     HPFSpec(HD_85488,sn18=373.6),
     HPFSpec(GJ_393,sn18=427.8),
     HPFSpec(GJ_205,sn18=370.6),
     HPFSpec(GJ_2066,sn18=499.5),
     HPFSpec(GJ_612,sn18=247.0),
     HPFSpec(GJ_3494,sn18=297.3),
     HPFSpec(BD+29_2279,sn18=379.9),
     HPFSpec(*_111_Tau_B,sn18=425.0),
     HPFSpec(GJ_9066,sn18=484.1),
     HPFSpec(GJ_488,sn18=435.3),
     HPFSpec(GJ_173,sn18=242.4),
     HPFSpec(GJ_15_B,sn18=457.9),
     HPFSpec(GJ_273,sn18=503.4),
     HPFSpec(HD_35112,sn18=334.2),
     HPFSpec(V*_V2689_Ori,sn18=315.0),
     HPFSpec(GJ_1172,sn18=431.4),
     HPFSpec(GJ_382,sn18=447.3),
     HPFSpec(GJ_251,sn18=568.3),
     HPFSpec(GJ_725_B,sn18=618.8),
     HPFSpec(GJ_338_B,sn18=277.8),
     HPFSpec(GJ_105_B,sn18=488.1),
     HPFSpec(GJ_3860,sn18=398.8),
     HPFSpec(NLTT_26380,sn18=313.4),
     HPFSpec(BD+00_444,sn18=345.3),
     HPFSpec(HD_61606,sn18=423.3),
     HPFSpec(BD+49_2126,sn18=291.6),
     HPFSpec(GJ_3358,sn18=266.6)]




```python
# More info on the targets
HLS.df
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
      <th>OBJECT_ID</th>
      <th>filename</th>
      <th>exptime</th>
      <th>sn18</th>
      <th>qprog</th>
      <th>rv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GJ_109</td>
      <td>../library/20200128_specmatch_nir/FITS/265_GL1...</td>
      <td>649.65</td>
      <td>501.315796</td>
      <td>PSU19-1-013</td>
      <td>30.515882</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GJ_411</td>
      <td>../library/20200128_specmatch_nir/FITS/390_957...</td>
      <td>63.90</td>
      <td>593.079102</td>
      <td>ENG19-1-003</td>
      <td>-84.711566</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GJ_176</td>
      <td>../library/20200128_specmatch_nir/FITS/391_GJ1...</td>
      <td>330.15</td>
      <td>261.819824</td>
      <td>UT20-1-008</td>
      <td>26.186852</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GJ_526</td>
      <td>../library/20200128_specmatch_nir/FITS/318_119...</td>
      <td>181.05</td>
      <td>482.063538</td>
      <td>UT20-1-008</td>
      <td>15.801779</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GJ_134</td>
      <td>../library/20200128_specmatch_nir/FITS/288_HIP...</td>
      <td>969.15</td>
      <td>480.095947</td>
      <td>UT20-1-008</td>
      <td>-4.232166</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GJ_436</td>
      <td>../library/20200128_specmatch_nir/FITS/401_HIP...</td>
      <td>649.65</td>
      <td>476.864380</td>
      <td>ENG19-1-003</td>
      <td>9.577289</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GJ_361</td>
      <td>../library/20200128_specmatch_nir/FITS/297_HIP...</td>
      <td>649.65</td>
      <td>242.642822</td>
      <td>UT20-1-008</td>
      <td>11.463286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HD_87883</td>
      <td>../library/20200128_specmatch_nir/FITS/169_878...</td>
      <td>330.15</td>
      <td>338.629211</td>
      <td>UT19-3-008</td>
      <td>9.826486</td>
    </tr>
    <tr>
      <th>8</th>
      <td>HD_24238</td>
      <td>../library/20200128_specmatch_nir/FITS/107_242...</td>
      <td>330.15</td>
      <td>469.626221</td>
      <td>UT19-3-008</td>
      <td>-108.488122</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NLTT_33716</td>
      <td>../library/20200128_specmatch_nir/FITS/305_HIP...</td>
      <td>969.15</td>
      <td>376.039368</td>
      <td>UT20-1-008</td>
      <td>-11.699894</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HD_28343</td>
      <td>../library/20200128_specmatch_nir/FITS/262_283...</td>
      <td>287.55</td>
      <td>430.919281</td>
      <td>UT19-3-008</td>
      <td>-34.946848</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GJ_338_A</td>
      <td>../library/20200128_specmatch_nir/FITS/385_792...</td>
      <td>138.45</td>
      <td>425.514191</td>
      <td>UT19-3-008</td>
      <td>12.458512</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GJ_447</td>
      <td>../library/20200128_specmatch_nir/FITS/301_HIP...</td>
      <td>330.15</td>
      <td>438.552124</td>
      <td>PSU19-1-013</td>
      <td>-31.144416</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GJ_625</td>
      <td>../library/20200128_specmatch_nir/FITS/273_GL6...</td>
      <td>649.65</td>
      <td>539.909912</td>
      <td>UT20-1-008</td>
      <td>-13.061425</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GJ_87</td>
      <td>../library/20200128_specmatch_nir/FITS/279_GL8...</td>
      <td>649.65</td>
      <td>381.879883</td>
      <td>UT20-1-008</td>
      <td>-2.580083</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GJ_418</td>
      <td>../library/20200128_specmatch_nir/FITS/248_HIP...</td>
      <td>649.65</td>
      <td>360.558899</td>
      <td>UT19-3-008</td>
      <td>17.023210</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NLTT_32537</td>
      <td>../library/20200128_specmatch_nir/FITS/21_1129...</td>
      <td>649.65</td>
      <td>330.296814</td>
      <td>UT20-1-008</td>
      <td>27.064545</td>
    </tr>
    <tr>
      <th>17</th>
      <td>HD_260655</td>
      <td>../library/20200128_specmatch_nir/FITS/267_GL2...</td>
      <td>649.65</td>
      <td>567.503235</td>
      <td>HET19-3-300</td>
      <td>-58.205038</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HD_232979</td>
      <td>../library/20200128_specmatch_nir/FITS/259_232...</td>
      <td>330.15</td>
      <td>386.030884</td>
      <td>UT19-3-008</td>
      <td>34.414724</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GJ_276</td>
      <td>../library/20200128_specmatch_nir/FITS/241_HIP...</td>
      <td>649.65</td>
      <td>438.571594</td>
      <td>UT19-3-008</td>
      <td>66.563203</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GJ_3428</td>
      <td>../library/20200128_specmatch_nir/FITS/144_539...</td>
      <td>649.65</td>
      <td>349.262207</td>
      <td>UT19-3-008</td>
      <td>20.182470</td>
    </tr>
    <tr>
      <th>21</th>
      <td>BD+23_2121</td>
      <td>../library/20200128_specmatch_nir/FITS/245_HIP...</td>
      <td>969.15</td>
      <td>263.068237</td>
      <td>UT19-3-008</td>
      <td>-35.743421</td>
    </tr>
    <tr>
      <th>22</th>
      <td>GJ_514</td>
      <td>../library/20200128_specmatch_nir/FITS/272_GL5...</td>
      <td>330.15</td>
      <td>339.472290</td>
      <td>UT20-1-008</td>
      <td>14.595074</td>
    </tr>
    <tr>
      <th>23</th>
      <td>HD_88230</td>
      <td>../library/20200128_specmatch_nir/FITS/389_882...</td>
      <td>63.90</td>
      <td>536.356506</td>
      <td>UT19-3-008</td>
      <td>-25.738879</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GJ_699</td>
      <td>../library/20200128_specmatch_nir/FITS/397_GL6...</td>
      <td>191.70</td>
      <td>622.323425</td>
      <td>ENG19-1-003</td>
      <td>-110.572337</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GJ_483</td>
      <td>../library/20200128_specmatch_nir/FITS/18_1108...</td>
      <td>202.35</td>
      <td>438.896149</td>
      <td>UT20-1-008</td>
      <td>10.457191</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GJ_3743</td>
      <td>../library/20200128_specmatch_nir/FITS/15_1104...</td>
      <td>649.65</td>
      <td>514.133606</td>
      <td>UT20-1-008</td>
      <td>-9.188611</td>
    </tr>
    <tr>
      <th>27</th>
      <td>GJ_96</td>
      <td>../library/20200128_specmatch_nir/FITS/285_HIP...</td>
      <td>330.15</td>
      <td>460.026642</td>
      <td>UT20-1-008</td>
      <td>-37.914717</td>
    </tr>
    <tr>
      <th>28</th>
      <td>HD_85488</td>
      <td>../library/20200128_specmatch_nir/FITS/246_HIP...</td>
      <td>649.65</td>
      <td>373.552734</td>
      <td>UT19-3-008</td>
      <td>22.078667</td>
    </tr>
    <tr>
      <th>29</th>
      <td>GJ_393</td>
      <td>../library/20200128_specmatch_nir/FITS/271_GL3...</td>
      <td>436.65</td>
      <td>427.817566</td>
      <td>UT20-1-008</td>
      <td>8.299900</td>
    </tr>
    <tr>
      <th>30</th>
      <td>GJ_205</td>
      <td>../library/20200128_specmatch_nir/FITS/374_363...</td>
      <td>127.80</td>
      <td>370.626740</td>
      <td>UT20-1-008</td>
      <td>8.690360</td>
    </tr>
    <tr>
      <th>31</th>
      <td>GJ_2066</td>
      <td>../library/20200128_specmatch_nir/FITS/266_GL2...</td>
      <td>649.65</td>
      <td>499.474548</td>
      <td>UT20-1-008</td>
      <td>62.185778</td>
    </tr>
    <tr>
      <th>32</th>
      <td>GJ_612</td>
      <td>../library/20200128_specmatch_nir/FITS/43_1448...</td>
      <td>649.65</td>
      <td>246.984924</td>
      <td>UT20-1-008</td>
      <td>23.992700</td>
    </tr>
    <tr>
      <th>33</th>
      <td>GJ_3494</td>
      <td>../library/20200128_specmatch_nir/FITS/243_HIP...</td>
      <td>969.15</td>
      <td>297.302368</td>
      <td>UT19-3-008</td>
      <td>7.644187</td>
    </tr>
    <tr>
      <th>34</th>
      <td>BD+29_2279</td>
      <td>../library/20200128_specmatch_nir/FITS/303_HIP...</td>
      <td>969.15</td>
      <td>379.905640</td>
      <td>PSU19-3-009</td>
      <td>-0.648506</td>
    </tr>
    <tr>
      <th>35</th>
      <td>*_111_Tau_B</td>
      <td>../library/20200128_specmatch_nir/FITS/240_HIP...</td>
      <td>330.15</td>
      <td>425.017181</td>
      <td>UT19-3-008</td>
      <td>38.899927</td>
    </tr>
    <tr>
      <th>36</th>
      <td>GJ_9066</td>
      <td>../library/20200128_specmatch_nir/FITS/278_GL8...</td>
      <td>969.15</td>
      <td>484.133362</td>
      <td>PSU19-3-014</td>
      <td>-28.698145</td>
    </tr>
    <tr>
      <th>37</th>
      <td>GJ_488</td>
      <td>../library/20200128_specmatch_nir/FITS/253_111...</td>
      <td>308.85</td>
      <td>435.294250</td>
      <td>UT20-1-008</td>
      <td>5.103958</td>
    </tr>
    <tr>
      <th>38</th>
      <td>GJ_173</td>
      <td>../library/20200128_specmatch_nir/FITS/290_HIP...</td>
      <td>649.65</td>
      <td>242.362946</td>
      <td>UT20-1-008</td>
      <td>-6.787271</td>
    </tr>
    <tr>
      <th>39</th>
      <td>GJ_15_B</td>
      <td>../library/20200128_specmatch_nir/FITS/254_132...</td>
      <td>649.65</td>
      <td>457.891266</td>
      <td>PSU19-1-013</td>
      <td>10.960822</td>
    </tr>
    <tr>
      <th>40</th>
      <td>GJ_273</td>
      <td>../library/20200128_specmatch_nir/FITS/269_GL2...</td>
      <td>330.15</td>
      <td>503.425354</td>
      <td>HET18-3-300</td>
      <td>18.145497</td>
    </tr>
    <tr>
      <th>41</th>
      <td>HD_35112</td>
      <td>../library/20200128_specmatch_nir/FITS/123_351...</td>
      <td>330.15</td>
      <td>334.198303</td>
      <td>UT19-3-008</td>
      <td>38.575472</td>
    </tr>
    <tr>
      <th>42</th>
      <td>V*_V2689_Ori</td>
      <td>../library/20200128_specmatch_nir/FITS/260_245...</td>
      <td>330.15</td>
      <td>314.954285</td>
      <td>UT19-3-008</td>
      <td>22.095539</td>
    </tr>
    <tr>
      <th>43</th>
      <td>GJ_1172</td>
      <td>../library/20200128_specmatch_nir/FITS/250_HIP...</td>
      <td>969.15</td>
      <td>431.379913</td>
      <td>UT20-1-008</td>
      <td>15.009398</td>
    </tr>
    <tr>
      <th>44</th>
      <td>GJ_382</td>
      <td>../library/20200128_specmatch_nir/FITS/270_GL3...</td>
      <td>330.15</td>
      <td>447.286896</td>
      <td>UT20-1-008</td>
      <td>7.907094</td>
    </tr>
    <tr>
      <th>45</th>
      <td>GJ_251</td>
      <td>../library/20200128_specmatch_nir/FITS/261_265...</td>
      <td>617.70</td>
      <td>568.272217</td>
      <td>UT20-1-007</td>
      <td>22.899544</td>
    </tr>
    <tr>
      <th>46</th>
      <td>GJ_725_B</td>
      <td>../library/20200128_specmatch_nir/FITS/341_173...</td>
      <td>298.20</td>
      <td>618.789978</td>
      <td>PSU19-1-013</td>
      <td>-0.719776</td>
    </tr>
    <tr>
      <th>47</th>
      <td>GJ_338_B</td>
      <td>../library/20200128_specmatch_nir/FITS/393_GL3...</td>
      <td>127.80</td>
      <td>277.841675</td>
      <td>UT19-3-008</td>
      <td>12.475246</td>
    </tr>
    <tr>
      <th>48</th>
      <td>GJ_105_B</td>
      <td>../library/20200128_specmatch_nir/FITS/264_GL1...</td>
      <td>969.15</td>
      <td>488.130493</td>
      <td>PSU20-1-001</td>
      <td>26.169476</td>
    </tr>
    <tr>
      <th>49</th>
      <td>GJ_3860</td>
      <td>../library/20200128_specmatch_nir/FITS/30_1283...</td>
      <td>308.85</td>
      <td>398.791565</td>
      <td>UT20-1-008</td>
      <td>-8.854523</td>
    </tr>
    <tr>
      <th>50</th>
      <td>NLTT_26380</td>
      <td>../library/20200128_specmatch_nir/FITS/176_966...</td>
      <td>649.65</td>
      <td>313.364990</td>
      <td>UT19-3-008</td>
      <td>-35.575660</td>
    </tr>
    <tr>
      <th>51</th>
      <td>BD+00_444</td>
      <td>../library/20200128_specmatch_nir/FITS/236_HIP...</td>
      <td>969.15</td>
      <td>345.286316</td>
      <td>UT19-3-008</td>
      <td>73.395850</td>
    </tr>
    <tr>
      <th>52</th>
      <td>HD_61606</td>
      <td>../library/20200128_specmatch_nir/FITS/148_616...</td>
      <td>244.95</td>
      <td>423.264404</td>
      <td>UT19-3-008</td>
      <td>-17.725577</td>
    </tr>
    <tr>
      <th>53</th>
      <td>BD+49_2126</td>
      <td>../library/20200128_specmatch_nir/FITS/302_HIP...</td>
      <td>969.15</td>
      <td>291.579651</td>
      <td>UT19-3-008</td>
      <td>-13.927847</td>
    </tr>
    <tr>
      <th>54</th>
      <td>GJ_3358</td>
      <td>../library/20200128_specmatch_nir/FITS/125_370...</td>
      <td>330.15</td>
      <td>266.640015</td>
      <td>UT19-3-008</td>
      <td>-45.368845</td>
    </tr>
  </tbody>
</table>
</div>



## Reading in Library information 


```python
# Read in required information on all of the 
# This has the Teff, Fe/H, and logg for all of the stars
# OBJECT_ID is the HPF name of the star
df_lib = pd.read_csv(LIBRARY_DIR+'20200128_specmatch_nir.csv')
df_lib
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
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Teff</th>
      <th>e_Teff</th>
      <th>R*</th>
      <th>e_R*</th>
      <th>log(g)</th>
      <th>e_log(g)</th>
      <th>[Fe/H]</th>
      <th>e_[Fe/H]</th>
      <th>...</th>
      <th>simbadnames_x</th>
      <th>ID_NAME</th>
      <th>OBJECT_ID</th>
      <th>filename</th>
      <th>exptime</th>
      <th>sn18</th>
      <th>qprog</th>
      <th>rv</th>
      <th>basenames</th>
      <th>simbadnames_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>HD_110463</td>
      <td>4906</td>
      <td>60</td>
      <td>0.760</td>
      <td>0.030</td>
      <td>4.55</td>
      <td>0.05</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>...</td>
      <td>SPOCS_2843|V*_NP_UMa|NSV_19460|AC2000_1833644|...</td>
      <td>Gaia_DR2_1571411233756165248</td>
      <td>GJ_3743</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>514.133606</td>
      <td>UT20-1-008</td>
      <td>-9.188611</td>
      <td>Slope-20191203T123535_R01.optimal.fits</td>
      <td>SPOCS_2843|V*_NP_UMa|NSV_19460|AC2000_1833644|...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>HD_110833</td>
      <td>4999</td>
      <td>60</td>
      <td>0.800</td>
      <td>0.040</td>
      <td>4.51</td>
      <td>0.05</td>
      <td>0.18</td>
      <td>0.05</td>
      <td>...</td>
      <td>SBC9_3625|PLX_2933|SPOCS_2845|LSPM_J1244+5145|...</td>
      <td>Gaia_DR2_1568219729458240128</td>
      <td>GJ_483</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>202.35</td>
      <td>438.896149</td>
      <td>UT20-1-008</td>
      <td>10.457191</td>
      <td>Slope-20191221T105911_R01.optimal.fits</td>
      <td>SBC9_3625|PLX_2933|SPOCS_2845|LSPM_J1244+5145|...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>HD_112914</td>
      <td>4816</td>
      <td>60</td>
      <td>0.700</td>
      <td>0.030</td>
      <td>4.51</td>
      <td>0.05</td>
      <td>-0.26</td>
      <td>0.05</td>
      <td>...</td>
      <td>SBC9_1742|LSPM_J1259+4159|ASCC__409814|UCAC2__...</td>
      <td>Gaia_DR2_1527631807474248448</td>
      <td>NLTT_32537</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>330.296814</td>
      <td>UT20-1-008</td>
      <td>27.064545</td>
      <td>Slope-20200108T094852_R01.optimal.fits</td>
      <td>SBC9_1742|LSPM_J1259+4159|ASCC__409814|UCAC2__...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>HD_128311</td>
      <td>4918</td>
      <td>60</td>
      <td>0.790</td>
      <td>0.040</td>
      <td>4.49</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.05</td>
      <td>...</td>
      <td>Gaia_DR2_1176209886733406592|LSPM_J1436+0944|T...</td>
      <td>Gaia_DR2_1176209886733406592</td>
      <td>GJ_3860</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>308.85</td>
      <td>398.791565</td>
      <td>UT20-1-008</td>
      <td>-8.854523</td>
      <td>Slope-20200105T125333_R01.optimal.fits</td>
      <td>Gaia_DR2_1176209886733406592|LSPM_J1436+0944|T...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>HD_144872</td>
      <td>4781</td>
      <td>60</td>
      <td>0.690</td>
      <td>0.030</td>
      <td>4.52</td>
      <td>0.05</td>
      <td>-0.25</td>
      <td>0.05</td>
      <td>...</td>
      <td>PLX_3654|SPOCS_2917|LSPM_J1606+3837|ASCC__4998...</td>
      <td>Gaia_DR2_1379416712336909440</td>
      <td>GJ_612</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>246.984924</td>
      <td>UT20-1-008</td>
      <td>23.992700</td>
      <td>Slope-20200112T124748_R01.optimal.fits</td>
      <td>PLX_3654|SPOCS_2917|LSPM_J1606+3837|ASCC__4998...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>HD_24238</td>
      <td>4951</td>
      <td>60</td>
      <td>0.670</td>
      <td>0.030</td>
      <td>4.54</td>
      <td>0.05</td>
      <td>-0.45</td>
      <td>0.05</td>
      <td>...</td>
      <td>2MASS_J03550380+6110006|PLX__848|LSPM_J0355+61...</td>
      <td>Gaia_DR2_474648763011926272</td>
      <td>HD_24238</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>469.626221</td>
      <td>UT19-3-008</td>
      <td>39.301968</td>
      <td>Slope-20191118T052618_R01.optimal.fits</td>
      <td>2MASS_J03550380+6110006|PLX__848|LSPM_J0355+61...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>HD_35112</td>
      <td>4895</td>
      <td>60</td>
      <td>0.760</td>
      <td>0.040</td>
      <td>4.46</td>
      <td>0.05</td>
      <td>-0.08</td>
      <td>0.05</td>
      <td>...</td>
      <td>uvby98_100035112|**_A_2641|1RXS_J052238.8+0236...</td>
      <td>Gaia_DR2_3234412606443085824</td>
      <td>HD_35112</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>334.198303</td>
      <td>UT19-3-008</td>
      <td>38.575472</td>
      <td>Slope-20191117T070203_R01.optimal.fits</td>
      <td>uvby98_100035112|**_A_2641|1RXS_J052238.8+0236...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>HD_37008</td>
      <td>4980</td>
      <td>60</td>
      <td>0.690</td>
      <td>0.030</td>
      <td>4.53</td>
      <td>0.05</td>
      <td>-0.42</td>
      <td>0.05</td>
      <td>...</td>
      <td>2MASS_J05381191+5126445|PLX_1274|LSPM_J0538+51...</td>
      <td>Gaia_DR2_215395053733297024</td>
      <td>GJ_3358</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>266.640015</td>
      <td>UT19-3-008</td>
      <td>-45.368845</td>
      <td>Slope-20191127T053426_R01.optimal.fits</td>
      <td>2MASS_J05381191+5126445|PLX_1274|LSPM_J0538+51...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>HD_53927</td>
      <td>4899</td>
      <td>60</td>
      <td>0.700</td>
      <td>0.030</td>
      <td>4.56</td>
      <td>0.05</td>
      <td>-0.29</td>
      <td>0.05</td>
      <td>...</td>
      <td>2MASS_J07080426+2950047|AG+29__830|SPOCS_2685|...</td>
      <td>Gaia_DR2_884951489919636352</td>
      <td>GJ_3428</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>349.262207</td>
      <td>UT19-3-008</td>
      <td>20.182470</td>
      <td>Slope-20191125T122553_R01.optimal.fits</td>
      <td>2MASS_J07080426+2950047|AG+29__830|SPOCS_2685|...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>HD_61606</td>
      <td>4918</td>
      <td>60</td>
      <td>0.770</td>
      <td>0.030</td>
      <td>4.53</td>
      <td>0.05</td>
      <td>0.09</td>
      <td>0.05</td>
      <td>...</td>
      <td>PLX_1809|V*_V869_Mon|BD-03__2001|CCDM_J07400-0...</td>
      <td>Gaia_DR2_3057712223051571200</td>
      <td>HD_61606</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>244.95</td>
      <td>423.264404</td>
      <td>UT19-3-008</td>
      <td>-17.725577</td>
      <td>Slope-20191118T101626_R01.optimal.fits</td>
      <td>PLX_1809|V*_V869_Mon|BD-03__2001|CCDM_J07400-0...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>HD_87883</td>
      <td>4942</td>
      <td>60</td>
      <td>0.780</td>
      <td>0.040</td>
      <td>4.50</td>
      <td>0.05</td>
      <td>0.13</td>
      <td>0.05</td>
      <td>...</td>
      <td>AG+34_1066|BD+34__2089|GSC_02506-00894|HD__878...</td>
      <td>Gaia_DR2_747266452000257280</td>
      <td>HD_87883</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>338.629211</td>
      <td>UT19-3-008</td>
      <td>9.826486</td>
      <td>Slope-20191118T102934_R01.optimal.fits</td>
      <td>AG+34_1066|BD+34__2089|GSC_02506-00894|HD__878...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>HD_96612</td>
      <td>4862</td>
      <td>60</td>
      <td>0.730</td>
      <td>0.030</td>
      <td>4.53</td>
      <td>0.05</td>
      <td>-0.11</td>
      <td>0.05</td>
      <td>...</td>
      <td>SPOCS_2778|LSPM_J1108+3825|TYC_3010-596-1|ASCC...</td>
      <td>Gaia_DR2_764805169996366720</td>
      <td>NLTT_26380</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>313.364990</td>
      <td>UT19-3-008</td>
      <td>-35.575660</td>
      <td>Slope-20191201T102853_R01.optimal.fits</td>
      <td>SPOCS_2778|LSPM_J1108+3825|TYC_3010-596-1|ASCC...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>HIP_12493</td>
      <td>4276</td>
      <td>193</td>
      <td>0.662</td>
      <td>0.047</td>
      <td>4.68</td>
      <td>0.04</td>
      <td>-0.29</td>
      <td>0.10</td>
      <td>...</td>
      <td>2MASS_J02404288+0111554|Pul_-3__230109|LSPM_J0...</td>
      <td>Gaia_DR2_2501948402746099456</td>
      <td>BD+00_444</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>345.286316</td>
      <td>UT19-3-008</td>
      <td>73.395850</td>
      <td>Slope-20191116T070359_R01.optimal.fits</td>
      <td>2MASS_J02404288+0111554|Pul_-3__230109|LSPM_J0...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>HIP_25220</td>
      <td>4505</td>
      <td>179</td>
      <td>0.702</td>
      <td>0.042</td>
      <td>4.63</td>
      <td>0.04</td>
      <td>0.05</td>
      <td>0.10</td>
      <td>...</td>
      <td>PLX_1222|LSPM_J0523+1719|TYC_1300-284-1|ASCC__...</td>
      <td>Gaia_DR2_3394298532176344960</td>
      <td>*_111_Tau_B</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>425.017181</td>
      <td>UT19-3-008</td>
      <td>38.899927</td>
      <td>Slope-20191118T105426_R01.optimal.fits</td>
      <td>PLX_1222|LSPM_J0523+1719|TYC_1300-284-1|ASCC__...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>HIP_36551</td>
      <td>4395</td>
      <td>193</td>
      <td>0.669</td>
      <td>0.046</td>
      <td>4.66</td>
      <td>0.04</td>
      <td>-0.30</td>
      <td>0.10</td>
      <td>...</td>
      <td>LSPM_J0731+1436|TYC__776-242-1|ASCC__940588|UC...</td>
      <td>Gaia_DR2_3165287719154543232</td>
      <td>GJ_276</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>438.571594</td>
      <td>UT19-3-008</td>
      <td>66.563203</td>
      <td>Slope-20191201T115615_R01.optimal.fits</td>
      <td>LSPM_J0731+1436|TYC__776-242-1|ASCC__940588|UC...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>HIP_40910</td>
      <td>4159</td>
      <td>228</td>
      <td>0.633</td>
      <td>0.053</td>
      <td>4.67</td>
      <td>0.05</td>
      <td>-0.06</td>
      <td>0.10</td>
      <td>...</td>
      <td>PLX_1982|LSPM_J0820+1404|TYC__807-557-1|ASCC__...</td>
      <td>Gaia_DR2_652005932802958976</td>
      <td>GJ_3494</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>297.302368</td>
      <td>UT19-3-008</td>
      <td>7.644187</td>
      <td>Slope-20191125T084533_R01.optimal.fits</td>
      <td>PLX_1982|LSPM_J0820+1404|TYC__807-557-1|ASCC__...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>HIP_47201</td>
      <td>4207</td>
      <td>212</td>
      <td>0.696</td>
      <td>0.054</td>
      <td>4.66</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>...</td>
      <td>LSPM_J0937+2241|TYC_1959-1522-1|ASCC__773684|U...</td>
      <td>Gaia_DR2_641154096631794432</td>
      <td>BD+23_2121</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>263.068237</td>
      <td>UT19-3-008</td>
      <td>-35.743421</td>
      <td>Slope-20191116T101405_R01.optimal.fits</td>
      <td>LSPM_J0937+2241|TYC_1959-1522-1|ASCC__773684|U...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>HIP_48411</td>
      <td>4357</td>
      <td>199</td>
      <td>0.742</td>
      <td>0.072</td>
      <td>4.63</td>
      <td>0.04</td>
      <td>0.20</td>
      <td>0.10</td>
      <td>...</td>
      <td>PLX_2336.1|LSPM_J0952+0313|TYC__240-2143-1|ASC...</td>
      <td>Gaia_DR2_3847128380281953664</td>
      <td>HD_85488</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>373.552734</td>
      <td>UT19-3-008</td>
      <td>22.078667</td>
      <td>Slope-20191118T112544_R01.optimal.fits</td>
      <td>PLX_2336.1|LSPM_J0952+0313|TYC__240-2143-1|ASC...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>HIP_54810</td>
      <td>4395</td>
      <td>185</td>
      <td>0.681</td>
      <td>0.044</td>
      <td>4.64</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.10</td>
      <td>...</td>
      <td>LSPM_J1113+0428|TYC__266-586-1|ASCC_1135032|UC...</td>
      <td>Gaia_DR2_3815264842546968192</td>
      <td>GJ_418</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>360.558899</td>
      <td>UT19-3-008</td>
      <td>17.023210</td>
      <td>Slope-20191125T120712_R01.optimal.fits</td>
      <td>LSPM_J1113+0428|TYC__266-586-1|ASCC_1135032|UC...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>HIP_66222</td>
      <td>3904</td>
      <td>266</td>
      <td>0.667</td>
      <td>0.087</td>
      <td>4.70</td>
      <td>0.07</td>
      <td>-0.11</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J13343+046|PLX_3099.1|LSPM_J1334+0440|AS...</td>
      <td>Gaia_DR2_3714667225186824064</td>
      <td>GJ_1172</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>431.379913</td>
      <td>UT20-1-008</td>
      <td>15.009398</td>
      <td>Slope-20200114T110022_R01.optimal.fits</td>
      <td>Karmn_J13343+046|PLX_3099.1|LSPM_J1334+0440|AS...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>HD_111631</td>
      <td>3989</td>
      <td>60</td>
      <td>0.646</td>
      <td>0.020</td>
      <td>4.66</td>
      <td>0.05</td>
      <td>0.24</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_2951|AG-00_1774|BD+00__2989|Ci_18_1633|DO_...</td>
      <td>Gaia_DR2_3689602277083844480</td>
      <td>GJ_488</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>308.85</td>
      <td>435.294250</td>
      <td>UT20-1-008</td>
      <td>5.103958</td>
      <td>Slope-20200113T110819_R01.optimal.fits</td>
      <td>PLX_2951|AG-00_1774|BD+00__2989|Ci_18_1633|DO_...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>HD_1326B</td>
      <td>3218</td>
      <td>60</td>
      <td>0.192</td>
      <td>0.008</td>
      <td>5.07</td>
      <td>0.05</td>
      <td>-0.30</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J00184+440|ASCC__373580|2MASS_J00182549+...</td>
      <td>Gaia_DR2_385334196532776576</td>
      <td>GJ_15_B</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>457.891266</td>
      <td>PSU19-1-013</td>
      <td>10.960822</td>
      <td>Slope-20181224T033526_R01.optimal.fits</td>
      <td>Karmn_J00184+440|ASCC__373580|2MASS_J00182549+...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>HD_232979</td>
      <td>3929</td>
      <td>60</td>
      <td>0.608</td>
      <td>0.020</td>
      <td>4.68</td>
      <td>0.05</td>
      <td>-0.11</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J04376+528|2MASS_J04374092+5253372|PLX_1...</td>
      <td>Gaia_DR2_272855565762295680</td>
      <td>HD_232979</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>386.030884</td>
      <td>UT19-3-008</td>
      <td>34.414724</td>
      <td>Slope-20191116T103941_R01.optimal.fits</td>
      <td>Karmn_J04376+528|2MASS_J04374092+5253372|PLX_1...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>HD_245409</td>
      <td>3966</td>
      <td>60</td>
      <td>0.601</td>
      <td>0.020</td>
      <td>4.69</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J05365+113|2MASS_J05363099+1119401|2RE_J...</td>
      <td>Gaia_DR2_3339921875389105152</td>
      <td>V*_V2689_Ori</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>314.954285</td>
      <td>UT19-3-008</td>
      <td>22.095539</td>
      <td>Slope-20191116T105902_R01.optimal.fits</td>
      <td>Karmn_J05365+113|2MASS_J05363099+1119401|2RE_J...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>HD_265866</td>
      <td>3448</td>
      <td>60</td>
      <td>0.358</td>
      <td>0.013</td>
      <td>4.88</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J06548+332|2MASS_J06544902+3316058|IRAS_...</td>
      <td>Gaia_DR2_939072613334579328</td>
      <td>GJ_251</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>617.70</td>
      <td>568.272217</td>
      <td>UT20-1-007</td>
      <td>22.899544</td>
      <td>Slope-20200114T091114_R01.optimal.fits</td>
      <td>Karmn_J06548+332|2MASS_J06544902+3316058|IRAS_...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>HD_28343</td>
      <td>4124</td>
      <td>62</td>
      <td>0.687</td>
      <td>0.023</td>
      <td>4.64</td>
      <td>0.05</td>
      <td>0.39</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J04290+219|PM_J04290+2155|2MASS_J0429001...</td>
      <td>Gaia_DR2_145421309108301184</td>
      <td>HD_28343</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>287.55</td>
      <td>430.919281</td>
      <td>UT19-3-008</td>
      <td>-34.946848</td>
      <td>Slope-20191119T101149_R01.optimal.fits</td>
      <td>Karmn_J04290+219|PM_J04290+2155|2MASS_J0429001...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>GL_105B</td>
      <td>3284</td>
      <td>60</td>
      <td>0.278</td>
      <td>0.010</td>
      <td>4.94</td>
      <td>0.05</td>
      <td>-0.12</td>
      <td>0.03</td>
      <td>...</td>
      <td>Karmn_J02362+068|LSPM_J0236+0652|2MASS_J023615...</td>
      <td>Gaia_DR2_18565464288396416</td>
      <td>GJ_105_B</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>488.130493</td>
      <td>PSU20-1-001</td>
      <td>26.169476</td>
      <td>Slope-20200108T040109_R01.optimal.fits</td>
      <td>Karmn_J02362+068|LSPM_J0236+0652|2MASS_J023615...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>GL_109</td>
      <td>3405</td>
      <td>60</td>
      <td>0.364</td>
      <td>0.014</td>
      <td>4.85</td>
      <td>0.05</td>
      <td>-0.10</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J02442+255|PLX__555|LSPM_J0244+2531|ASCC...</td>
      <td>Gaia_DR2_114207651462714880</td>
      <td>GJ_109</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>501.315796</td>
      <td>PSU19-1-013</td>
      <td>30.515882</td>
      <td>Slope-20181221T061620_R01.optimal.fits</td>
      <td>Karmn_J02442+255|PLX__555|LSPM_J0244+2531|ASCC...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>GL_2066</td>
      <td>3500</td>
      <td>60</td>
      <td>0.461</td>
      <td>0.017</td>
      <td>4.77</td>
      <td>0.05</td>
      <td>-0.12</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J08161+013|LSPM_J0816+0118|TYC__196-1309...</td>
      <td>Gaia_DR2_3089711447388931584</td>
      <td>GJ_2066</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>499.474548</td>
      <td>UT20-1-008</td>
      <td>62.185778</td>
      <td>Slope-20200107T065409_R01.optimal.fits</td>
      <td>Karmn_J08161+013|LSPM_J0816+0118|TYC__196-1309...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>GL_239</td>
      <td>3801</td>
      <td>60</td>
      <td>0.423</td>
      <td>0.015</td>
      <td>4.86</td>
      <td>0.05</td>
      <td>-0.34</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_1538|LSPM_J0637+1733|ASCC__848699|UCAC2__3...</td>
      <td>Gaia_DR2_3359074685047632640</td>
      <td>HD_260655</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>567.503235</td>
      <td>HET19-3-300</td>
      <td>-58.205038</td>
      <td>Slope-20191115T121100_R01.optimal.fits</td>
      <td>PLX_1538|LSPM_J0637+1733|ASCC__848699|UCAC2__3...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>GL_273</td>
      <td>3317</td>
      <td>60</td>
      <td>0.315</td>
      <td>0.012</td>
      <td>4.89</td>
      <td>0.05</td>
      <td>-0.11</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J07274+052|Gaia_DR2_3139847906304421632|...</td>
      <td>Gaia_DR2_3139847906304421632</td>
      <td>GJ_273</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>503.425354</td>
      <td>HET18-3-300</td>
      <td>18.145497</td>
      <td>Slope-20181104T095012_R01.optimal.fits</td>
      <td>Karmn_J07274+052|Gaia_DR2_3139847906304421632|...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>GL_382</td>
      <td>3623</td>
      <td>60</td>
      <td>0.522</td>
      <td>0.019</td>
      <td>4.72</td>
      <td>0.05</td>
      <td>0.13</td>
      <td>0.08</td>
      <td>...</td>
      <td>8pc_127.99|BD-03__2870|G_162-25|G__53-29|GCRV_...</td>
      <td>Gaia_DR2_3828238392559860992</td>
      <td>GJ_382</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>447.286896</td>
      <td>UT20-1-008</td>
      <td>7.907094</td>
      <td>Slope-20191207T123306_R01.optimal.fits</td>
      <td>8pc_127.99|BD-03__2870|G_162-25|G__53-29|GCRV_...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>GL_393</td>
      <td>3548</td>
      <td>60</td>
      <td>0.420</td>
      <td>0.016</td>
      <td>4.82</td>
      <td>0.05</td>
      <td>-0.18</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_2456|LSPM_J1028+0050|ASCC_1133482|UCAC2__3...</td>
      <td>Gaia_DR2_3855208897392952192</td>
      <td>GJ_393</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>436.65</td>
      <td>427.817566</td>
      <td>UT20-1-008</td>
      <td>8.299900</td>
      <td>Slope-20200114T104423_R01.optimal.fits</td>
      <td>PLX_2456|LSPM_J1028+0050|ASCC_1133482|UCAC2__3...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>GL_514</td>
      <td>3727</td>
      <td>61</td>
      <td>0.483</td>
      <td>0.016</td>
      <td>4.79</td>
      <td>0.05</td>
      <td>-0.09</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_3079|LSPM_J1329+1022|TYC__895-317-1|ASCC__...</td>
      <td>Gaia_DR2_3738099879558957952</td>
      <td>GJ_514</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>339.472290</td>
      <td>UT20-1-008</td>
      <td>14.595074</td>
      <td>Slope-20191216T122849_R01.optimal.fits</td>
      <td>PLX_3079|LSPM_J1329+1022|TYC__895-317-1|ASCC__...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>GL_625</td>
      <td>3475</td>
      <td>60</td>
      <td>0.331</td>
      <td>0.012</td>
      <td>4.90</td>
      <td>0.05</td>
      <td>-0.35</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J16254+543|Gaia_DR2_1428427236986209024|...</td>
      <td>Gaia_DR2_1428427236986209024</td>
      <td>GJ_625</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>539.909912</td>
      <td>UT20-1-008</td>
      <td>-13.061425</td>
      <td>Slope-20200124T122211_R01.optimal.fits</td>
      <td>Karmn_J16254+543|Gaia_DR2_1428427236986209024|...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>GL_83.1</td>
      <td>3080</td>
      <td>60</td>
      <td>0.187</td>
      <td>0.010</td>
      <td>5.05</td>
      <td>0.05</td>
      <td>-0.16</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J02002+130|LSPM_J0200+1303|UCAC2__364034...</td>
      <td>Gaia_DR2_76868614540049408</td>
      <td>GJ_9066</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>484.133362</td>
      <td>PSU19-3-014</td>
      <td>-28.698145</td>
      <td>Slope-20190915T111342_R01.optimal.fits</td>
      <td>Karmn_J02002+130|LSPM_J0200+1303|UCAC2__364034...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>GL_87</td>
      <td>3638</td>
      <td>62</td>
      <td>0.443</td>
      <td>0.017</td>
      <td>4.79</td>
      <td>0.05</td>
      <td>-0.36</td>
      <td>0.08</td>
      <td>...</td>
      <td>WDS_J02123+0335A|**_PIN____3A|PLX__450|LSPM_J0...</td>
      <td>Gaia_DR2_2515037264041041536</td>
      <td>GJ_87</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>381.879883</td>
      <td>UT20-1-008</td>
      <td>-2.580083</td>
      <td>Slope-20191207T055134_R01.optimal.fits</td>
      <td>WDS_J02123+0335A|**_PIN____3A|PLX__450|LSPM_J0...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>HIP_11048</td>
      <td>3785</td>
      <td>62</td>
      <td>0.599</td>
      <td>0.021</td>
      <td>4.67</td>
      <td>0.05</td>
      <td>0.14</td>
      <td>0.08</td>
      <td>...</td>
      <td>2MASS_J02221463+4752481|PLX__481|LSPM_J0222+47...</td>
      <td>Gaia_DR2_354077348697687424</td>
      <td>GJ_96</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>460.026642</td>
      <td>UT20-1-008</td>
      <td>-37.914717</td>
      <td>Slope-20191208T013316_R01.optimal.fits</td>
      <td>2MASS_J02221463+4752481|PLX__481|LSPM_J0222+47...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>HIP_15366</td>
      <td>3700</td>
      <td>61</td>
      <td>0.628</td>
      <td>0.031</td>
      <td>4.65</td>
      <td>0.05</td>
      <td>0.53</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J03181+382|2MASS_J03180742+3815081|PLX__...</td>
      <td>Gaia_DR2_235383247412615936</td>
      <td>GJ_134</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>480.095947</td>
      <td>UT20-1-008</td>
      <td>-4.232166</td>
      <td>Slope-20191224T011605_R01.optimal.fits</td>
      <td>Karmn_J03181+382|2MASS_J03180742+3815081|PLX__...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>HIP_21556</td>
      <td>3671</td>
      <td>61</td>
      <td>0.444</td>
      <td>0.017</td>
      <td>4.82</td>
      <td>0.05</td>
      <td>-0.04</td>
      <td>0.08</td>
      <td>...</td>
      <td>Karmn_J04376-110|2MASS_J04374188-1102198|PLX_1...</td>
      <td>Gaia_DR2_3184351876391975808</td>
      <td>GJ_173</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>242.362946</td>
      <td>UT20-1-008</td>
      <td>-6.787271</td>
      <td>Slope-20191213T060232_R01.optimal.fits</td>
      <td>Karmn_J04376-110|2MASS_J04374188-1102198|PLX_1...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40</td>
      <td>HIP_47513</td>
      <td>3500</td>
      <td>60</td>
      <td>0.485</td>
      <td>0.019</td>
      <td>4.74</td>
      <td>0.05</td>
      <td>-0.05</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_2298|AC_+13__1301|LSPM_J0941+1312|ASCC__94...</td>
      <td>Gaia_DR2_614543647497149056</td>
      <td>GJ_361</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>242.642822</td>
      <td>UT20-1-008</td>
      <td>11.463286</td>
      <td>Slope-20191219T124524_R01.optimal.fits</td>
      <td>PLX_2298|AC_+13__1301|LSPM_J0941+1312|ASCC__94...</td>
    </tr>
    <tr>
      <th>41</th>
      <td>41</td>
      <td>HIP_57548</td>
      <td>3192</td>
      <td>60</td>
      <td>0.197</td>
      <td>0.008</td>
      <td>5.08</td>
      <td>0.05</td>
      <td>-0.02</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_2730|LSPM_J1147+0048|ASCC_1136005|UCAC2__3...</td>
      <td>Gaia_DR2_3796072592206250624</td>
      <td>GJ_447</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>438.552124</td>
      <td>PSU19-1-013</td>
      <td>-31.144416</td>
      <td>Slope-20190319T054134_R01.optimal.fits</td>
      <td>PLX_2730|LSPM_J1147+0048|ASCC_1136005|UCAC2__3...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>42</td>
      <td>HIP_59748</td>
      <td>3900</td>
      <td>61</td>
      <td>0.669</td>
      <td>0.038</td>
      <td>4.63</td>
      <td>0.06</td>
      <td>0.14</td>
      <td>0.08</td>
      <td>...</td>
      <td>LSPM_J1215+4843|ASCC__329726|USNO-B1.0_1387-00...</td>
      <td>Gaia_DR2_1545795529110395136</td>
      <td>BD+49_2126</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>291.579651</td>
      <td>UT19-3-008</td>
      <td>-13.927847</td>
      <td>Slope-20191121T122807_R01.optimal.fits</td>
      <td>LSPM_J1215+4843|ASCC__329726|USNO-B1.0_1387-00...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>43</td>
      <td>HIP_60093</td>
      <td>3993</td>
      <td>60</td>
      <td>0.643</td>
      <td>0.037</td>
      <td>4.67</td>
      <td>0.06</td>
      <td>0.46</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_2838|GJ__9404|LSPM_J1219+2822|ASCC__684708...</td>
      <td>Gaia_DR2_4010201828880793984</td>
      <td>BD+29_2279</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>379.905640</td>
      <td>PSU19-3-009</td>
      <td>-0.648506</td>
      <td>Slope-20191123T122518_R01.optimal.fits</td>
      <td>PLX_2838|GJ__9404|LSPM_J1219+2822|ASCC__684708...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>44</td>
      <td>HIP_65016</td>
      <td>3650</td>
      <td>60</td>
      <td>0.598</td>
      <td>0.025</td>
      <td>4.66</td>
      <td>0.05</td>
      <td>0.40</td>
      <td>0.08</td>
      <td>...</td>
      <td>PLX_3047.1|CSI+33-13174|LSPM_J1319+3320|TYC_25...</td>
      <td>Gaia_DR2_1472718211053416320</td>
      <td>NLTT_33716</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>969.15</td>
      <td>376.039368</td>
      <td>UT20-1-008</td>
      <td>-11.699894</td>
      <td>Slope-20200114T095403_R01.optimal.fits</td>
      <td>PLX_3047.1|CSI+33-13174|LSPM_J1319+3320|TYC_25...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45</td>
      <td>HD_119850</td>
      <td>3618</td>
      <td>31</td>
      <td>0.484</td>
      <td>0.008</td>
      <td>4.78</td>
      <td>0.02</td>
      <td>-0.30</td>
      <td>0.10</td>
      <td>...</td>
      <td>PM_J13457+1453|PLX_3135|LSPM_J1345+1453|ASCC__...</td>
      <td>Gaia_DR2_3741297293732404352</td>
      <td>GJ_526</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>181.05</td>
      <td>482.063538</td>
      <td>UT20-1-008</td>
      <td>15.801779</td>
      <td>Slope-20191222T122719_R01.optimal.fits</td>
      <td>PM_J13457+1453|PLX_3135|LSPM_J1345+1453|ASCC__...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>46</td>
      <td>HD_173740</td>
      <td>3104</td>
      <td>28</td>
      <td>0.323</td>
      <td>0.006</td>
      <td>4.92</td>
      <td>0.01</td>
      <td>-0.36</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J18427+596S|ASCC__201777|USNO-B1.0_1496-...</td>
      <td>Gaia_DR2_2154880616772528384</td>
      <td>GJ_725_B</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>298.20</td>
      <td>618.789978</td>
      <td>PSU19-1-013</td>
      <td>-0.719776</td>
      <td>Slope-20190316T115749_R01.optimal.fits</td>
      <td>Karmn_J18427+596S|ASCC__201777|USNO-B1.0_1496-...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>47</td>
      <td>HD_36395</td>
      <td>3801</td>
      <td>9</td>
      <td>0.574</td>
      <td>0.004</td>
      <td>4.70</td>
      <td>0.01</td>
      <td>0.35</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J05314-036|PM_J05314-0340|8pc_175.72|PLX...</td>
      <td>Gaia_DR2_3209938366665770752</td>
      <td>GJ_205</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>127.80</td>
      <td>370.626740</td>
      <td>UT20-1-008</td>
      <td>8.690360</td>
      <td>Slope-20191213T062214_R01.optimal.fits</td>
      <td>Karmn_J05314-036|PM_J05314-0340|8pc_175.72|PLX...</td>
    </tr>
    <tr>
      <th>48</th>
      <td>48</td>
      <td>HD_79210</td>
      <td>3907</td>
      <td>35</td>
      <td>0.577</td>
      <td>0.013</td>
      <td>4.70</td>
      <td>0.02</td>
      <td>-0.18</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J09143+526|PM_J09143+5241|PLX_2198|ASCC_...</td>
      <td>Gaia_DR2_1022456139210632064</td>
      <td>GJ_338_A</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>138.45</td>
      <td>425.514191</td>
      <td>UT19-3-008</td>
      <td>12.458512</td>
      <td>Slope-20191130T090648_R01.optimal.fits</td>
      <td>Karmn_J09143+526|PM_J09143+5241|PLX_2198|ASCC_...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>49</td>
      <td>HD_88230</td>
      <td>4085</td>
      <td>14</td>
      <td>0.640</td>
      <td>0.005</td>
      <td>4.64</td>
      <td>0.01</td>
      <td>-0.16</td>
      <td>0.10</td>
      <td>...</td>
      <td>PLX_2390|LSPM_J1011+4927|ASCC__327385|USNO-B1....</td>
      <td>Gaia_DR2_823773494718931968</td>
      <td>HD_88230</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>63.90</td>
      <td>536.356506</td>
      <td>UT19-3-008</td>
      <td>-25.738879</td>
      <td>Slope-20191119T103851_R01.optimal.fits</td>
      <td>PLX_2390|LSPM_J1011+4927|ASCC__327385|USNO-B1....</td>
    </tr>
    <tr>
      <th>50</th>
      <td>50</td>
      <td>HD_95735</td>
      <td>3464</td>
      <td>15</td>
      <td>0.392</td>
      <td>0.003</td>
      <td>4.86</td>
      <td>0.01</td>
      <td>-0.41</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J11033+359|PLX_2576|LSPM_J1103+3558|ASCC...</td>
      <td>HD__95735</td>
      <td>GJ_411</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>63.90</td>
      <td>593.079102</td>
      <td>ENG19-1-003</td>
      <td>-84.711566</td>
      <td>Slope-20190224T102122_R01.optimal.fits</td>
      <td>Karmn_J11033+359|PLX_2576|LSPM_J1103+3558|ASCC...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>51</td>
      <td>GJ_176</td>
      <td>3679</td>
      <td>77</td>
      <td>0.453</td>
      <td>0.022</td>
      <td>4.80</td>
      <td>0.03</td>
      <td>0.15</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J04429+189|2MASS_J04425581+1857285|PLX_1...</td>
      <td>Gaia_DR2_3409711211681795584</td>
      <td>GJ_176</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>330.15</td>
      <td>261.819824</td>
      <td>UT20-1-008</td>
      <td>26.186852</td>
      <td>Slope-20191208T040844_R01.optimal.fits</td>
      <td>Karmn_J04429+189|2MASS_J04425581+1857285|PLX_1...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>52</td>
      <td>GL_338B</td>
      <td>3867</td>
      <td>37</td>
      <td>0.567</td>
      <td>0.014</td>
      <td>4.71</td>
      <td>0.02</td>
      <td>-0.15</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J09144+526|PM_J09144+5241|ASCC__253742|2...</td>
      <td>Gaia_DR2_1022456104850892928</td>
      <td>GJ_338_B</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>127.80</td>
      <td>277.841675</td>
      <td>UT19-3-008</td>
      <td>12.475246</td>
      <td>Slope-20191201T092225_R01.optimal.fits</td>
      <td>Karmn_J09144+526|PM_J09144+5241|ASCC__253742|2...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>53</td>
      <td>GL_699</td>
      <td>3222</td>
      <td>10</td>
      <td>0.187</td>
      <td>0.001</td>
      <td>5.11</td>
      <td>0.01</td>
      <td>-0.39</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J17578+046|PLX_4098|LSPM_J1757+0441|ASCC...</td>
      <td>Gaia_DR2_4472832130942575872</td>
      <td>GJ_699</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>191.70</td>
      <td>622.323425</td>
      <td>ENG19-1-003</td>
      <td>-110.572337</td>
      <td>Slope-20190319T113007_R01.optimal.fits</td>
      <td>Karmn_J17578+046|PLX_4098|LSPM_J1757+0441|ASCC...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>54</td>
      <td>HIP_57087</td>
      <td>3416</td>
      <td>53</td>
      <td>0.455</td>
      <td>0.018</td>
      <td>4.83</td>
      <td>0.03</td>
      <td>0.04</td>
      <td>0.10</td>
      <td>...</td>
      <td>Karmn_J11421+267|TYC_1984-2613-1|PLX_2704.1|1R...</td>
      <td>Gaia_DR2_4017860992519744384</td>
      <td>GJ_436</td>
      <td>../data/FITS/20200128_extending/20200128_exten...</td>
      <td>649.65</td>
      <td>476.864380</td>
      <td>ENG19-1-003</td>
      <td>9.577289</td>
      <td>Slope-20181221T095756_R01.optimal.fits</td>
      <td>Karmn_J11421+267|TYC_1984-2613-1|PLX_2704.1|1R...</td>
    </tr>
  </tbody>
</table>
<p>55 rows × 34 columns</p>
</div>



# Read in example target 


```python
# Target data
targetfilename = '../input/G_9-40/Slope-20190301T024821_R01.optimal.fits'
Htarget = hpfspec.HPFSpectrum(targetfilename,targetname='G 9-40')

# Reference data
Hrefs   = HLS.splist
```

    Reading from file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/G 9-40.config
    could not convert string to float: 'None' File does not exist!
    Querying SIMBAD for data
    Saving to file /home/sejones/anaconda3/lib/python3.8/site-packages/hpfspec-0.1.0-py3.8.egg/hpfspec/data/target_files/G 9-40.config
    ra 134.71804124999997
    dec 21.07616722222222
    pmra 175.512
    pmdec -318.469
    px 35.7781
    rv None
    epoch 2451545.0
    Done
    WARNING: LEAP UPDATE=FALSE
    Barycentric shifting
    AIRTOVAC (TRUE for HARPS, FALSE TRUE HPF):  False
    First iteration:  RVabs = 14.33333km/s
    Second iteration: RVabs = 14.48091km/s, sigma=2.82539
    RVabs it #1: 14.33333+- 0.00000km/s
    RVabs it #2: 14.48091+- 0.00000km/s
    berv=-13.85248087461531,rv=14.48090854621002


# Now run spectral matching algorithm on a few orders 


```python
# Which orders are good in HPF ?
orders = list(hpfspecmatch.BOUNDS.keys())
orders
```




    ['4', '5', '6', '14', '15', '16', '17']



### Run matching algorithm on first 2 orders 


```python
# Bounds to use for different HPF orders (used for knowing where to start/stop resampling the orders)
hpfspecmatch.BOUNDS
```




    {'4': [8540.0, 8640.0],
     '5': [8670.0, 8750.0],
     '6': [8790.0, 8885.0],
     '14': [9940.0, 10055.0],
     '15': [10105.0, 10220.0],
     '16': [10280.0, 10395.0],
     '17': [10460.0, 10570.0]}




```python
# Run spectral matching algorithm for first two orders
# in principle we should run all orders, just first two as an example
for o in orders:
    print("########################")
    print("Order {}".format(o))
    print("########################")
    wmin = hpfspecmatch.BOUNDS[o][0] # Lower wavelength bound in A
    wmax = hpfspecmatch.BOUNDS[o][1] # Upper wavelength bound in A
    ww = np.arange(wmin,wmax,0.01)   # Wavelength array to resample to
    v = np.linspace(-125,125,1501)   # Velocities in km/s to use for absolute RV consideration
    savefolder = '../output/20200730_toi1468/{}/{}_{}/'.format(Htarget.object,Htarget.object,o) # foldername to save
    
    #############################################################
    # Run first Spectral Matching Step: Loop through the full library to find which ones are best
    #############################################################
    df_chi, df_chi_best, Hbest = hpfspecmatch.chi2spectraPolyLoop(ww,            # Wavelength to resample to
                                                                  Htarget,       # Target class
                                                                  HLS.splist,    # Target library spectra
                                                                  plot_all=False,# if True, will create a lot more plots 
                                                                  verbose=True,  # if verbose
                                                                  vsini=True)    # recommend always having on
    
    #############################################################
    # Run the Second step: creating the composite spectrum
    #############################################################
    t,f,l,vis,te,fe,le,df_chi,LCS = hpfspecmatch.run_specmatch(Htarget,   # Target class
                                                               HLS.splist,# Library spectra
                                                               ww,        # Wavelength to resample to
                                                               v,         # velocity range to use for absolute rv
                                                               df_lib,    # dataframe with info on Teff/FeH/logg for the library stars
                                                               savefolder=savefolder)
```

    ########################
    Order 4
    ########################
    Performing first Chebfit
    Found centers: [ 5.01706246e+06 -1.75185777e+03  6.79644248e-02  3.95608230e-06
     -3.45324151e-10  6.69879204e-15]
    With CHI 1.5908446430787733
    7 6
    Optimization terminated successfully.
             Current function value: 1.556158
             Iterations: 431
             Function evaluations: 858
    0 G_9-40 GJ_109 1.5561581825415096
    Performing first Chebfit
    Found centers: [ 1.65811856e+07 -5.78090082e+03  2.23712075e-01  1.30520210e-05
     -1.13645324e-09  2.20115745e-14]
    With CHI 3.844235336712811
    7 6
    Optimization terminated successfully.
             Current function value: 3.828141
             Iterations: 515
             Function evaluations: 983
    1 G_9-40 GJ_411 3.8281408175011844
    Performing first Chebfit
    Found centers: [ 9.92084759e+06 -3.45791544e+03  1.33758806e-01  7.80696174e-06
     -6.79470409e-10  1.31569730e-14]
    With CHI 3.4433372396654516
    7 6
    Optimization terminated successfully.
             Current function value: 3.391509
             Iterations: 526
             Function evaluations: 981
    2 G_9-40 GJ_176 3.391508764784823
    Performing first Chebfit
    Found centers: [ 2.08473270e+07 -7.26903552e+03  2.81349660e-01  1.64121280e-05
     -1.42927012e-09  2.76860182e-14]
    With CHI 4.658322500768743
    7 6
    Optimization terminated successfully.
             Current function value: 4.642392
             Iterations: 424
             Function evaluations: 859
    3 G_9-40 GJ_526 4.642392339285339
    Performing first Chebfit
    Found centers: [ 2.60099748e+07 -9.06773137e+03  3.50879939e-01  2.04728366e-05
     -1.78245414e-09  3.45221325e-14]
    With CHI 7.585069345540184
    7 6
    Optimization terminated successfully.
             Current function value: 7.423810
             Iterations: 588
             Function evaluations: 1097
    4 G_9-40 GJ_134 7.4238099394523385
    Performing first Chebfit
    Found centers: [ 8.28944418e+06 -2.89170892e+03  1.12008950e-01  6.52933212e-06
     -5.69042835e-10  1.10278692e-14]
    With CHI 1.8595009097350004
    7 6
    Optimization terminated successfully.
             Current function value: 1.819800
             Iterations: 431
             Function evaluations: 851
    5 G_9-40 GJ_436 1.8197997359371727
    Performing first Chebfit
    Found centers: [ 1.40321439e+07 -4.89174220e+03  1.89274546e-01  1.10443591e-05
     -9.61500808e-10  1.86212849e-14]
    With CHI 4.388940912627513
    7 6
    Optimization terminated successfully.
             Current function value: 4.321936
             Iterations: 548
             Function evaluations: 1032
    6 G_9-40 GJ_361 4.321935947142886
    Performing first Chebfit
    Found centers: [ 5.02957064e+07 -1.75309889e+04  6.78157254e-01  3.95799626e-05
     -3.44492474e-09  6.67075359e-14]
    With CHI 28.317168344924905
    7 6
    Optimization terminated successfully.
             Current function value: 21.633691
             Iterations: 627
             Function evaluations: 1140
    7 G_9-40 HD_87883 21.63369110118124
    Performing first Chebfit
    Found centers: [-1.31638144e+08  4.59688047e+04 -1.78359843e+00 -1.03808759e-04
      9.06245479e-09 -1.75810875e-13]
    With CHI 110.58568691006302
    7 6
    Optimization terminated successfully.
             Current function value: 100.724338
             Iterations: 865
             Function evaluations: 1590
    8 G_9-40 HD_24238 100.7243378565824
    Performing first Chebfit
    Found centers: [ 2.16060067e+07 -7.53146237e+03  2.91374463e-01  1.70040292e-05
     -1.48014643e-09  2.86635525e-14]
    With CHI 6.1957779408463365
    7 6
    Optimization terminated successfully.
             Current function value: 6.078884
             Iterations: 470
             Function evaluations: 913
    9 G_9-40 NLTT_33716 6.078883792713195
    Performing first Chebfit
    Found centers: [ 4.70098320e+07 -1.63870849e+04  6.33996940e-01  3.69977481e-05
     -3.22063374e-09  6.23698614e-14]
    With CHI 19.595302401390562
    7 6
    Optimization terminated successfully.
             Current function value: 17.512234
             Iterations: 653
             Function evaluations: 1198
    10 G_9-40 HD_28343 17.512234214936186
    Performing first Chebfit
    Found centers: [ 3.15816879e+07 -1.10067909e+04  4.25699898e-01  2.48498158e-05
     -2.16245432e-09  4.18690093e-14]
    With CHI 12.459247303601867
    7 6
    Optimization terminated successfully.
             Current function value: 12.283612
             Iterations: 723
             Function evaluations: 1316
    11 G_9-40 GJ_338_A 12.283612283166251
    Performing first Chebfit
    Found centers: [-2.21973132e+07  7.73730127e+03 -2.99320614e-01 -1.74686700e-05
      1.52050625e-09 -2.94441376e-14]
    With CHI 8.95711318727654
    7 6
    Optimization terminated successfully.
             Current function value: 6.604258
             Iterations: 486
             Function evaluations: 978
    12 G_9-40 GJ_447 6.6042579971381095
    Performing first Chebfit
    Found centers: [ 8.26613067e+06 -2.87831896e+03  1.11159391e-01  6.49759503e-06
     -5.64599536e-10  1.09217571e-14]
    With CHI 3.3217254854921494
    7 6
    Optimization terminated successfully.
             Current function value: 3.302249
             Iterations: 455
             Function evaluations: 877
    13 G_9-40 GJ_625 3.302248601835497
    Performing first Chebfit
    Found centers: [ 2.28122518e+07 -7.95583514e+03  3.08037713e-01  1.79632692e-05
     -1.56488756e-09  3.03194132e-14]
    With CHI 4.9067927639506195
    7 6
    Optimization terminated successfully.
             Current function value: 4.900008
             Iterations: 511
             Function evaluations: 978
    14 G_9-40 GJ_87 4.900007691398814
    Performing first Chebfit
    Found centers: [ 4.80819491e+07 -1.67584622e+04  6.48216993e-01  3.78355492e-05
     -3.29281287e-09  6.37586924e-14]
    With CHI 22.991690813547336
    7 6
    Warning: Maximum number of function evaluations has been exceeded.
    15 G_9-40 GJ_418 20.718031151286077
    Performing first Chebfit
    Found centers: [ 4.19926487e+07 -1.46351937e+04  5.66031452e-01  3.30415995e-05
     -2.87530344e-09  5.56709249e-14]
    With CHI 19.414653943637422
    7 6
    Warning: Maximum number of function evaluations has been exceeded.
    16 G_9-40 NLTT_32537 17.95624976899556
    Performing first Chebfit
    Found centers: [ 1.28177541e+07 -4.46247461e+03  1.72291852e-01  1.00734909e-05
     -8.75085436e-10  1.69250623e-14]
    With CHI 7.912771817414013
    7 6
    Optimization terminated successfully.
             Current function value: 7.901660
             Iterations: 497
             Function evaluations: 963
    17 G_9-40 HD_260655 7.901659642153487
    Performing first Chebfit
    Found centers: [ 2.85622760e+07 -9.95119299e+03  3.84666805e-01  2.24656778e-05
     -1.95393600e-09  3.78191893e-14]
    With CHI 13.39506570413317
    7 6
    Optimization terminated successfully.
             Current function value: 13.171890
             Iterations: 556
             Function evaluations: 1085
    18 G_9-40 HD_232979 13.171890026600492
    Performing first Chebfit
    Found centers: [ 4.72116140e+07 -1.64563221e+04  6.36606033e-01  3.71537565e-05
     -3.23386031e-09  6.26217654e-14]
    With CHI 18.77449802909325
    7 6
    Optimization terminated successfully.
             Current function value: 17.762739
             Iterations: 643
             Function evaluations: 1189
    19 G_9-40 GJ_276 17.762739263248335
    Performing first Chebfit
    Found centers: [ 4.34325447e+07 -1.51394155e+04  5.85683512e-01  3.41806550e-05
     -2.97518936e-09  5.76140257e-14]
    With CHI 19.927420736806233
    7 6
    Warning: Maximum number of function evaluations has been exceeded.
    20 G_9-40 GJ_3428 18.22754014516365
    Performing first Chebfit
    Found centers: [ 5.05691886e+07 -1.76263695e+04  6.81851016e-01  3.97953091e-05
     -3.46369147e-09  6.70712529e-14]
    With CHI 22.269140592137326
    7 6
    Optimization terminated successfully.
             Current function value: 19.606487
             Iterations: 549
             Function evaluations: 1062
    21 G_9-40 BD+23_2121 19.60648727169881
    Performing first Chebfit
    Found centers: [ 1.95701964e+07 -6.81992003e+03  2.63727179e-01  1.53970139e-05
     -1.33965570e-09  2.59356759e-14]
    With CHI 6.017115445063879
    7 6
    Optimization terminated successfully.
             Current function value: 5.998700
             Iterations: 519
             Function evaluations: 965
    22 G_9-40 GJ_514 5.99869961689489
    Performing first Chebfit
    Found centers: [ 3.97965285e+07 -1.38693366e+04  5.36381931e-01  3.13123964e-05
     -2.72468073e-09  5.27528823e-14]
    With CHI 16.87748725165027
    7 6


### Plots should then be saved in the 'savefolder'


```python

```
