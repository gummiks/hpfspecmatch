import os
import astropy.time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import re
import pandas as pd
import radvel
from astroquery.simbad import Simbad
import wget, zipfile, shutil
norm_mean     = lambda x: x/np.nanmean(x)

def pickle_dump(filename,obj):
    """
    write obj to to filename and save
    
    INPUT:
        filename - name to save pickle file (str)
        obj - object to write to file
    
    OUTPUT:
        obj saved as pickle file
        
    EXAMPLE:
        pickle_dump()
    
    """
    savefile = open(filename,"w")
    pickle.dump(obj, savefile)
    savefile.close()
    print("Saved to {}".format(filename))

def pickle_load(filename,python3=True):
    """
    load obj from filename
    
    INPUT:
        filename - name of saved pickle file (str)
        obj - object to write to file
    
    OUTPUT:
        load obj from pickle file
        
    EXAMPLE:
        pickle_load()
    """
    if python3:
        openfile = open(filename,"rb")
    return pickle.load(openfile,encoding='latin1')


def jd2datetime(times):
    """
    convert jd to iso utc date/time
    
    INPUT:
        times in jd (array)
    
    OUTPUT:
        date/time in utc (array)
        
    EXAMPLE:
        
    """
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])

def iso2jd(times):
    """
    convert iso utc to jd date/time
    
    INPUT:
        date/time in iso uts (array)
    
    OUTPUT:
        date/time in jd (array)
        
    EXAMPLE:
        
    """
    return np.array([astropy.time.Time(time,format="iso",scale="utc").jd for time in times])

def make_dir(dirname,verbose=True):
    try:
        os.makedirs(dirname)
        if verbose==True: print("Created folder:",dirname)
    except OSError:
        if verbose==True: print(dirname,"already exists. Skipping")



def vac2air(wavelength,P,T,input_in_angstroms=True):
    """
    Convert vacuum wavelengths to air wavelengths

    INPUT:
        wavelength - in A if input_in_angstroms is True, else nm
        P - in Torr
        T - in Celsius

    OUTPUT:
        Wavelength in air in A if input_in_angstroms is True, else nm
    """
    if input_in_angstroms:
        nn = n_air(P,T,wavelength/10.)
    else:
        nn = n_air(P,T,wavelength)
    return wavelength/(nn+1.)

def n_air(P,T,wavelength):
    """
    The edlen equation for index of refraction of air with pressure
    
    INPUT:
        P - pressure in Torr
        T - Temperature in Celsius
        wavelength - wavelength in nm
        
    OUTPUT:
        (n-1)_tp - see equation 1, and 2 in REF below.
        
    REF:
        http://iopscience.iop.org/article/10.1088/0026-1394/30/3/004/pdf

    EXAMPLE:
        nn = n_air(763.,20.,500.)-n_air(760.,20.,500.)
        (nn/(nn + 1.))*3.e8
    """
    wavenum = 1000./wavelength # in 1/micron
    # refractivity is (n-1)_s for a standard air mixture
    refractivity = ( 8342.13 + 2406030./(130.-wavenum**2.) + 15997./(38.9-wavenum**2.))*1e-8
    return ((P*refractivity)/720.775) * ( (1.+P*(0.817-0.0133*T)*1e-6) / (1. + 0.0036610*T) )

def get_cmap_colors(cmap='jet',p=None,N=10):
    """

    """
    cm = plt.get_cmap(cmap)
    if p is None:
        return [cm(i) for i in np.linspace(0,1,N)]
    else:
        normalize = matplotlib.colors.Normalize(vmin=min(p), vmax=max(p))
        colors = [cm(normalize(value)) for value in p]
        return colors

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize=12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.3,alpha=0.3)

def ax_add_colorbar(ax,p,cmap='jet',tick_width=1,tick_length=3,direction='out',pad=0.02,minorticks=False,*kw_args):
    """
    Add a colorbar to a plot (e.g., of lines)

    INPUT:
        ax - axis to put the colorbar
        p  - parameter that will be used for the scaling (min and max)
        cmap - 'jet', 'viridis'

    OUTPUT:
        cax - the colorbar object

    NOTES:
        also see here:
            import matplotlib.pyplot as plt
            sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []
            plt.colorbar(sm)

    EXAMPLE:
        cmap = 'jet'
        colors = get_cmap_colors(N=len(bjds),cmap=cmap)
        fig, ax = plt.subplots(dpi=200)
        for i in range(len(bjds)):
            ax.plot(vin[i],ccf_sub[i],color=colors[i],lw=1)
        ax_apply_settings(ax,ticksize=14)
        ax.set_xlabel('Velocity [km/s]',fontsize=20)
        ax.set_ylabel('Relative Flux ',fontsize=20)
        cx = ax_add_colorbar(ax,obs_phases,cmap=cmap,pad=0.02)
        ax_apply_settings(cx)
        cx.axes.set_ylabel('Phase',fontsize=20,labelpad=0)


        cbar.set_clim(1.4,4.1)
        cbar.set_ticks([1.5,2.0,2.5,3.0,3.5,4.0])
    """
    cax, _ = matplotlib.colorbar.make_axes(ax,pad=pad,*kw_args)
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(p), vmax=np.nanmax(p))
    cm = plt.get_cmap(cmap)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=normalize)
    if minorticks:
        cax.minorticks_on()
    cax.axes.tick_params(width=tick_width,length=tick_length,direction=direction)
    return cax, cbar

def get_indices_of_items(arr,items):
    return np.where(pd.DataFrame(arr).isin(items))[0]


def remove_items_from_list(l,bad_items):
    ibad = np.where(pd.DataFrame(l).isin(bad_items))[0]
    return np.delete(l,ibad)

def savefigure(fig,savename,s1='{}',p1='',s2='{}',p2='',dpi=200):
    """
    Handy function to save figures and append suffixes to filenames
    
    EXAMPLE:
        savefigure(fig,'MASTER_FLATS/COMPARE_PLOTS/testing.png',s1='_o{}',p1=5,s2='_spi{}',p2=14)
    """
    fp = FilePath(savename)
    make_dir(fp.directory)
    fp.add_suffix(s1.format(p1))
    fp.add_suffix(s2.format(p2))
    fig.tight_layout()
    fig.savefig(fp._fullpath,dpi=dpi)
    print('Saved figure to: {}'.format(fp._fullpath))

def grep_date(string,intype="isot",outtype='iso'):
    """
    A function to extract date from string.

    INPUT:
        string: string
        intype: "isot" - 20181012T001823
                "iso"  - 20181012
        outtype: "iso" - iso
                 "datetime" - datetime

    OUTPUT:
        string with the date
    """
    if intype == "isot":
        date = re.findall(r'\d{8}T\d{6}',string)[0]
    elif intype == "iso":
        date = re.findall(r'\d{8}',string)[0]
    else:
        print("intype has to be 'isot' or 'iso'")
    if outtype == 'iso':
        return date
    elif outtype == 'datetime':
        return pd.to_datetime(date).to_pydatetime()
    else:
        print('outtype has to be "iso" or "datetime"')

def grep_dates(strings,intype="isot",outtype='iso'):
    """
    A function to extract date from strings

    INPUT:
        string: string
        intype: "isot" - 20181012T001823
                "iso"  - 20181012
        outtype: "iso" - iso
                 "datetime" - datetime

    OUTPUT:
        string with the date

    EXAMPLE:
        df = grep_dates(files,intype="isot",outtype='series')
        df['2018-06-26':].values
    """
    if outtype=='series':
        dates = [grep_date(i,intype=intype,outtype='datetime') for i in strings]
        return pd.Series(index=dates,data=strings)
    else:
        return [grep_date(i,intype,outtype) for i in strings]

def replace_dir(files,old,new):
    for i,f in enumerate(files):
        files[i] = files[i].replace(old,new)
    return files

def get_header_df(fitsfiles,keywords=["OBJECT","DATE-OBS"],verbose=True):
    """
    A function to read headers and returns a pandas dataframe

    INPUT:
    fitsfiles - a list of fitsfiles
    keywords - a list of header keywords to read

    OUTPUT:
    df - pandas dataframe with column names as the passed keywords
    """
    headers = []
    for i,name in enumerate(fitsfiles):
        if verbose: print(i,name)
        head = astropy.io.fits.getheader(name)
        values = [name]
        for key in keywords:
            values.append(head[key])
        headers.append(values)
    df_header = pd.DataFrame(headers,columns=["filename"]+keywords)
    return df_header

def plot_rv_model(bjd,RV,e_RV,P,T0,K,e,w,nbjd=None,nRV=None,ne_RV=None,title="",fig=None,ax=None,bx=None):
    """
    Plot RVs and the rv model on top. Also plot residuals
    """
    bjd_model = np.linspace(bjd[0]-2,bjd[-1]+2,10000)
    rv_model = get_rv_curve(bjd_model,P,T0,e=e,omega=w,K=K,plot=False)
    rv_obs = get_rv_curve(bjd,P,T0,e,w,K,plot=False)
    
    if nbjd is not None:
        rv_obs_bin = get_rv_curve(nbjd,P,T0,e,w,K,plot=False)

    # Plotting
    if fig is None and ax is None and bx is None:
        fig, (ax, bx) = plt.subplots(dpi=200,nrows=2,gridspec_kw={"height_ratios":[5,2]},figsize=(6,4),sharex=True)
    
    ax.errorbar(bjd,RV,e_RV,elinewidth=1,mew=0.5,capsize=5,marker="o",lw=0,markersize=6,alpha=0.5,
            label="Unbin, median errorbar={:0.1f}m/s".format(np.median(e_RV)))
    ax.plot(bjd_model,rv_model,label="Expected Orbit",lw=1,color="crimson")
    res = RV - rv_obs
    bx.errorbar(bjd,res,e_RV,elinewidth=1,mew=0.5,capsize=5,marker="o",lw=0,markersize=6,alpha=0.5,label='Residual: $\sigma$={:0.2f}m/s'.format(np.std(res)))
    
    if nbjd is not None:
        ax.errorbar(nbjd,nRV,ne_RV,elinewidth=1,mew=0.5,capsize=5,marker="h",lw=0,color="crimson",
            markersize=8,label="Bin, median errorbar={:0.1f}m/s".format(np.median(ne_RV)))
        bx.errorbar(nbjd,nRV-rv_obs_bin,ne_RV,elinewidth=1,mew=0.5,capsize=5,marker="h",lw=0,markersize=8,alpha=0.5,color="crimson")
    ax.legend(fontsize=8,loc="upper right")
    bx.legend(fontsize=8,loc="upper right")

    for xx in [ax,bx]:
        ax_apply_settings(xx,ticksize=12)
        xx.set_ylabel("RV [m/s]",fontsize=16)

    bx.set_xlabel("Date [UT]",labelpad=0,fontsize=16)
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    
def plot_rv_model_phased(bjd,RV,e_RV,P,T0,K,e,w,nbjd=None,nRV=None,ne_RV=None,title="",fig=None,ax=None,bx=None):
    """
    Plot RVs and the rv model on top. Also plot residuals
    """
    bjd_model = np.linspace(bjd[0]-2,bjd[-1]+2,10000)
    rv_model = get_rv_curve(bjd_model,P,T0,e=e,omega=w,K=K,plot=False)
    rv_obs = get_rv_curve(bjd,P,T0,e,w,K,plot=False)
    
    # Phases:
    df_phase = get_phases_sorted(bjd,P,T0).sort_values("time")
    df_phase_model = get_phases_sorted(bjd_model,P,T0,rvs=rv_model).sort_values("time")

    if nbjd is not None:
        rv_obs_bin = get_rv_curve(nbjd,P,T0,e,w,K,plot=False)
        df_phase_bin = get_phases_sorted(nbjd,P,T0).sort_values("time")

    # Plotting
    if fig is None and ax is None and bx is None:
        fig, (ax, bx) = plt.subplots(dpi=200,nrows=2,gridspec_kw={"height_ratios":[5,2]},figsize=(6,4),sharex=True)
    
    ax.errorbar(df_phase.phases,RV,e_RV,elinewidth=1,mew=0.5,capsize=5,marker="o",lw=0,markersize=6,alpha=0.5,
            label="Unbin, median errorbar={:0.1f}m/s".format(np.median(e_RV)))
    ax.plot(df_phase_model.phases,rv_model,label="Expected Orbit",lw=0,marker="o",markersize=2,color="crimson")
    res = RV - rv_obs
    bx.errorbar(df_phase.phases,res,e_RV,elinewidth=1,mew=0.5,capsize=5,marker="o",lw=0,markersize=6,alpha=0.5,label='Residual: $\sigma$={:0.2f}m/s'.format(np.std(res)))
    
    if nbjd is not None:
        ax.errorbar(df_phase_bin.phases,nRV,ne_RV,elinewidth=1,mew=0.5,capsize=5,marker="h",lw=0,color="crimson",
            markersize=8,label="Bin, median errorbar={:0.1f}m/s".format(np.median(ne_RV)),alpha=0.9)
        bx.errorbar(df_phase_bin.phases,nRV-rv_obs_bin,ne_RV,elinewidth=1,mew=0.5,capsize=5,marker="h",lw=0,markersize=8,alpha=0.9,color="crimson")
    ax.legend(fontsize=8,loc="upper right")
    bx.legend(fontsize=8,loc="upper right")

    for xx in [ax,bx]:
        ax_apply_settings(xx,ticksize=12)
        xx.set_ylabel("RV [m/s]",fontsize=16)

    bx.set_xlabel("Orbital phase",labelpad=0,fontsize=16)
    ax.set_title(title)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)

def get_phases_sorted(t, P, t0,rvs=None,rvs_err=None,sort=True,centered_on_0=True,tdur=None):
    """
    Get a sorted pandas dataframe of phases, times (and Rvs if supplied)
    
    INPUT:
    t  - times in jd
    P  - period in days
    t0 - time of periastron usually
    
    OUTPUT:
    df - pandas dataframe with columns:
     -- phases (sorted)
     -- time - time
     -- rvs  - if provided
    
    NOTES:
    Useful for RVs.    
    """
    phases = np.mod(t - t0,P)
    phases /= P
    df = pd.DataFrame(zip(phases,t),columns=['phases','time'])
    if rvs is not None:
        df['rvs'] = rvs
    if rvs_err is not None:
        df['rvs_err'] = rvs_err
    if centered_on_0:
        _p = df.phases.values
        m = df.phases.values > 0.5
        _p[m] = _p[m] - 1.
        df['phases'] = _p
    if tdur is not None:
        df["intransit"] = np.abs(df.phases) < tdur/(2.*P)
        print("Found {} in transit".format(len(df[df['intransit']])))
    if sort:
        df = df.sort_values('phases').reset_index(drop=True)
    return df

def get_rv_curve(times_jd,P,tc,e,omega,K,plot=True,ax=None,verbose=True,plot_tnow=True):
    """
    A function to plot an RV curve as a function of time (not phased)
    
    INPUT:
        times_jd: times in jd
        P: orbital period in days
        tc: transit center in jd
        e: eccentricity
        omega: periastron in degrees
        K: RV semi-amplitude in m/s
    
    OUTPUT:
        rv: array of RVs at times times_jd
    """
    t_peri = radvel.orbit.timetrans_to_timeperi(tc=tc,
                                                per=P,
                                                ecc=e,
                                                omega=np.deg2rad(omega))
    rvs = radvel.kepler.rv_drive(times_jd,[P,
                                            t_peri,
                                            e,
                                            np.deg2rad(omega),
                                            K])
    if verbose:
        print("Assuming:")
        print("P {}d".format(P))
        print("tc {}".format(tc))
        print("e {}".format(e))
        print("omega {}deg".format(omega))
        print("K {}m/s".format(K))
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,8))
        times = jd2datetime(times_jd)
        ax.plot(times,rvs)
        ax.set_xlabel("Time")
        ax.set_ylabel("RV [m/s]")
        ax.grid(lw=0.5,alpha=0.3)
        ax.minorticks_on()
        if plot_tnow:
            t_now = Time(datetime.datetime.utcnow())
            ax.axvline(t_now.datetime,color="red",label="Time now")
        xlim = ax.get_xlim()
        ax.hlines(0,xlim[0],xlim[1],alpha=0.5,color="k",lw=1)
        for label in (ax.get_xticklabels()):
            label.set_fontsize(10)
    return rvs


def filter_simbadnames(l):
    """
    Filter SIMBAD names
    """
    _n = find_str_in_list(l,'DR2')
    if _n=='':
        _n = find_str_in_list(l,'DR1')
    if _n=='':
        _n = find_str_in_list(l,'HD')
    if _n=='':
        _n = find_str_in_list(l,'GJ')
    return _n

def get_simbad_object_names(name,as_str=False):
    """
    Get all Simbad names for an object
    """
    result_table = Simbad.query_objectids(name).to_pandas()
    names = result_table.ID.values.astype(str)
    if as_str:
        names = [i.replace(' ','_') for i in names]
        return '|'.join(names)
    else:
        return names

def find_str_in_list(l,string):
    """
    Return first element that matches a string in a list
    """
    for element in l:
        if string in element:
            return element
    return ''

def get_library(url = 'https://www.dropbox.com/s/8fcraxmpgqdq9w9/20200128_specmatch_nir.zip?dl=1', outputdir='../library/20200128_specmatch_nir'):
    """
    Download stellar library if it does not already exist
    
    INPUT:
        url - url of library file to download
        outputdir - library save directory
    
    OUTPUT:
        saves downloaded zip file to outputdir folder
        
    EXAMPLE:
        get_library()
    
    """
    if not os.path.isdir('../library/{}'.format(outputdir)):
        
        print('Downloading library from: {}'.format(url))
        wget.download(url, '../library/20200128_specmatch_nir.zip')
        
        print('Extracting zip file')
        with zipfile.ZipFile('../library/20200128_specmatch_nir.zip', 'r') as zip_ref:
            zip_ref.extractall('../library/')  
        print('Extracting zip file complete')
        print('Deleting zip file')
        os.remove('../library/20200128_specmatch_nir.zip')
        shutil.rmtree('../library/__MACOSX')
        
        print('Download complete')
        
    else:
        
        print('Library already exists. Skipping download')