import hpfspec
import numpy as np
import pandas as pd
import glob
import astropy.io
import pickle
import os
from hpfspec import stats_help
from lmfit import minimize, Parameters
import pyde
import pyde.de
import emcee
import scipy.optimize
from hpfspec import rotbroad_help
from hpfspec import utils
import matplotlib.pyplot as plt
import astropy.modeling
from .priors import PriorSet, UP, NP, JP
from .likelihood import ll_normal_es_py, ll_normal_ev_py
from . import config
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams["savefig.dpi"] = 100
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['font.weight'] = "normal"
rcParams["axes.formatter.useoffset"] = False
rcParams['xtick.direction']='in'
rcParams['ytick.direction']='in'

def get_data_ready(H1,Hrefs,w,v,polyvals=None,vsinis=None,plot=False):
    """
    Get data ready for MCMC
    
    INPUT:
        H1 - target spectrum (HPFSpectrum object)
        Hrefs - reference spectra (HPFSpectraList object)
        w - wavelength grid to interpolate on (array)
        v - velocities in km/s to use for absolute RV consideration (array)
        polyvals - polynomial coefficients (array)
        vsinis - vsini values km/s (array)
        plot - (boolean)

    OUTPUT:
        f1 - spectrum
        e1 - spectrum flux error
        ffrefs - spectra  (array)
        eerefs - spectra flux errors (array)
       
    EXAMPLE:
        "Target={}, rv={:0.3f}km/s, rvmed={:0.3f}km/s".format(H1.target.name,H1.rv,np.median(rabs))
        files = sorted(glob.glob('20200209_ad_leos/AD_Leo/*/*.pkl'))
        summarize_values_from_orders(files,'AD_Leo')
    """
    H1.deblaze()
    _, rabs = H1.rvabs_for_orders(v,orders=[5],plot=plot)
    H1.redshift(rv=np.median(rabs))
    f1, e1 = H1.resample_order(w)
    print("Target={}, rv={:0.3f}km/s, rvmed={:0.3f}km/s".format(H1.target.name,H1.rv,np.median(rabs)))
    
    ffrefs = []
    eerefs = []
    for i, H in enumerate(Hrefs.splist):
        H.deblaze()
        _, rabs = H.rvabs_for_orders(v,orders=[5],plot=plot)
        H.redshift(rv=np.median(rabs))
        if polyvals is None and vsinis is None:
            _f, _e = H.resample_order(w)
        else:
            _f, _e = H.resample_order(w,p=polyvals[i],vsini=vsinis[i])
        ffrefs.append(_f)
        eerefs.append(_e)
        print("Target={}, rv={:0.3f}km/s, rvmed={:0.3f}km/s".format(H1.target.name,H1.rv,np.median(rabs)))
    return f1, e1, ffrefs, eerefs 

class LPFunctionLinComb(object):
    """
    Log Likelihood function for optimizing a combination of 5 top spectra
    """
    def __init__(self,w,f1,e1,f2s,e2s):
        self.w = w
        self.data_target = {'f': f1,
                            'e': e1}
        self.data_refs   = {'f': f2s,
                            'e': e2s}
        self.num_refs = len(self.data_refs['f'])
        
        self.priors = [UP( 0., 1., 'c1', '$c1$',priortype="model"),
                       UP( 0., 1., 'c2', '$c2$',priortype="model"),
                       UP( 0., 1., 'c3', '$c3$',priortype="model"),
                       UP( 0., 1., 'c4', '$c4$',priortype="model")]
        self.ps     = PriorSet(self.priors)
        
    def get_pv_all(self,pv):
        pv_all = list(pv) + [1.-np.sum(pv)] 
        return pv_all
        
    def compute_model(self,pv):
        pv_all = self.get_pv_all(pv)    
        return np.sum([self.data_refs['f'][i]*pv_all[i] for i in range(self.num_refs)],axis=0)
        
    def __call__(self,pv):
        if 1.-np.sum(pv) < 0. or 1.-np.sum(pv)> 1.:
            return -np.inf
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -np.inf
        flux_model = self.compute_model(pv)
        flux_target = self.data_target['f']
        error_target = self.data_target['e']
        log_of_priors = self.ps.c_log_prior(pv)
        log_of_model = ll_normal_ev_py(flux_target,flux_model,error_target)
        log_ln = log_of_priors + log_of_model
        #log_constraint = (np.sum(pv)-1.)/(2.*self.sigma)
        #print('lnPrior = {} lnModel = {} lnExtra = {}'.format(log_of_priors,log_of_model,log_constraint))
        return log_ln
    
class FitLinCombSpec(object):
    """
    A class to fit 5 lin-comb spectra together. Note: look at LPFunctionLinComb
    """
    def __init__(self,LPFunctionLinComb,teffs=[],fehs=[],loggs=[],vsinis=[],
                 teff_known=None,tefferr_known=None,
                 feh_known=None,feherr_known=None,
                 logg_known=None,loggerr_known=None,targetname='',calibrate_feh=True):
        self.lpf = LPFunctionLinComb
        self.teffs = teffs
        self.fehs = fehs
        self.loggs = loggs
        self.vsinis = vsinis
        self.teff_known = teff_known
        self.tefferr_known = tefferr_known
        self.feh_known = feh_known
        self.feherr_known = feherr_known
        self.logg_known = logg_known
        self.loggerr_known = loggerr_known
        self.targetname = targetname
        self.calibrate_feh = calibrate_feh
        
    def calculate_stellar_parameters(self,weights):
        if self.teffs != []:
            self.teff = weighted_value(self.teffs,weights)
        else: 
            self.teff = np.nan
        if self.fehs != []:
            self.feh = weighted_value(self.fehs,weights)
            if self.calibrate_feh:
                print('Calibrating feh: {:0.3f} -> {:0.3f}'.format(self.feh,detrend_feh(self.feh)))
                self.feh = detrend_feh(self.feh)
        else: 
            self.feh = np.nan
        if self.loggs != []: 
            self.logg = weighted_value(self.loggs,weights)
        else: 
            self.logg = np.nan
        if self.vsinis != []:
            self.vsini = weighted_value(self.vsinis,weights)
        print('Stellar parameters:')
        print('Teff [K]:',self.teff)
        print('Fe/H [dex]:',self.feh)
        print('logg [dex]:',self.logg)
        print('vsini [km/s]:',self.vsini)
        
    def plot_model(self,pv):
        pv_all = self.lpf.get_pv_all(pv)
        fig, (ax, bx) = plt.subplots(nrows=2,dpi=200,sharex=True,gridspec_kw={'height_ratios':[5,2]})
        ax.plot(self.lpf.w,self.lpf.data_target['f'],color='black',label='Target',lw=1)
        ff = self.lpf.compute_model(pv)
        
        ax.plot(self.lpf.w,ff,color='crimson',label='Composite',alpha=0.5,lw=1)
        ax.legend(fontsize=10)
        bx.plot(self.lpf.w,self.lpf.data_target['f']-ff,lw=1)
        bx.set_xlabel('Wavelength [A]',fontsize=12)
        bx.set_ylabel('Residual',fontsize=12)
        
        self.calculate_stellar_parameters(pv_all)
        
        title = 'log_ln={}\n'.format(self.lpf(pv))
        title += ' '.join(['c_{}={:0.5f}'.format(i,pv_all[i]) for i in range(5)])
        title += '\nTeff={:0.3f}, Fe/H={:0.3f}, logg={:0.3f}'.format(self.teff,self.feh,self.logg)
        if self.teff_known is not None and self.feh_known is not None and self.logg_known is not None:
            title += '\nTeff_0={:0.3f}+-{:0.1f}K, Fe/H_0={:0.3f}+-{:0.3f}, logg_0={:0.3f}+-{:0.3f}'.format(self.teff_known,self.tefferr_known,self.feh_known,self.feherr_known,self.logg_known,self.loggerr_known)
        ax.set_title(title,fontsize=10)
        for xx in (ax,bx):
            utils.ax_apply_settings(xx,ticksize=10)
        fig.subplots_adjust(hspace=0.05)
        
    def plot_model_with_components(self,pv,fig=None,ax=None,names=None,savename='compositeComparison.pdf',title='',scaleres=1.):
        """
        INPUT:
            scaleres - amount to scale residuals from composite spectrum (default 1)
        Make a plot of the 5 best stars and compare to targets spectrum.
        Compare composite spectrum to target spectrum and plot residuals.
        """
        w = self.lpf.w
        pv_all = self.lpf.get_pv_all(pv)
        self.calculate_stellar_parameters(pv_all)
        if fig is None and ax is None:
            fig, ax = plt.subplots(dpi=200)
        ax.plot(w,self.lpf.data_target['f'],color='black',label='Target',lw=1)
        ff = self.lpf.compute_model(pv)
        ax.plot(w,ff,color='crimson',label='Composite',alpha=0.5,lw=1.5)

        for i in range(5):
            ax.plot(w,self.lpf.data_refs['f'][i]+(5.0-i),lw=1,color='black')
            if names is not None:
                label = '{}, $c_{}$={:0.5f}'.format(names[i].replace('_',' '),i+1,self.lpf.get_pv_all(pv)[i])
                label+= ', Teff={:0.0f}K'.format(self.teffs[i])
            else:
                label = '$c_{}$={:0.5f}'.format(i+1,pv_all[i])
                label+= ', Teff={:0.0f}K'.format(self.teffs[i])
            ax.text(w[0],(6.12-i),label,color='black',fontsize=8)

        ax.text(w[0],1.15,'Target Spectrum (Black), Composite Spectrum (Red)',fontsize=8)
        ax.text(w[0],0.15,'Residual: Target - Composite (Scale: {:0.0f}x)'.format(scaleres),fontsize=8)

        title += 'Target={}, Teff={:0.3f}, Fe/H={:0.3f}, logg={:0.3f}, vsini={:0.3f}km/s'.format(self.targetname,self.teff,self.feh,self.logg,self.vsini)
        ax.set_title(title,fontsize=10)
        
        ax.plot(w,(self.lpf.data_target['f']-ff)*scaleres,color='black',lw=1)
        ax.set_xlabel('Wavelength [A]',fontsize=12,labelpad=2)
        ax.set_ylabel('Flux (+offset)',fontsize=12,labelpad=2)
        #ax.set_title('log_ln={}, c={}'.format(self.lpf(pv),str(self.lpf.get_pv_all(pv))),fontsize=10)
        utils.ax_apply_settings(ax,ticksize=10)
        fig.tight_layout()
        fig.savefig(savename,dpi=200)
        print('Saved to {}'.format(savename))
            
    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False):
        """
        Minimize using the PyDE
        
        NOTES:
        https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.lpf.ps.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.lpf, self.lpf.ps.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogLn value:",self.min_pv_lnval)
        print("Log priors",self.lpf.ps.c_log_prior(self.min_pv))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps.ndim, self.lpf,threads=threads)
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
            print("Finished MCMC")
            
    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps.labels,self.lpf.ps.centers,self.lpf.ps.bounds[:,0],self.lpf.ps.bounds[:,1],pv,self.lpf.ps.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics
    
    #def plot_chains(self,labels=None,burn=0,thin=1):
    #    print("Plotting chains")
    #    if labels==None:
    #        labels = self.lpf.ps.descriptions
    #    mcFunc.plot_chains(self.sampler.chain,labels=labels,burn=burn,thin=thin)
        
    #def plot_corner(self,labels=None,burn=0,thin=1,title_fmt='.5f',**kwargs):
    #    if labels==None:
    #        labels = self.lpf.ps.descriptions
    #    self.fig = mcFunc.plot_corner(self.sampler.chain,labels=labels,burn=burn,thin=thin,title_fmt=title_fmt,**kwargs)
                
def sample(df_chain,N=500):
    spectra = []
    for i in range(N):
        c = df_chain.iloc[np.random.randint(len(df_chain))].values[0:4]
        spectra.append(self.lpf.data_target['f']-LCS.lpf.compute_model(c))
    std = np.std(spectra,axis=0)#np.std(spectra,axis=0)
    return std


class Chi2FunctionVsiniPolynomial(object):
    def __init__(self,w,f1,e1,f2,e2,maxvsini):
        self.w = w
        self.data_target = {'f': f1,
                            'e': e1}
        self.data_ref    = {'f': f2,
                            'e': e2}
        self.priors = [UP( 0.     , maxvsini    , 'vsini', '$v \sin i$',priortype="model"),
                       UP( -1e10  , 1e10        , 'c0'   , 'c_0'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c1'   , 'c_1'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c2'   , 'c_2'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c3'   , 'c_3'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c4'   , 'c_4'       ,priortype="model"),
                       UP( -1e10  , 1e10        , 'c5'   , 'c_5'       ,priortype="model")]
        self.ps     = PriorSet(self.priors)
        
    def compute_model(self,pv,eps=0.3):
        """
        Multiply reference by polynomial and then rotationally broaden.
        """
        vsini = pv[0]
        coeffs = pv[1:]
        ff_ref = self.data_ref['f']*np.polynomial.chebyshev.chebval(self.w,coeffs)
        ff_ref = rotbroad_help.broaden(self.w,ff_ref,vsini,u1=eps)
        return ff_ref
        
    def __call__(self,pv,verbose=False):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return np.inf        
        flux_model = self.compute_model(pv)
        flux_target = self.data_target['f']
        dummy_error = np.ones(len(flux_target))
        chi2 = stats_help.chi2(flux_target-flux_model,dummy_error,1,verbose=verbose)
        #print(pv,chi2)
        #log_of_priors = self.ps.c_log_prior(pv)
        return chi2

class FitTargetRefStarVsiniPolynomial(object):
    """
    A class to fit 5 lin-comb spectra together. Note: look at LPFunctionLinComb
    """
    def __init__(self,Chi2FunctionVsiniPolynomial):
        self.chi2f = Chi2FunctionVsiniPolynomial

    def plot_model(self,pv):
        vsini = pv[0]
        coeffs = pv[1:]
        fig, (ax, bx) = plt.subplots(nrows=2,dpi=200,sharex=True,gridspec_kw={'height_ratios':[5,2]})
        ax.plot(self.chi2f.w,self.chi2f.data_target['f'],color='black',label='Target',lw=1)
        ff = self.chi2f.compute_model(pv)
        ax.plot(self.chi2f.w,np.polynomial.chebyshev.chebval(self.chi2f.w,coeffs))
        
        ax.plot(self.chi2f.w,ff,color='crimson',label='Reference (vsini={:0.3f}km/s)'.format(vsini),alpha=0.5,lw=1)
        ax.legend(fontsize=10)
        bx.plot(self.chi2f.w,self.chi2f.data_target['f']-ff,lw=1)
        bx.set_xlabel('Wavelength [A]',fontsize=12)
        bx.set_ylabel('Residual',fontsize=12)
                
        title = '$\chi^2$={}, coeffs={}'.format(self.chi2f(pv),coeffs)
        ax.set_title(title,fontsize=10)
        for xx in (ax,bx):
            utils.ax_apply_settings(xx,ticksize=10)
        fig.subplots_adjust(hspace=0.05)
        
    def minimize_AMOEBA(self,centers = [0.01,0.,0.,0.,0.,0.,1.]):
        #print('Performing first Chebfit')
        #centers_coeffs = np.poly(self.chi2f.w,self.chi2f.data_target['f']-self.chi2f.data_ref['f']+1.,5)
        centers_coeffs = np.polynomial.chebyshev.chebfit(self.chi2f.w,self.chi2f.data_target['f']-self.chi2f.data_ref['f']+1.,5)
        
        #print('Found centers:',centers_coeffs)
        
        # ####################################################
        # CHANGING 20190629
        centers = [1.]+list(centers_coeffs)
        #centers = [0.5]+list(centers_coeffs)
        # ####################################################
        
        #print('With CHI',self.chi2f(centers))
        #print(len(centers),len(centers_coeffs))
        #random = self.lpf.ps.random
        #centers = np.array(self.lpf.ps.centers)
        
        #self.res = scipy.optimize.minimize(self.chi2f,centers,method='Nelder-Mead',tol=1e-1,
        #                           options={'maxiter': 1000, 'maxfev': 2000, 'disp': True})
        self.res = scipy.optimize.minimize(self.chi2f,centers,method='Powell',tol=1e-1,
                                   options={'maxiter': 1000, 'maxfev': 2000, 'disp': True})
        #self.res = scipy.optimize.minimize(self.chi2f,centers,method='Nelder-Mead',tol=1e-7,
        #                           options={'maxiter': 10000, 'maxfev': 5000, 'disp': True})
        
        self.min_pv = self.res.x
        
        
    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=False,plot_priors=True,sample_ball=False):
        """
        Minimize using the PyDE
        
        NOTES:
        https://github.com/hpparvi/PyDE
        """
        centers = np.array(self.chi2f.ps.centers)
        print("Running PyDE Optimizer")
        self.de = pyde.de.DiffEvol(self.chi2f, self.chi2f.ps.bounds, npop, maximize=False) # we want to maximize the likelihood
        self.min_pv, self.min_pv_chi2val = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        #print("LogLn value:",self.min_pv_lnval)
        #print("Log priors",self.lpf.ps.c_log_prior(self.min_pv))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.chi2f.ps.ndim, self.chi2f,threads=threads)
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
            print("Finished MCMC")

    
def chi2spectraPolyVsini(ww,H1,H2,rv1=None,rv2=None,plot=False,verbose=False,maxvsini=30.):
    """
    INPUT:
        ww - wavelength grid to interpolate on (array)
        H1 - target spectrum (HPFSpectrum object)
        H2 - reference spectrum (HPFSpectrum object)
        rv1 - radial velocity H1 km/s (float)
        rv2 - radial velocity H2 km/s (float)
        plot - (boolean)
        verbose - print additional info (boolean)

    OUTPUT:
        chi2 - chi2 values for the comparison
        vsini - 
        coeffs - 
        
    EXAMPLE:
        H1 = HPFSpectrum(df[df.name=='G_9-40'].filename.values[0])
        H2 = HPFSpectrum(df[df.name=='AD_Leo'].filename.values[0])

        wmin = 10280.
        wmax = 10380.
        ww = np.arange(wmin,wmax,0.01)
        chi2spectraPolyVsini(ww,H1,H2,rv1=14.51,plot=True)
        
    """
    ff1, ee1 = H1.resample_order(ww)
    ff2, ee2 = H2.resample_order(ww)
    
    C = Chi2FunctionVsiniPolynomial(ww,ff1,ee1,ff2, ee2, maxvsini)
    FTRSVP = FitTargetRefStarVsiniPolynomial(C)
    FTRSVP.minimize_AMOEBA()
    vsini = FTRSVP.min_pv[0]
    coeffs = FTRSVP.min_pv[1:]
    chi2 = C(FTRSVP.min_pv)
    
    if plot:
        FTRSVP.plot_model(FTRSVP.min_pv)
        
    return chi2, vsini, coeffs


def chi2spectraPolyLoop(ww,H1,Hrefs,plot_all=False,plot_chi=True,verbose=True,maxvsini=30.):
    """
    Calculate chi square - target and list of reference spectra
    
    INPUT:
        ww - wavelength grid to interpolate on (array)
        H1 - target spectrum (HPFSpectrum object)
        Hrefs - reference spectra (HPFSpectraList object)
        plot_all - creates many additional plots (boolean)
        plot_chi = plot chi2 H1 vs other stars (boolean)
        verbose - print additional info (boolean)
    
    OUTPUT:
        df - dataframe of all reference stars sorted by chi2 (chi2, poly_params, and vsini values)
        df_best - dataframe with 5 best reference stars (chi2, poly_params, and vsini values)
        Hrefs_best - 5 best fitting reference stars
        
    EXAMPLE:
        
    """
    chis = []
    poly_params = []
    vsinis = []
    for i, H2 in enumerate(Hrefs):
        if i == 0:#SEJ
            print('First step: Matching target star to all library stars')
            print("##################")

        chi, vsini, p  = chi2spectraPolyVsini(ww,H1,H2,plot=plot_all,maxvsini=maxvsini)
        if verbose: 
            print('{:3d}/{:2d}, Target = {:18s} Library Star = {:18s} chi2 = {:6.3f}'.format(i+1,len(Hrefs),H1.object,H2.object, chi))
        chis.append(chi)
        poly_params.append(p)
        vsinis.append(vsini)
        
    refnames = [H.object for H in Hrefs]
    df = pd.DataFrame(list(zip(refnames,chis,poly_params,vsinis)),columns=['OBJECT_ID','chi2','poly_params','vsini'])
    df = df.sort_values('chi2')
    df = df.reset_index(drop=False)
    df_best = df[0:5]
    Hrefs_best = hpfspec.HPFSpecList(np.array(Hrefs)[df_best['index'].values]);#SEJ
    
    if plot_chi:
        fig, ax = plt.subplots(dpi=200)
        ax.plot(df.chi2,lw=0,marker='h')
        ax.set_xticks(range(len(chis)))
        ax.set_ylabel('$\chi^2$')
        utils.ax_apply_settings(ax)
        ax.set_xticklabels(df['OBJECT_ID'])
        ax.tick_params('x',labelsize=8)
        ax.set_title('{} vs other stars'.format(H1.object))
        for lab in ax.get_xticklabels():
            lab.set_rotation(90.)
        ax.set_yscale('log')
        #x.set_ylim(0.01,1e3)
    return df, df_best, Hrefs_best

def weighted_value(values,weights):
    """
    Array of weighted values
    """
    return np.dot(values,weights)

def run_specmatch(Htarget,Hrefs,ww,v,df_library,df_target=None,plot=True,savefolder='out/',
                  maxvsini=30.,calibrate_feh=True,scaleres=1.):
    """
    Second chi2 loop, creates composite spectrum 
    
    INPUT:
        Htarget - target spectrum (HPFSpectrum object)
        Hrefs - reference spectra (HPFSpectraList object)
        ww - wavelength grid to interpolate on (array)
        v - velocities in km/s to use for absolute RV consideration (array)
        df_library - dataframe with info on Teff/FeH/logg for the library stars
        df_target - dataframe with target parameter info
        plot - save SpecMatch plots (boolean)
        savefolder - output directory name (String)
        scaleres - amount to scale residuals from composite spectrum (default 1)
    
    OUTPUT:
        stellar parameters teff, feh, logg, vsini, and their errors
        df_chi_total, LCS  
    
    EXAMPLE:
        
    """
    print('##################')
    print('Saving results to {}'.format(savefolder))
    print('##################')
    utils.make_dir(savefolder)
    targetname = Htarget.object
    ##############################
    
    print('##################')
    print('Running Chi2 loop')
    print('##################')
    ##############################
    # STEP 1: Chi2 Loop
    df_chi, df_chi_best, Hbest = chi2spectraPolyLoop(ww,Htarget,Hrefs,plot_all=False,verbose=True,
                                                     maxvsini=maxvsini)
    ##############################
    # Combine best data
    df_chi_best_total = pd.merge(df_chi_best,df_library,on='OBJECT_ID')
    df_chi_total = pd.merge(df_chi,df_library,on='OBJECT_ID')
    
    if plot:
        plot_chi_teff_feh_logg_panel(df_chi_total.chi2,
                                     df_chi_total.Teff,
                                     df_chi_total['[Fe/H]'],
                                     df_chi_total['log(g)'],
                                     savename=savefolder+targetname+'_chi2_3panel.png')
        plot_teff_feh_logg_corr_panel(df_chi_total.Teff.values,
                                      df_chi_total['[Fe/H]'].values,
                                      df_chi_total['log(g)'].values,
                                      df_chi_total.chi2.values,
                                      e_teff=df_chi_total['e_Teff'],
                                      e_feh=df_chi_total['e_[Fe/H]'],
                                      e_logg=df_chi_total['e_log(g)'],
                                      savename=savefolder+targetname+'_chi2_corrpanel.png')
    if df_target is not None:
        teff_known = df_target.Teff.values[0]
        tefferr_known = df_target.e_Teff.values[0]
        feh_known  = df_target['[Fe/H]'].values[0]
        feherr_known = df_target['e_[Fe/H]'].values[0]
        logg_known = df_target['log(g)'].values[0]
        loggerr_known = df_target['e_log(g)'].values[0]
    else: 
        teff_known = np.nan
        tefferr_known = np.nan
        feh_known  = np.nan
        feherr_known = np.nan
        logg_known = np.nan
        loggerr_known = np.nan
        
    print('##################')
    print('Performing Linear Combination Optimization')
    print('##################')

    ##############################
    # STEP 2 LINEAR COMBINATION
    ##############################
    f1, e1, ffrefs, eerefs  = get_data_ready(Htarget,Hbest,ww,v,polyvals=df_chi_best.poly_params.values,
                                             vsinis=df_chi_best.vsini.values)
    L = LPFunctionLinComb(ww,f1,e1,ffrefs,eerefs)
    LCS = FitLinCombSpec(L,df_chi_best_total.Teff.values,
                         df_chi_best_total['[Fe/H]'].values,
                         df_chi_best_total['log(g)'].values,
                         df_chi_best_total['vsini'].values,
                         teff_known=teff_known,
                         tefferr_known=tefferr_known,
                         feh_known=feh_known,
                         feherr_known=feherr_known,
                         logg_known=logg_known,
                         loggerr_known=loggerr_known,
                         targetname=targetname,
                         calibrate_feh=calibrate_feh)
    LCS.minimize_PyDE(mcmc=False)
    print(LCS.min_pv)
    #LCS.plot_model(LCS.min_pv)# SEJ
    if plot:
        LCS.plot_model_with_components(LCS.min_pv,names=df_chi_best['OBJECT_ID'].values,title = "",
                                       savename=savefolder+targetname+'_compositecomparison.png',scaleres=scaleres)
        
    teff = LCS.teff
    feh = LCS.feh
    logg = LCS.logg
    vsini = LCS.vsini
    teff_delta = teff - teff_known
    feh_delta = feh - feh_known
    logg_delta = logg - logg_known
    
    # Saving to pickle file
    weights = LCS.lpf.get_pv_all(LCS.min_pv)
    results = {'teff': teff,
               'logg': logg,
               'feh':  feh,
               'vsini': vsini,
               'weights': weights}
    savefile = open(savefolder+targetname+'_results.pkl',"wb")
    pickle.dump(results,savefile)
    savefile.close()
    print('Saved results to {}'.format(savefolder+targetname+'_results.pkl'))
    
    print('##################')
    print('Saving Chi2 dataframe to {}'.format(savefolder+targetname+'_chi2results.csv'))
    print('##################')
    df_chi_total.to_csv(savefolder+targetname+'_chi2results.csv',index=False)
    df_chi_best_total['weights'] = weights
    df_chi_best_total.to_csv(savefolder+targetname+'_chi2results_best.csv',index=False)
    return teff, feh, logg, vsini, teff_delta, feh_delta, logg_delta, df_chi_total, LCS

def plot_chi_teff_feh_logg_panel(chis,teff,feh,logg,savename='chi2panel.pdf',fig=None,title=''):
    """
    3panel plot chi2 vs Teff, FeH, logg for 5 best fit stars among all library stars
    
    INPUT:
        chis - (array)
        teff - (array)
        feh - (array)
        logg - (array)
        savename - directory name to save plot (str)
        title - plot title (str)

    OUTPUT:
        saves 3panel plot
            
    EXAMPLE:
        plot_chi_teff_feh_logg_panel(df_chi.chi2,df_chi.Teff,df_chi['[Fe/H]'],df_chi['log(g)'],savename='chi2.pdf')
    """
    if fig is None:
        fig, (ax, bx, cx) = plt.subplots(dpi=200,ncols=3,sharey=True,figsize=(6,2))
    df = pd.DataFrame(list(zip(chis,teff,feh,logg)),columns=['chi2','teff','feh','logg'])
    df_best = df.sort_values('chi2').reset_index(drop=True)[0:5]
    ax.plot(teff,chis,marker='o',lw=0)
    ax.plot(df_best.teff,df_best.chi2,color='crimson',marker='h',lw=0)
    #ax.set_xlim(2950.,4050)
    #ax.set_xticks(np.arange(3000,4100,200))

    ax.set_yscale('log')
    ax.set_ylabel('$\chi^2$',fontsize=10,labelpad=2)
    ax.set_xlabel('$T_{\mathrm{eff}}$ [K]',fontsize=10,labelpad=2)
    bx.set_xlabel('[Fe/H] ',fontsize=10,labelpad=2)
    cx.set_xlabel('$\log(g)$',fontsize=10,labelpad=2)

    bx.plot(feh,chis,marker='o',lw=0)
    bx.plot(df_best.feh,df_best.chi2,color='crimson',marker='h',lw=0)
    #bx.set_xlim(-0.55,0.55)
    #bx.set_xticks(np.arange(-0.5,0.55,0.25))

    cx.plot(logg,chis,marker='o',lw=0)
    cx.plot(df_best.logg,df_best.chi2,color='crimson',marker='h',lw=0)
    #cx.set_xlim(4.7,5.1)

    for xx in (ax,bx,cx):
        utils.ax_apply_settings(xx,ticksize=7)
        xx.tick_params(pad=2)
        #xx.set_ylim(3e3,1.5e5)
    bx.set_title(title,fontsize=14)
    fig.subplots_adjust(wspace=0.1,right=0.97,left=0.08,bottom=0.15,top=0.95)
    fig.savefig(savename,dpi=200)
    print('Saved to: {}'.format(savename))
    
def plot_teff_feh_logg_corr_panel(teff,feh,logg,chis,e_teff=None,e_feh=None,e_logg=None,fig=None,ax=None,bx=None,
                                  savename='corr_panel.pdf',scale_factor_for_points=200.):
    """
    Plot of teff vs feh, logg for 5 best fit stars among all library stars
    
    INPUT:
        teff - (array)
        feh - (array)
        logg - (array)
        chis - (array)
        savename - directory name to save plot (str)
        title - plot title (str)

    OUTPUT:
        saves correlation plot
        
    """
    if ax is None and bx is None and fig is None:
        fig, (ax,bx) = plt.subplots(ncols=2,dpi=200,sharey=True,figsize=(10,3))

    colors = utils.get_cmap_colors(p=chis)
    df_chi = pd.DataFrame(zip(teff,feh,logg,chis),columns=['Teff','[Fe/H]','log(g)','chi2'])
    df_chi = df_chi.sort_values('chi2').reset_index(drop=True)
    df_chi_best = df_chi[0:5]

    # First plot:
    ax.scatter(df_chi['[Fe/H]'],df_chi.Teff,s=scale_factor_for_points/df_chi.chi2.values**2.,color='black')
    ax.scatter(df_chi_best['[Fe/H]'],df_chi_best.Teff,lw=0,color='crimson',s=scale_factor_for_points/df_chi_best.chi2.values**2.)

    if e_feh is not None and e_logg is not None and e_teff is not None:
        median_xerr = np.median(e_feh)
        median_yerr = np.median(e_teff)
        ax.errorbar(-0.4,3950,xerr=median_xerr,yerr=median_yerr,capsize=3,elinewidth=0.5,mew=0.5,color='black')
        ax.text(-0.40+0.0,3820,'Median error',fontsize=10,horizontalalignment='center')

    # Second plot
    bx.scatter(df_chi['log(g)'],df_chi.Teff,s=scale_factor_for_points/df_chi.chi2.values**2.,color='black',
               label='Library spectra:\n(radius$\propto 1/\chi^2$)')
    bx.scatter(df_chi_best['log(g)'],df_chi_best.Teff,lw=0,color='crimson',s=scale_factor_for_points/df_chi_best.chi2.values**2.,label='5 Best-fit spectra')
    bx.legend(loc='upper right',fontsize=9,frameon=False)
    
    if e_feh is not None and e_logg is not None and e_teff is not None:
        median_xerr = np.median(e_logg)
        bx.errorbar(4.68,3230,xerr=median_xerr,yerr=median_yerr,capsize=3,elinewidth=0.5,mew=0.5,color='black')
        bx.text(4.685,3100,'Median error',fontsize=10,horizontalalignment='center')

    ax.set_ylabel('$T_{\mathrm{eff}}$ [K]',fontsize=12,labelpad=2)
    ax.set_xlabel('Fe/H [dex]',fontsize=12,labelpad=1)
    bx.set_xlabel('$\log (g)$ [dex]',fontsize=12,labelpad=1)

    fig.subplots_adjust(wspace=0.05)

    for xx in (ax,bx):
        utils.ax_apply_settings(xx,ticksize=9)
        xx.grid(lw=0.4,alpha=0.3)
        utils.ax_set_linewidth(xx,1.3)
    
    fig.savefig(savename,dpi=200)
    print('Saved to: {}'.format(savename))


def summarize_values_from_orders(files_pkl,targetname):
    """
    Summarize values from different orders from SpecMatch analysis

    INPUT:
        Name of pickle files

    OUTPUT:
        saves two .csv files:
            1) a .csv file with the Teff, Fe/H, and logg values from all of the orders
            2) a .csv file with the median and the standard deviation from different orders

    Example:
        files = sorted(glob.glob('20200209_ad_leos/AD_Leo/*/*.pkl'))
        summarize_values_from_orders(files,'AD_Leo')
    """
    #basedir = files_pkl[0].split(os.sep)[0:2]
    #basedir = os.path.join(*files[0].split(os.sep)[0:-2])
    savefolder = os.path.join(*files_pkl[0].split(os.sep)[0:-2])
    target= targetname
    params = ['teff','logg','feh','vsini']
    results = []
    for filename in files_pkl:
        f = open(filename,'rb')
        res = pickle.load(f)
        results.append(res)
        f.close()
    df = pd.DataFrame(results)
    df['filenames'] = files_pkl
    medians = df.median(axis=0).values
    stds = df.std(axis=0).values
    
    df_med = pd.DataFrame(list(zip(params,medians,stds)),columns=['parameters','median','std'])
    
    df.to_csv(savefolder+os.sep+target+'_overview.csv',index=False)
    df_med.to_csv(savefolder+os.sep+target+'_med.csv',index=False)
    
    print('Saved to {}'.format(savefolder+os.sep+target+'_overview.csv'))
    print('Saved to {}'.format(savefolder+os.sep+target+'_med.csv'))
    return df, df_med

def run_specmatch_for_orders(targetfile, targetname, outputdirectory='specmatch_results', HLS=None, 
                             path_df_lib=config.PATH_LIBRARY_DB, orders = ['4','5','6','14','15','16','17'],
                             maxvsini=30.,calibrate_feh=True,scaleres=1.):
    """
    run hpfspecmatch for a given target file and orders
    
    INPUT:
        targetfile - name of target file
        targetname - target name, queried via simbad or tic ('GJ_251' or TIC 68581262)
        outputdirectory - folder to save overall results and plots
        HLS - refence stars as an HPFSpecList object, defaults to normal library
        path_df_lib - path to .csv file containing info on Teff/FeH/logg for all library stars
                    - defaults to config.PATH_LIBRARY_DB
        orders - hpf orders to run (orders 4, 5, 6, 14, 15, 16, and 17
                    recommended as they are the cleanest orders with minimal tellurics)
        maxvsini - maximum vsini to consider (default = 30 km/s)
    
    OUTPUT:
        result files will be saved to outputdirectory
    
    EXAMPLE:
        filename = '../input/20201020_hpf_gto_targets/Slope-20200114T091114_R01.optimal.fits'
        targetname = 'GJ_251'
        HLS = hpfspec.HPFSpecList(filelist=library_fitsfiles)
        outputdir = '20201020_hpf_gto_targets/GJ_251'
        run_specmatch_for_orders(filename, targetname , outputdir, HLS)
    
    NOTES:
        targetname will be queried via simbad or tic which saves a configuration file to target config directory
    
    """
    # Target data
    Htarget = hpfspec.HPFSpectrum(targetfile,targetname = targetname)

    print('Reading Library DataBase from: {}'.format(path_df_lib))
    df_lib = pd.read_csv(path_df_lib)
    
    # Reference data
    if HLS is None:
        print('No HLS supplied, defaulting to default library')
        HLS = hpfspec.HPFSpecList(filelist=config.LIBRARY_FITSFILES)
        Hrefs   = HLS.splist

    # Run spectral matching algorithm for first two orders
    # in principle we should run all orders, just first two as an example
    for o in orders:
        print("##################")
        print("Order {}".format(o))
        print("##################")
        wmin = config.BOUNDS[o][0] # Lower wavelength bound in A
        wmax = config.BOUNDS[o][1] # Upper wavelength bound in A
        ww = np.arange(wmin,wmax,0.01)   # Wavelength array to resample to
        v = np.linspace(-125,125,1501)   # Velocities in km/s to use for absolute RV consideration
        savefolder = '{}/{}_{}/'.format(outputdirectory,Htarget.object,o) # foldername to save

        #############################################################
        # Run specmatch for order 
        #############################################################
        t,f,l,vis,te,fe,le,df_chi,LCS = run_specmatch(Htarget,   # Target class
                                                      HLS.splist,# Library spectra
                                                      ww,        # Wavelength to resample to
                                                      v,         # velocity range to use for absolute rv
                                                      df_lib,    # dataframe with info on Teff/FeH/logg for the library stars
                                                      savefolder=savefolder,
                                                      maxvsini=maxvsini,
                                                      calibrate_feh=calibrate_feh,
                                                      scaleres=scaleres)

        
def plot_crossvalidation_results_1d(order,df_crossval,savefolder):
    """
    
    """
    print('plotting 1D crossval')
    fig, (ax,bx,cx) = plt.subplots(nrows=3,sharex=True,dpi=200)
    x =range(len(df_crossval))
    label = 'std={:0.2f}K'.format(np.std(df_crossval.d_teff))
    print(label)
    ax.plot(df_crossval.d_teff,'k.',markersize=8)
    ax.set_ylim(ax.get_ylim()[0]*1.6,ax.get_ylim()[1]*2.2)

    label = 'std={:0.2f}dex'.format(np.std(df_crossval.d_feh))
    print(label)
    bx.plot(df_crossval.d_feh,'k.',markersize=8)
    bx.set_ylim(bx.get_ylim()[0]*2.0,bx.get_ylim()[1]*2.2)

    label = 'std={:0.2f}dex'.format(np.std(df_crossval.d_logg))
    print(label)
    cx.plot(df_crossval.d_logg,'k.',markersize=8)
    cx.set_ylim(cx.get_ylim()[0]*2.2,cx.get_ylim()[1]*2.2)

    for xx in (ax,bx,cx):
        utils.ax_apply_settings(xx)
        xx.tick_params(labelsize=9,pad=2)
        xx.grid(lw=0.3)
        #xx.legend(bbox_to_anchor=(1.,1.),fontsize=12)
    fig.subplots_adjust(hspace=0.05)
    ax.set_ylabel('$\Delta$Teff [K]',fontsize=12,labelpad=-4)
    bx.set_ylabel('$\Delta$Fe/H [K]',fontsize=12,labelpad=-2)
    cx.set_ylabel('$\Delta$logg [K]',fontsize=12,labelpad=-2)
    cx.set_xlabel('Library spectrum #',fontsize=12)
    ax.set_title('Library Performance (HPF Order {})'.format(order))
    plt.savefig('{}/crossvalidation_o{}_plot1D.png'.format(savefolder,order))
    
def plot_crossvalidation_results_2d(order,df_crossval,savefolder):
    """
    """
    print('plotting 2D crossval')
    dd = df_crossval[["d_teff","d_feh","d_logg"]]
    fig, axx = plt.subplots(nrows=3,ncols=3,dpi=200,figsize=(6,6))

    for xx in [axx[0,1],axx[0,2],axx[1,2]]:
        xx.axes.set_axis_off()

    labels = ["$\Delta T_{\mathrm{eff}}$","$\Delta$[Fe/H]","$\Delta \log g$"]
    xlabels = ["$\Delta T_{\mathrm{eff}}$ [K]","$\Delta$[Fe/H] [dex]","$\Delta \log g$ [dex]"]

    #xlims = [xx.get_xlim() for xx in [axx[1,0],axx[2,0],axx[2,1]]]
    xlims = []
    diag = [axx[0,0],axx[1,1],axx[2,2]]
    for i,xx in enumerate(diag):
        xx.set_title(labels[i],fontsize=12)
        _y, _x,_ = xx.hist(dd.iloc[:,i].values,color="black",histtype="step",bins=7,
                           range=[dd.iloc[:,i].values.min(),dd.iloc[:,i].values.max()])
        g = astropy.modeling.models.Gaussian1D(stddev=np.std(dd.iloc[:,i].values))
        xmax = np.max(np.abs(dd.iloc[:,i].values))
        x = np.linspace(-xmax,xmax,1000)
        y = (g(x)/(np.max(g(x))))*(np.max(_y))
        xx.plot(x,y,color="crimson",lw=1,ls="--")

        if i == 0: 
            xx.xaxis.set_visible(False)
            xx.yaxis.set_visible(False)    
            #xx.set_xlim(xlims[0][0],xlims[0][1])
        if i == 1: 
            xx.xaxis.set_visible(False)
            xx.yaxis.set_visible(False)
            #xx.set_xlim(xlims[2][0],xlims[2][1])
        if i == 2: 
            xx.yaxis.set_visible(False)
        xlims.append(xx.get_xlim())

    axx[1,0].plot(df_crossval.d_teff.values,df_crossval.d_feh.values,marker="o",lw=0,color="k",alpha=0.8)
    axx[1,0].set_xlim(xlims[0][0],xlims[0][1])
    axx[2,0].plot(df_crossval.d_teff.values,df_crossval.d_logg.values,marker="o",lw=0,color="k",alpha=0.8)
    axx[2,0].set_xlim(xlims[0][0],xlims[0][1])
    axx[2,1].plot(df_crossval.d_feh.values,df_crossval.d_logg.values,marker="o",lw=0,color="k",alpha=0.8)
    axx[2,1].set_xlim(xlims[1][0],xlims[1][1])

    axx[2,1].yaxis.set_visible(False)
    axx[1,0].xaxis.set_visible(False)

    axx[1,0].set_ylabel(xlabels[1],fontsize=12,labelpad=3)
    axx[2,0].set_ylabel(xlabels[2],fontsize=12,labelpad=-2)
    axx[2,0].set_xlabel(xlabels[0],fontsize=12)
    axx[2,1].set_xlabel(xlabels[1],fontsize=12)
    axx[2,2].set_xlabel(xlabels[2],fontsize=12)

    for xx in axx.flatten():
        utils.ax_apply_settings(xx,ticksize=10)
        xx.grid(lw=0)
        xx.tick_params(pad=1)

    fig.subplots_adjust(wspace=0.02,hspace=0.02)
    plt.savefig('{}/crossvalidation_o{}_plot2D.png'.format(savefolder,order))
    
def run_crossvalidation_for_orders(order, df_lib=config.PATH_LIBRARY_DB, HLS=None,
                                   outputdir=config.PATH_LIBRARY_CROSSVAL,
                                   plot_results = True,
                                   calibrate_feh=True,
                                   scaleres=1.):
    """
    Run cross validation for a given order
    
    Here, cross validation is performed using a "leave one out" approach.
    We loop through the full stellar library leaving one library star out,
    where we try and recover the stellar parameters for that spectrum and compare it with the known values
    
    INPUT:
        order - order to run cross validation for (orders 4, 5, 6, 14, 15, 16, and 17
                    recommended as they are the cleanest orders with minimal tellurics)
        df_lib - dataframe with info on Teff/FeH/logg for the library stars
        HLS - refence stars as an HPFSpecList object
        outputdir - folder to save overall results and plots
        plot_results - plots figures summarizing the cross validation results
        scaleres - amount to scale residuals from composite spectrum (default 1)
    
    OUTPUT:
        returns df_crossval dataframe with columns 'teff','feh','logg','vsini',
            'd_teff','d_feh','d_logg' where d_*param* is the difference between the known value
             and the recovered value from the cross validation
        df_crossval saved to csv in outputdir
    
    EXAMPLE:
        order = '17' # recommended order
        
        LIBRARY_DIR = '../library/20201008_specmatch_nir/'
        
        df_lib = pd.read_csv(LIBRARY_DIR+'20201008_specmatch_nir.csv')
        library_fitsfiles = glob2.glob(LIBRARY_DIR+'FITS/*/*.fits')
        HLS = hpfspec.HPFSpecList(filelist=library_fitsfiles)
        
        outputdir = 'cross_validation_{}'.format(order)
        
        run_crossvalidation_for_orders(order, df_lib,HLS, outputdir)
        
    
    NOTES:
        
    
    """
    wmin = config.BOUNDS[order][0]
    wmax = config.BOUNDS[order][1]
    ww = np.arange(wmin,wmax,0.01)
    v = np.linspace(-125,125,1501)
    res = []
    obj_names = []
    
    outputdir = '{}/o{}_crossval'.format(outputdir, order)
    
    # Reference data
    if HLS is None:
        print('No HLS supplied, defaulting to default library')
        HLS = hpfspec.HPFSpecList(filelist=config.LIBRARY_FITSFILES)
    
    # Looping over every star in the library
    for i in range(len(df_lib)):
        print(i)
        # Target data
        Htarget = HLS.splist[i]
        df_target = df_lib[df_lib['OBJECT_ID'] == Htarget.object]
        # Removing the target star from the library
        Hrefs   = np.delete(np.array(HLS.splist),i)
        # Run specmatch without the target star in the library
        _res = run_specmatch(Htarget,Hrefs,ww,v,df_lib,df_target,plot=True,
                             savefolder='{}/plots/'.format(outputdir),
                             calibrate_feh=calibrate_feh,
                             scaleres=scaleres)
        res.append(_res)
        obj_names.append(Htarget.object)
        
    # Collect all of the results and save
    df_crossval = pd.DataFrame(res,columns=['teff','feh','logg','vsini','d_teff','d_feh','d_logg','_','__'])
    df_crossval = df_crossval[['teff','feh','logg','vsini','d_teff','d_feh','d_logg']]
    df_crossval['teff_true'] = df_crossval.teff - df_crossval.d_teff
    df_crossval['feh_true'] = df_crossval.feh - df_crossval.d_feh
    df_crossval['logg_true'] = df_crossval.logg - df_crossval.d_logg
    df_crossval['targetname'] = obj_names

    fig, ax = plt.subplots(dpi=200)
    plot_crossval_feh_delta_feh(df_crossval['feh_true'].values,df_crossval['d_feh'].values,ax=ax)
    fig.savefig('{}/crossvalidation_feh_delta_feh_o{}.png'.format(outputdir,order),dpi=200)
    
    result_savename = '{}/crossvalidation_results_o{}.csv'.format(outputdir,order)
    df_crossval.to_csv(result_savename, index=False)
    print('Saved tesult to: {}'.format(result_savename))
    
    if plot_results == True:
        plot_crossvalidation_results_1d(order,df_crossval,outputdir)
        plot_crossvalidation_results_2d(order,df_crossval,outputdir)
        plot_crossvalidation_results_main(order,df_crossval,outputdir)
        
    
    return df_crossval

def detrend_feh(feh):
    """
    Calibrate trend in (Delta Fe/H = Fe/H_recovered - Fe/H_true) vs Fe/H_true
    
    INPUT:
        recovered feh value from specmatch
        
    OUTPUT:
        calibrated feh after removing trend
        
    NOTES:
        Currently correct for order 5
    """
    # Paramters for order 5
    p1 = -0.28935366
    p0 = -0.00095669
    a = (1./(p1+1.))  # 1.4071697041315936
    b = -(p0/(p1+1.)) # -0.0013462251842456543
    return a*feh + b

def plot_crossvalidation_results_main(order,df_crossval,savefolder):
    """
    Main crossvalidation results plot
    """
    PW = 10.
    PH = 3.
    fig = plt.figure(figsize=(PW,PH),dpi=200)
    gs0 = GridSpec(1,2)
    gs0.update(top=0.92, bottom = 0.11,wspace=0.01,left=0.05,right=0.6)

    gs1 = GridSpec(1,1)
    gs1.update(top=0.92, bottom = 0.11,hspace=0.18,left=0.68,right=0.98)

    ax = plt.subplot(gs0[0,0])
    bx = plt.subplot(gs0[0,1])
    cx = plt.subplot(gs1[0,0])
    
    ax.plot(df_crossval['feh_true'],df_crossval['teff_true'],marker='o',lw=0,markeredgecolor='black')
    bx.plot(df_crossval['logg_true'],df_crossval['teff_true'],marker='o',lw=0,markeredgecolor='black')
    for i in range(len(df_crossval)):
        _x = [df_crossval['feh_true'].values[i],df_crossval['feh'].values[i]]
        _y = [df_crossval['teff_true'].values[i],df_crossval['teff'].values[i]]
        ax.plot(_x,_y,color='crimson',lw=0.5,zorder=-10)

        _x = [df_crossval['logg_true'].values[i],df_crossval['logg'].values[i]]
        if i == 0:
            bx.plot(_x,_y,color='crimson',lw=0.5,label='Crossvalidation Value',zorder=-10)
        else:
            bx.plot(_x,_y,color='crimson',lw=0.5,zorder=-10)

    ax.set_xlim(-0.6,0.6)
    ax.set_xlabel('[Fe/H]',labelpad=0)
    ax.set_ylabel('$T_{\mathrm{eff}}$ [K]',labelpad=0)
    bx.set_xlabel('$\log (g)$',labelpad=0)
    ax.minorticks_on()
    bx.minorticks_on()
    bx.yaxis.set_ticks([])
    bx.set_ylim(*ax.get_ylim())
    fig.subplots_adjust(wspace=0.05)
    bx.legend(loc='upper right')
    fig.suptitle('Library Performance (HPF Order {})'.format(order),y=0.98)
    plot_crossval_feh_delta_feh(df_crossval.feh_true.values,df_crossval.d_feh.values,ax=cx)
    fig.savefig('{}/crossvalidation_o{}_main.png'.format(savefolder,order))

def plot_crossval_feh_delta_feh(feh_true,d_feh,ax=None):
    """
    INPUT:
        feh_true - True Fe/H
        d_feh - delta Fe/H = Fe/H_specmatch - Fe/H_True

    EXAMPLE:
        plot_crossval_feh_delta_feh(df_res['[Fe/H]'].values,df_res['d_feh'].values)
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=200)
    ax.plot(feh_true,d_feh,marker='o',lw=0,color='black')
    
    p = np.polyfit(feh_true,d_feh,deg=1)
    xx = np.linspace(-0.5,0.5,200)
    yy = np.polyval(p,xx)
    ax.plot(xx,yy,color='crimson',label='Linear fit\n$p_1$={:0.4f}\n$p_2$={:0.4f}'.format(p[0],p[1]))
    ax.legend(loc='upper right')
    ax.set_xlabel('[Fe/H]',labelpad=0)
    ax.set_ylabel('$\Delta$[Fe/H] = $\mathrm{[Fe/H]}_{\mathrm{Recovered}}$ - $\mathrm{[Fe/H]}_{\mathrm{True}}$')
    ax.minorticks_on()
