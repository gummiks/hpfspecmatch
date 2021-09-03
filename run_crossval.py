import hpfspec
import hpfspecmatch
import hpfspecmatch.config
import argparse
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="HPF SpecMatch: Cross Validation")
    parser.add_argument("order",type=int,default=17,help="Order to use for cross validation, e.g., order 17")
    parser.add_argument("--df_lib",type=str,default=hpfspecmatch.config.PATH_LIBRARY_DB,help="Path to stellar library csv, e.g., 20201008_specmatch_nir_library.csv")
    parser.add_argument("--HLS",type=str,default=hpfspecmatch.config.LIBRARY_FITSFILES,help="List of stellar library fits files, e.g., hpfspecmatch.config.LIBRARY_FITSFILES")
    parser.add_argument("--savefolder",type=str,default=hpfspecmatch.config.PATH_LIBRARY_CROSSVAL,help="Specify foldername to save (e.g. o17_crossval)")
    parser.add_argument("--plot_results",default=True,help="Save cross validation summary plots",action="store_true")
    parser.add_argument("--calibrate_feh",default=True,help="Calibrate the Fe/H",action="store_true")
    parser.add_argument("--scaleres",type=float,default=1.,help="Residual Scaling Factor")

    args = parser.parse_args()

    # Make sure library is available, if not, download it
    hpfspecmatch.get_library()

    order = str(args.order)
    df_lib = pd.read_csv(args.df_lib)
    HLS = hpfspec.HPFSpecList(filelist=hpfspecmatch.config.LIBRARY_FITSFILES)
    outputdir = args.savefolder
    plot_results = args.plot_results
    calibrate_feh = args.calibrate_feh
    scaleres = args.scaleres

    # Run cross validation for orders
    hpfspecmatch.run_crossvalidation_for_orders(order=order,
                                                df_lib=df_lib,
                                                HLS=HLS,
                                                outputdir=outputdir,
                                                plot_results=plot_results,
                                                calibrate_feh=calibrate_feh,
                                                scaleres=scaleres)
