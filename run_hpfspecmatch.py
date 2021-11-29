import hpfspec
import hpfspecmatch
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="HPF SpecMatch: Match")
    parser.add_argument("filename",type=str,help="filename to reduce")
    parser.add_argument("object",type=str,help="HPF object to reduce (SIMBAD or TIC Queryable)")
    parser.add_argument("--savefolder",type=str,default="specmatch_results",help="Specify foldername to save (e.g. results_123)")
    parser.add_argument("--orders",type=int,default=[4,5,6,14,15,16,17],help="Orders to use for HPF SpecMatch, e.g., --orders 4 5 6",nargs='+')
    parser.add_argument("--vsinimax",type=int,default=40.,help="Maximum vsini to fit for in km/s")
    parser.add_argument("--calibrate_feh",default=False,help="Calibrate the Fe/H",action="store_true")
    parser.add_argument("--scaleres",type=float,default=1.,help="Residual Scaling Factor")

    args = parser.parse_args()

    # Make sure library is availabe, if not, download it
    hpfspecmatch.get_library()

    filename = args.filename
    targetname = args.object
    outputdir = args.savefolder
    orders = [str(i) for i in args.orders]
    path_df_lib = hpfspecmatch.config.PATH_LIBRARY_DB
    maxvsini = args.vsinimax
    calibrate_feh = args.calibrate_feh
    scaleres = args.scaleres

    # Run specmatch for orders
    hpfspecmatch.run_specmatch_for_orders(targetfile=filename,
                                          targetname=targetname,
                                          outputdirectory=outputdir,
                                          path_df_lib=path_df_lib,
                                          orders=orders,
                                          maxvsini=maxvsini,
                                          calibrate_feh=calibrate_feh,
                                          scaleres=scaleres)
