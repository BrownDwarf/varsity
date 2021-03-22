code
---

The file `batch.py` contains scripts to run Starfish in *batch* mode.

Here are the arguments:

```
-h, --help            show this help message and exit
--convert_csv_to_hdf5_files
                      Convert csv .dat files to HDF5 files
--set_wl_lims         Set the wavelength limits of the spectra
--config              Generate config files
--grid_create         Run grid.py create
--train_pca           Create, train, and store the PCA emulator
--init_run_dirs       Initialize output and run directories
--generate_s0_o0phi_file
                      Generate the s0_o0phi file
--save_csv_spectrum   Save a model spectrum to a csv file
--revise_nuisance_params
                      Manually revise the nuisance parameters with
                      heuristics
--write_user_prior    Write the user_prior.py file from limits in the Excel
                      sheet
--run_Starfish        Run Starfish spectral inference
--qsub_PBS            Write PBS script and run Starfish spectral inference
--diagnose            Produce posterior predictive diagnostics after a
                      Starfish run
```

Right now you have to *manually* set the  
`run_num` and `orders`  
arguments at the top of the `batch.py` file.  Sorry!

You should run them in sequence like so:

```
python batch.py --convert_csv_to_hdf5_files
python batch.py --set_wl_lims
python batch.py --config
python batch.py --grid_create
python batch.py --train_pca         # this step takes a while
python batch.py --init_run_dirs --run_num=1
python batch.py --generate_s0_o0phi_file --run_num=1
python batch.py --save_csv_spectrum --run_num=1
python batch.py --revise_nuisance_params --run_num=1
python batch.py --write_user_prior --run_num=1
```

After that you can:   
`python batch.py --run_Starfish --run_num=1` as a proof-of concept, then  

`python batch.py --qsub_PBS --run_num=1` on the super computer

Finally, you can make some plots with:  
`python batch.py --diagnose`
