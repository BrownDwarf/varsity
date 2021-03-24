 #!/usr/bin/env python
import os
import yaml
import numpy as np
import h5py
from scipy.signal import medfilt
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from muler.igrins import IGRINSSpectrumList

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
parser = argparse.ArgumentParser(prog="batch.py", description="Batch common starfish tasks")
parser.add_argument("--run_num", type=int, default=None, help="Which run_number, defaults to run01")
parser.add_argument("--convert_fits_to_hdf5_files", action="store_true", help="Convert csv .dat files to HDF5 files")
parser.add_argument("--set_wl_lims", action="store_true", help="Set the wavelength limits of the spectra")
parser.add_argument("--config", action="store_true", help="Generate config files")
parser.add_argument("--grid_create", action="store_true", help="Run grid.py create")
parser.add_argument("--train_pca", action="store_true", help="Create, train, and store the PCA emulator")
parser.add_argument("--init_run_dirs", action="store_true", help="Initialize output and run directories")
parser.add_argument("--generate_s0_o0phi_file", action="store_true", help="Generate the s0_o0phi file")
parser.add_argument("--save_csv_spectrum", action="store_true", help="Save a model spectrum to a csv file")
parser.add_argument("--revise_nuisance_params", action="store_true",
                    help="Manually revise the nuisance parameters with heuristics")
parser.add_argument("--write_user_prior", action="store_true",
                    help="Write the user_prior.py file from limits in the Excel sheet")
parser.add_argument("--run_Starfish", action="store_true", help="Run Starfish spectral inference")
parser.add_argument("--qsub_PBS", action="store_true", help="Write PBS script and run Starfish spectral inference")
parser.add_argument("--diagnose", action="store_true", help="Produce posterior predictive diagnostics after a Starfish run")
args = parser.parse_args()

src_dict = {'luhman16A_btjd_2285p65':'GS-2021A-DD-104/20210311/reduced/SDC{}_20210311_0025.spec_a0v.fits'} # hardcoded for now

sources = ['luhman16A_btjd_2285p65']
orders = [103, 104, 105]
run_num = args.run_num


def modify_and_save_excel_file(source, order, keyword, new_value):
    '''Read in entire Excel file, change single entry, and resave'''
    db_init_file = os.path.expandvars('$varsity/db/varsity_init.xlsx')
    xlf = pd.ExcelFile(db_init_file)
    sheets = xlf.sheet_names
    xlf.close()
    writer = pd.ExcelWriter(db_init_file)
    for sheet in sheets:
        df = pd.read_excel(db_init_file, sheet_name=sheet, index_col=0)
        if sheet == source:
            df.at[keyword, "m{:03d}".format(order)] = new_value
        df.to_excel(writer, sheet_name=sheet)
    writer.save()

def cd_to(source, order, run_num=None):
    '''Change directories to the source/order, create if needed'''
    path_src = os.path.expandvars('$varsity/sf/{}'.format(source))
    path_out = os.path.expandvars('$varsity/sf/{}/m{}'.format(source, order))
    os.makedirs(path_src, exist_ok=True)
    os.makedirs(path_out, exist_ok=True)
    os.chdir(path_out)
    if run_num is not None:
        run_dir = 'output/marley_grid/run{:02d}'.format(run_num)
        os.chdir(run_dir)


def generate_s0_o0phi_file(source, order, run_num):
    '''Programmatically generate the s0_o0phi.json files'''
    cd_to(source, order)

    m = 'm{:03d}'.format(order)

    with open(os.path.expandvars('$varsity/db/s0_o0phi_template.json')) as f:
        jf = json.load(f)

    db_init_file = os.path.expandvars('$varsity/db/varsity_init.xlsx')
    df = pd.read_excel(db_init_file, sheet_name=source, index_col=0)

    jf['sigAmp'] = float(df[m]['sigAmp'])
    jf['logAmp'] = float(df[m]['logAmp'])
    jf['l'] = float(df[m]['ll'])

    file_out = os.path.expandvars('$varsity/sf/{}/{}/output/marley_grid/run{:02d}/'\
                's0_o0phi.json'.format(source, m, run_num))
    with open(file_out, mode='w') as f:
        json.dump(jf, f, indent=2)


def generate_config_yaml(source, order, run_num=None):
    '''Programmatically generate the config files'''
    cd_to(source, order, run_num)

    m = 'm{:03d}'.format(order)

    f = open(os.path.expandvars('$varsity/db/config_template.yaml'))
    config = yaml.load(f)
    f.close()

    db_init_file = os.path.expandvars('$varsity/db/varsity_init.xlsx')
    df = pd.read_excel(db_init_file, sheet_name=source, index_col=0)

    config['data']['files'] = ['$varsity/data/homoscedastic/{}_{:03d}.hdf5'.format(source, order)]
    config['grid']['hdf5_path'] = '$varsity/sf/{}/m{:03d}/libraries/Bobcat_grid.hdf5'.format(source, order)
    lb = int(np.floor(df['m{}'.format(order)]['wl_lo']))
    ub = int( np.ceil(df['m{}'.format(order)]['wl_hi']))

    config['grid']['wl_range'] = [lb, ub]
    config['grid']['parrange'] = [[int(df[m]['Teff_lo']), int(df[m]['Teff_hi'])],
                                  [float(df[m]['logg_lo']), float(df[m]['logg_hi'])]]
    config['PCA']['path'] = '$varsity/sf/{}/m{:03d}/libraries/Bobcat_PCA.hdf5'.format(source, order)

    if run_num is None:
        config['Theta_priors'] = '$varsity/sf/{}/m{:03d}/user_prior.py'.format(source, order)
    else:
        prior_out = '$varsity/sf/{}/m{:03d}/output/marley_grid/run{:02d}/user_prior.py'
        config['Theta_priors'] = prior_out.format(source, order, run_num)
    config['Theta']['grid'] = [int(df[m]['Teff']), float(df[m]['logg'])]
    config['Theta']['logOmega'] = float(df[m]['logOmega'])
    config['Theta']['vsini'] = float(df[m]['vsini'])
    config['Theta']['vz'] = float(df[m]['vz'])

    with open('config.yaml', mode='w') as outfile:
        outfile.write(yaml.dump(config))


def run_grid_create(source, order, background_process=True):
    '''Run grid.py --create for source and order'''
    cd_to(source, order)

    os.makedirs('libraries', exist_ok=True)
    cmd = '$Starfish/scripts/grid.py --create > grid.out'
    if background_process:
        os.system(cmd +' &')
    else:
        os.system(cmd)

def train_pca(source, order, n_samples=3):
    '''Run grid.py --create for source and order'''
    cd_to(source, order)

    cmd1 = '$Starfish/scripts/pca.py --create > pca_create.out'
    cmd2 = '$Starfish/scripts/pca.py --optimize=emcee --samples={} '  \
        '> pca_optimize.out'.format(n_samples)
    cmd3 = '$Starfish/scripts/pca.py --store --params=emcee > pca_store.out'

    for cmd in [cmd1, cmd2, cmd3]:
        os.system(cmd)


def init_run_dirs(source, order, run_num=1):
    '''Run grid.py --create for source and order'''
    cd_to(source, order)

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/marley_grid', exist_ok=True)
    run_dir = 'output/marley_grid/run{:02d}'.format(run_num)
    os.makedirs(run_dir, exist_ok=True)
    generate_config_yaml(source, order, run_num=run_num)


def run_Starfish(source, order, run_num=1, samples=3, incremental_save=1):
    '''Run grid.py --create for source and order'''
    cd_to(source, order)
    run_dir = 'output/marley_grid/run{:02d}'.format(run_num)
    os.chdir(run_dir)
    os.system('$varsity/code/star_marley_beta.py '\
            '--samples={} --incremental_save={}'.format(samples, incremental_save))

def revise_nuisance_params(source, order, run_num=1):
    '''Tune the nuisance parameters'''
    cd_to(source, order, run_num)
    df_spec = pd.read_csv('spec_config.csv')

    db_init_file = os.path.expandvars('$varsity/db/varsity_init.xlsx')
    df_init = pd.read_excel(db_init_file, sheet_name=source, index_col=0)
    m = 'm{:02}'.format(order)
    logOmega_delta = np.log10(df_spec['model_composite'].median()/df_spec['data'].median())
    new_logOmega = df_init[m]['logOmega'] - logOmega_delta
    modify_and_save_excel_file(source, order, 'logOmega', new_logOmega)
    generate_config_yaml(source, order, run_num=run_num)

    ## TODO: revise sigAmp, logAmp, and ell
    ##       right now they are revised manually after a first run.

def set_wl_lims(source, order):
    '''Sets the wavelength limits based on the input spectra'''
    cd_to(source, order)

    #dat_fn = os.path.expandvars('$varsity/data/varsityhighres_all/{}_{:02d}.dat'.format(source, order),)
    #df = pd.read_csv(dat_fn, comment='#',
#            delim_whitespace=True, names = ['wl_um', 'flux'])

    # Make outname based on the basename
    file_name = '{}_{:03d}.hdf5'.format(source, order)


    out_path = os.path.expandvars('$varsity/data/homoscedastic/')
    file = h5py.File(out_path+file_name, 'r')

    wl_lo, wl_hi = (file['wls'][j] for j in [0, -1])
    wl_lo, wl_hi = int(np.floor(wl_lo)), int(np.ceil(wl_hi))
    file.close()

    modify_and_save_excel_file(source, order, 'wl_lo', wl_lo)
    modify_and_save_excel_file(source, order, 'wl_hi', wl_hi)


def save_csv_spectrum(source, order, run_num=1, inference=False):
    '''Tune the nuisance parameters'''
    cd_to(source, order, run_num)
    os.system('$varsity/code/plot_specific_mix_model_marley.py --config')
    if inference:
        os.system('$varsity/code/plot_specific_mix_model_marley.py --static')


def write_user_prior(source, order, run_num=None):
    '''write the user_prior.py file programmatically'''
    cd_to(source, order, run_num)
    m = 'm{:03d}'.format(order)

        # Read in the file
    with open(os.path.expandvars('$varsity/db/user_prior_template.py'), 'r') as file:
      filedata = file.read()

    # Replace the target string
    limits = ['Teff_lo','Teff_hi','logg_lo','logg_hi','logOmega_lo','logOmega_hi',
            'vsini_lo','vsini_hi','vz_lo','vz_hi','sigAmp_lo','sigAmp_hi',
            'logAmp_lo','logAmp_hi','ll_lo','ll_hi']

    db_init_file = os.path.expandvars('$varsity/db/varsity_init.xlsx')
    df = pd.read_excel(db_init_file, sheet_name=source, index_col=0)
    # for now we only support source-level priors.  order-level priors is possible.
    fmt_dict = {'Teff':"{:.0f}", 'logg':"{:.2f}", 'vz':"{:.1f}", 'vsini':"{:.1f}",
                'logOmega':"{:.1f}", 'sigAmp':"{:0.3f}", 'logAmp':"{:.2f}",
                'll':"{:.1f}"}

    for key in list(fmt_dict.keys()):
        fmt_dict[key+'_lo'], fmt_dict[key+'_hi'] = fmt_dict[key], fmt_dict[key]

    for limit in limits:
        new_value = fmt_dict[limit].format(df.at[limit, m])
        filedata = filedata.replace(limit, new_value)

    # Write the file out again
    with open('user_prior.py', 'w') as file:
      file.write(filedata)


def write_PBS_script(source, order, run_num):
    '''write the user_prior.py file programmatically'''
    cd_to(source, order, run_num=run_num)

    # Read in the file
    with open(os.path.expandvars('$varsity/db/pbs_template.sh'), 'r') as file:
        filedata = file.read()

    new_value = "varsity{}_m{:03d}_r{:02d}".format(source, order, run_num)
    filedata = filedata.replace('JOBNAME_HERE', new_value)

    filedata = filedata.replace('N_SAMPLES', "5000")
    filedata = filedata.replace('INC_SAVE', "100")

    with open('PBS.sh', 'w') as file:
        file.write(filedata)

def diagnose(source, order, run_num):
    '''Produce posterior predictive checks'''
    plot_path = os.path.expandvars('$varsity/results')
    os.makedirs(plot_path, exist_ok=True)
    os.chdir(plot_path)
    emcee_path = "$varsity/sf/{}/m{:03d}/output/marley_grid/run{:02d}/emcee_chain.npy"
    ws = np.load(os.path.expandvars(emcee_path.format(source, order, run_num)))
    burned = ws[:, :,:]
    xs, ys, zs = burned.shape
    fc = burned.reshape(xs*ys, zs)
    nx, ny = fc.shape
    label = [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$",
         r"$c^1$", r"$c^2$", r"$c^3$", r"sigAmp", r"logAmp", r"$l$"]
    fig, axes = plt.subplots(11, 1, sharex=True, figsize=(8, 14))
    for i in range(0, 11, 1):
        axes[i].plot(burned[:, :, i].T, color="k", alpha=0.2)
        axes[i].yaxis.set_major_locator(MaxNLocator(5))
        axes[i].set_ylabel(label[i])

    #TODO: set axes limits by the Excel _lo and _hi limits

    axes[10].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    out_name = "varsity_{}_m{:03d}_run{:02d}_chain.png".format(source, order, run_num)
    plt.savefig(out_name, dpi=300, bbox_inches='tight')

    #TODO: Plot the spectra


if args.convert_fits_to_hdf5_files:
    for source in sources:
        for band in ["H", "K"]:
            directory = os.path.expandvars('$varsity/data/IGRINS/originals/')
            filename = src_dict[source].format(band)
            full_path = directory + filename
            print(full_path)
            spec = IGRINSSpectrumList.read(full_path)
            spec = spec.normalize().remove_nans().trim_edges()

## Main program
for source in sources:
    print(source)
    for order in orders:
        print(order)


        if args.set_wl_lims:
            set_wl_lims(source, order)

        if args.config:
            generate_config_yaml(source, order)

        if args.grid_create:
            run_grid_create(source, order, background_process=False)

        if args.train_pca:
            train_pca(source, order, n_samples=1)

        if args.init_run_dirs:
            init_run_dirs(source, order, run_num=run_num)

        if args.generate_s0_o0phi_file:
            generate_s0_o0phi_file(source, order, run_num)

        if args.save_csv_spectrum:
            save_csv_spectrum(source, order, run_num=run_num)

        if args.revise_nuisance_params:
            revise_nuisance_params(source, order, run_num=run_num)

        if args.write_user_prior:
            write_user_prior(source, order, run_num)

        if args.run_Starfish:
            run_Starfish(source, order, run_num=run_num, samples=5000, incremental_save=50)

        if args.qsub_PBS:
            write_PBS_script(source, order, run_num=run_num)
            os.system('pwd')
            os.system('qsub PBS.sh')

        if args.diagnose:
            save_csv_spectrum(source, order, run_num=run_num, inference=True)
            diagnose(source, order, run_num)
