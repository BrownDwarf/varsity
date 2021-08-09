import numpy as np
import pandas as pd
import glob
import astropy.units as u
import os

from pandas.core import base
from muler.igrins import IGRINSSpectrum

from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Slider, Range1d
from bokeh.layouts import layout, Spacer
from bokeh.models.widgets import Button, Div

from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict

## We can optionally cache the spectra
## This should be faster, but is only slightly faster
precache = False  # Toggle for caching

# Data
band = "K"
data_path = "../../data/IGRINS/originals/GS-2021A-DD-104/*/reduced/SDC{}*.spec_a0v.fits".format(
    band
)
reduced_fns = glob.glob(data_path)
reduced_fns = sorted(reduced_fns)
reduced_fns = reduced_fns[slice(0, 8, 2)]
print(reduced_fns)
#reduced_fns = reduced_fns[slice(1, 8, 2)]

spec1 = (
    IGRINSSpectrum(file=reduced_fns[3], order=13).remove_nans().normalize().trim_edges()
)

## The subset with overlapping points
teff_points = np.array([1000, 1300, 1600, 1700, 2100, 2200])
logg_points = np.arange(3.75, 4.51, 0.25)
logg_par_dict = {3.75: "56", 4.0: "100", 4.25: "178", 4.5: "316"}
teff_dict = OrderedDict()
for i, teff in enumerate(teff_points):
    teff_dict[i] = str(teff)

cloudy_suffix_dict = {False: "_cloudfree.spec", True: ".spec"}
models_path = os.path.expandvars("$varsity/models/models_forGully/")
fns_morley = sorted(glob.glob(models_path + "/t*.spec"))


def basename_constructor(teff, logg, cloudy=True):
    """Construct the model basename base on Teff, logg, and cloudy flag"""
    basename = "t{0:0>.0f}g{1:}f2_m0.0_co1.0_4100_5000_0.005{2:}".format(
        np.float(teff), logg_par_dict[logg], cloudy_suffix_dict[cloudy]
    )
    return basename


def get_lookup_dictionaries(fns, cloudy=True):
    """Get lookup dictionaries for each existing filename"""
    basenames = [fn.split("/")[-1] for fn in fns]
    if cloudy:
        suffix = "f2_m0.0_co1.0_4100_5000_0.005.spec"
    if not cloudy:
        suffix = "f2_m0.0_co1.0_4100_5000_0.005_cloudfree.spec"
    basenames = [basename for basename in basenames if suffix in basename]

    teff_labels = [int(basename[1 : 1 + 4]) for basename in basenames]
    g_labels = [int(basename[6:].split("f")[0]) for basename in basenames]

    label_pairs = list(zip(teff_labels, g_labels))

    ## Filename dictionary given a tuple of (teff, g)
    lookup_dict = {key: value for key, value in zip(label_pairs, basenames)}

    ## g dictionary lookup given a teff
    g_lookup_dict = {}
    for teff_label in set(teff_labels):
        g_lookup_dict[teff_label] = []
    for teff_label, g_label in label_pairs:
        g_lookup_dict[teff_label].append(g_label)
    for key in g_lookup_dict.keys():
        g_lookup_dict[key] = sorted(g_lookup_dict[key])

    return (g_lookup_dict, lookup_dict)


def load_and_prep_spectrum(fn, wl_low=2.108, wl_high=2.134, downsample=4):
    df_native = (
        pd.read_csv(
            fn, skiprows=[0, 1], delim_whitespace=True, names=["wavelength", "flux_raw"]
        )
        .sort_values("wavelength")
        .reset_index(drop=True)
    )

    assertion_msg = "We can only model K-band clouds currently: {:0.5f}, {:0.5f}".format(
        wl_low, wl_high
    )
    # assert (wl_low > 2.0) & (wl_high < 2.51), assertion_msg

    ## Standardize flux density units to cgs
    morley_flux_w_units = df_native.flux_raw.values * u.Watt / u.m ** 2 / u.m
    flux_cgs_units = morley_flux_w_units.to(
        u.erg / u.cm ** 2 / u.s / u.Hz,
        equivalencies=u.spectral_density(df_native.wavelength.values * u.micron),
    )
    df_native["flux"] = flux_cgs_units.value

    ## Trim to the wavelength bounds
    nir_mask = (df_native.wavelength > wl_low) & (df_native.wavelength < wl_high)

    ## Decimate the data:
    df_nir = df_native[["wavelength", "flux"]]
    df_nir = df_nir[nir_mask]
    df_nir = df_nir.rolling(int(downsample * 2.5), win_type="gaussian")
    df_nir = df_nir.mean(std=downsample / 2)
    df_nir = df_nir.iloc[::downsample, :]
    df_nir = df_nir.dropna()
    df_nir = df_nir.reset_index(drop=True)
    

    return df_nir


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def create_interact_ui(doc):

    ### Models
    ## Make the spectrum sources
    # Initialize
    teff = 1300
    logg = 4.0
    ff = 0.3
    basename = basename_constructor(np.float(teff), logg, False)

    # Cloud-free
    if band == "H":
        wl_low = 2.1  # np.nanmin(spec1.wavelength.value / 10_000)
        wl_high = 2.12  # np.nanmax(spec1.wavelength.value / 10_000)
    else:
        wl_low = np.nanmin(spec1.wavelength.value / 10_000)
        wl_high = np.nanmax(spec1.wavelength.value / 10_000)
    df_nir = load_and_prep_spectrum(
        models_path + basename, downsample=4, wl_low=wl_low, wl_high=wl_high
    )

    # Cloudy
    basename = basename_constructor(np.float(teff), logg, True)
    df_cloud = load_and_prep_spectrum(
        models_path + basename, downsample=4, wl_low=wl_low, wl_high=wl_high
    )

    # Fixed normalization constant for all spectra
    scalar_norm = np.percentile(df_nir.flux.values, 50)

    spec_source = ColumnDataSource(
        data=dict(
            wavelength=df_nir.wavelength.values,
            flux=gaussian_filter1d(df_nir.flux.values / scalar_norm, 0.1),
            native_flux=df_nir.flux.values / scalar_norm,
            native_wavelength=df_nir.wavelength.values,
        )
    )

    spec_source_cloud = ColumnDataSource(
        data=dict(
            wavelength=df_cloud.wavelength.values,
            flux=gaussian_filter1d(df_cloud.flux.values / scalar_norm, 0.1),
            native_flux=df_cloud.flux.values / scalar_norm,
            native_wavelength=df_cloud.wavelength.values,
        )
    )

    def cloud_mixture_model(ff):
        """Compute the composite spectrum based on cloudy and cloud-free components"""
        composite = (
            ff * spec_source_cloud.data["flux"] + (1 - ff) * spec_source.data["flux"]
        )
        return composite

    spec_source_data = ColumnDataSource(
        data=dict(wavelength=spec1.wavelength.value / 10_000, flux=spec1.flux.value)
    )

    spec_source_net = ColumnDataSource(
        data=dict(
            wavelength=spec_source.data["wavelength"], flux=cloud_mixture_model(ff)
        )
    )

    spec_sources = {True: spec_source_cloud, False: spec_source}

    fig = figure(
        title="Custom Cloudy Sonora Model Dashboard",
        plot_height=340,
        plot_width=600,
        tools="pan,wheel_zoom,box_zoom,tap,reset,save",
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )
    fig.title.offset = -10
    fig.yaxis.axis_label = "Flux"
    fig.xaxis.axis_label = "Wavelength (micron)"
    fig.y_range = Range1d(start=0, end=1.5)
    xmin, xmax = df_nir.wavelength.min() * 0.998, df_nir.wavelength.max() * 1.002
    fig.x_range = Range1d(start=xmin, end=xmax)

    fig.step(
        "wavelength",
        "flux",
        line_width=2,
        color="darkslateblue",
        source=spec_source_net,
        legend_label="Partly cloudy mixture model",
        nonselection_line_color="darkslateblue",
        nonselection_line_alpha=1.0,
    )
    fig.step(
        "wavelength",
        "flux",
        line_width=2,
        color="sandybrown",
        source=spec_source_data,
        legend_label="IGRINS data",
        nonselection_line_color="sandybrown",
        nonselection_line_alpha=1.0,
    )

    fig.legend.location = "bottom_right"
    # Slider to smooth the data
    smoothing_slider = Slider(
        start=0.1,
        end=40,
        value=0.1,
        step=0.1,
        title="Spectral resolution kernel",
        width=490,
    )

    vz_slider = Slider(
        start=-0.0005,
        end=0.0005,
        value=0.00,
        step=0.00001,
        title="Radial Velocity",
        width=490,
        format="0.000f",
    )

    teff_slider = Slider(
        start=min(teff_points),
        end=max(teff_points),
        value=1300,
        step=100,
        title="Teff",
        width=490,
    )
    teff_message = Div(text="Closest grid point: {}".format(teff), width=100, height=10)
    logg_slider = Slider(
        start=min(logg_points),
        end=max(logg_points),
        value=max(logg_points),
        step=0.25,
        title="logg",
        width=490,
    )

    ff_slider = Slider(
        start=0,
        end=1.0,
        value=0.35,
        step=0.01,
        title="Filling factor of clouds",
        width=490,
    )
    file_slider = Slider(
        start=0, end=3, value=0, step=1, title="Data File Index", width=490,
    )

    order_slider = Slider(
        start=0, end=25, value=13, step=1, title="Echelle Order Index", width=490,
    )

    r_button = Button(label=">", button_type="default", width=30)
    l_button = Button(label="<", button_type="default", width=30)

    def update_upon_file_change(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec1 = (
            IGRINSSpectrum(file=reduced_fns[new], order=order_slider.value)
            .remove_nans()
            .normalize()
            .trim_edges()
        )
        spec_source_data.data["wavelength"] = spec1.wavelength.value / 10_000
        spec_source_data.data["flux"] = spec1.flux.value

    def update_upon_order_change(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec1 = (
            IGRINSSpectrum(file=reduced_fns[file_slider.value], order=new)
            .remove_nans()
            .normalize()
            .trim_edges()
        )
        spec_source_data.data = dict(
            wavelength=spec1.wavelength.value / 10_000,
            flux=spec1.flux.value
        )

        fig.x_range.start = np.min(spec_source_data.data["wavelength"]) * 0.9995
        fig.x_range.end = np.max(spec_source_data.data["wavelength"]) * 1.0005

        # We need to trigger the model update in Teff selection just to get the new model.
        teff_slider.value +=1.0e-9 # Hack, but it works!

    def update_upon_smooth(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec_source.data["flux"] = gaussian_filter1d(
            spec_source.data["native_flux"], new
        )
        spec_source_cloud.data["flux"] = gaussian_filter1d(
            spec_source_cloud.data["native_flux"], new
        )
        composite = cloud_mixture_model(ff_slider.value)
        spec_source_net.data["flux"] = composite / np.median(composite)

    def update_upon_vz(attr, old, new):
        """Callback to take action when vz slider changes"""
        spec_source.data["wavelength"] = spec_source.data["native_wavelength"] - new
        spec_source_cloud.data["wavelength"] = (
            spec_source_cloud.data["native_wavelength"] - new
        )
        spec_source_net.data["wavelength"] = spec_source.data["wavelength"]

    def update_upon_filling_factor_selection(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        composite = cloud_mixture_model(new)
        spec_source_net.data["flux"] = composite / np.median(composite)
        # new_const = np.median(spec_source_net.data["flux"])
        # spec_source_net.data["flux"] = spec_source_net.data["flux"] / new_const
        # spec_source_cloud.data["flux"] /= new_const
        # spec_source.data["flux"] /= new_const
        # renormalize_and_sync_flux_levels()

    def update_upon_teff_selection(attr, old, new):
        """Callback to take action when teff slider changes"""
        teff = find_nearest(teff_points, new)

        teff_message.text = "Closest grid point: {}".format(teff)

        wl_low = np.nanmin(spec_source_data.data["wavelength"])
        wl_high = np.nanmax(spec_source_data.data["wavelength"])

        for cloudy, source in zip([True, False], [spec_source_cloud, spec_source]):
            basename = basename_constructor(np.float(teff), logg_slider.value, cloudy,)

            fn = models_path + basename
            if precache:
                df_nir = precached_grid[basename]
            else:
                df_nir = load_and_prep_spectrum(
                    fn, downsample=4, wl_low=wl_low, wl_high=wl_high
                )

            source.data = dict(
                wavelength=df_nir.wavelength.values - vz_slider.value,
                flux=gaussian_filter1d(df_nir.flux.values, smoothing_slider.value),
                native_flux=df_nir.flux.values,
                native_wavelength=df_nir.wavelength.values,
            )

        composite = cloud_mixture_model(ff_slider.value)
        spec_source_net.data = dict(
                wavelength=spec_source.data["wavelength"], flux=composite / np.median(composite)
            )

    def update_upon_logg_selection(attr, old, new):
        """Callback to take action when logg slider changes"""
        teff = find_nearest(teff_points, teff_slider.value)

        wl_low = np.nanmin(spec_source_data.data["wavelength"])
        wl_high = np.nanmax(spec_source_data.data["wavelength"])

        for cloudy in [True, False]:
            basename = basename_constructor(np.float(teff), new, cloudy)

            fn = models_path + basename
            if precache:
                df_nir = precached_grid[basename]
            else:
                df_nir = load_and_prep_spectrum(
                    fn, downsample=4, wl_low=wl_low, wl_high=wl_high
                )

            spec_sources[cloudy].data["native_wavelength"] = df_nir.wavelength.values
            spec_sources[cloudy].data["native_flux"] = df_nir.flux.values
            spec_sources[cloudy].data["wavelength"] = (
                df_nir.wavelength.values - vz_slider.value
            )
            spec_sources[cloudy].data["flux"] = gaussian_filter1d(
                df_nir.flux.values, smoothing_slider.value
            )
            composite = cloud_mixture_model(ff_slider.value)
            spec_source_net.data["flux"] = composite / np.median(composite)

    def go_right_by_one():
        """Step forward in time by a single cadence"""
        current_index = np.abs(teff_points - teff_slider.value).argmin()
        new_index = current_index + 1
        if new_index <= (len(teff_points) - 1):
            teff_slider.value = teff_points[new_index]

    def go_left_by_one():
        """Step back in time by a single cadence"""
        current_index = np.abs(teff_points - teff_slider.value).argmin()
        new_index = current_index - 1
        if new_index >= 0:
            teff_slider.value = teff_points[new_index]

    r_button.on_click(go_right_by_one)
    l_button.on_click(go_left_by_one)
    smoothing_slider.on_change("value", update_upon_smooth)
    vz_slider.on_change("value", update_upon_vz)
    teff_slider.on_change("value", update_upon_teff_selection)
    logg_slider.on_change("value", update_upon_logg_selection)
    ff_slider.on_change("value", update_upon_filling_factor_selection)
    file_slider.on_change("value", update_upon_file_change)
    order_slider.on_change("value", update_upon_order_change)

    sp1, sp2, sp3, sp4 = (
        Spacer(width=5),
        Spacer(width=10),
        Spacer(width=20),
        Spacer(width=100),
    )

    widgets_and_figures = layout(
        [fig],
        [l_button, sp1, r_button, sp2, teff_slider, sp3, teff_message],
        [sp4, logg_slider],
        [sp4, smoothing_slider],
        [sp4, vz_slider],
        [sp4, ff_slider],
        [sp4, file_slider],
        [sp4, order_slider],
    )
    doc.add_root(widgets_and_figures)
