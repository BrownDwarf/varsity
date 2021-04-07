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
reduced_fns = glob.glob(
    "../../data/IGRINS/originals/GS-2021A-DD-104/*/reduced/SDCK*.spec_a0v.fits"
)
spec1 = IGRINSSpectrum(file=reduced_fns[1], order=13).remove_nans().normalize()

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

    assertion_msg = "We can only model K-band clouds currently."
    assert (wl_low > 2.0) & (wl_high < 2.51), assertion_msg

    ## Standardize flux density units to cgs
    morley_flux_w_units = df_native.flux_raw.values * u.Watt / u.m ** 2 / u.m
    flux_cgs_units = morley_flux_w_units.to(
        u.erg / u.cm ** 2 / u.s / u.Hz,
        equivalencies=u.spectral_density(df_native.wavelength.values * u.micron),
    )
    df_native["flux"] = flux_cgs_units

    ## Trim to the wavelength bounds
    nir_mask = (df_native.wavelength > wl_low) & (df_native.wavelength < wl_high)

    ## Decimate the data:
    df_nir = (
        df_native[["wavelength", "flux"]][nir_mask]
        .rolling(int(downsample * 2.5), win_type="gaussian")
        .mean(std=downsample / 2)
        .iloc[::downsample, :]
        .dropna()
        .reset_index(drop=True)
    )

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
    basename = basename_constructor(np.float(teff), logg, False)

    # Cloud-free
    df_nir = load_and_prep_spectrum(models_path + basename, downsample=4)

    # Cloudy
    basename = basename_constructor(np.float(teff), logg, True)
    df_cloud = load_and_prep_spectrum(models_path + basename, downsample=4)

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

    spec_source_data = ColumnDataSource(
        data=dict(wavelength=spec1.wavelength.value / 10_000, flux=spec1.flux.value)
    )

    spec_sources = {True: spec_source_cloud, False: spec_source}

    fig = figure(
        title="Sonora Bobcat in Bokeh",
        plot_height=340,
        plot_width=600,
        tools="pan,wheel_zoom,box_zoom,tap,reset",
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
        line_width=1,
        color="darkmagenta",
        source=spec_source,
        nonselection_line_color="darkmagenta",
        nonselection_line_alpha=1.0,
    )

    fig.step(
        "wavelength",
        "flux",
        line_width=1,
        color="sandybrown",
        source=spec_source_data,
        nonselection_line_color="sandybrown",
        nonselection_line_alpha=1.0,
    )

    fig.step(
        "wavelength",
        "flux",
        line_width=1,
        color="slategray",
        source=spec_sources[True],
        nonselection_line_color="slategray",
        nonselection_line_alpha=1.0,
    )

    # Slider to decimate the data
    smoothing_slider = Slider(
        start=0.1,
        end=40,
        value=0.1,
        step=0.1,
        title="Spectral resolution kernel",
        width=490,
    )

    vz_slider = Slider(
        start=-0.002,
        end=0.002,
        value=0.00,
        step=0.00005,
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
        value=4.0,
        step=0.25,
        title="logg",
        width=490,
    )
    r_button = Button(label=">", button_type="default", width=30)
    l_button = Button(label="<", button_type="default", width=30)

    def update_upon_smooth(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec_source.data["flux"] = gaussian_filter1d(
            spec_source.data["native_flux"], new
        )
        spec_source_cloud.data["flux"] = gaussian_filter1d(
            spec_source_cloud.data["native_flux"], new
        )

    def update_upon_vz(attr, old, new):
        """Callback to take action when vz slider changes"""
        spec_source.data["wavelength"] = spec_source.data["native_wavelength"] - new
        spec_source_cloud.data["wavelength"] = (
            spec_source_cloud.data["native_wavelength"] - new
        )
        # spec_source.data["flux"] = gaussian_filter1d(df_nir.flux.values, new)

    def update_upon_teff_selection(attr, old, new):
        """Callback to take action when teff slider changes"""
        teff = find_nearest(teff_points, new)
        if teff != old:
            teff_message.text = "Closest grid point: {}".format(teff)

            for cloudy in [True, False]:
                basename = basename_constructor(
                    np.float(teff), logg_slider.value, cloudy,
                )

                fn = models_path + basename
                if precache:
                    df_nir = precached_grid[basename]
                else:
                    df_nir = load_and_prep_spectrum(fn, downsample=4)

                scalar_norm = np.percentile(df_nir.flux.values, 50)
                spec_sources[cloudy].data[
                    "native_wavelength"
                ] = df_nir.wavelength.values
                spec_sources[cloudy].data["native_flux"] = (
                    df_nir.flux.values / scalar_norm
                )
                spec_sources[cloudy].data["wavelength"] = (
                    df_nir.wavelength.values - vz_slider.value
                )
                spec_sources[cloudy].data["flux"] = gaussian_filter1d(
                    df_nir.flux.values / scalar_norm, smoothing_slider.value
                )

        else:
            pass

    def update_upon_logg_selection(attr, old, new):
        """Callback to take action when logg slider changes"""
        teff = find_nearest(teff_points, teff_slider.value)

        for cloudy in [True, False]:
            basename = basename_constructor(np.float(teff), new, cloudy)

            fn = models_path + basename
            if precache:
                df_nir = precached_grid[basename]
            else:
                df_nir = load_and_prep_spectrum(fn, downsample=4)

            scalar_norm = np.percentile(df_nir.flux.values, 50)
            spec_sources[cloudy].data["native_wavelength"] = df_nir.wavelength.values
            spec_sources[cloudy].data["native_flux"] = df_nir.flux.values / scalar_norm
            spec_sources[cloudy].data["wavelength"] = (
                df_nir.wavelength.values - vz_slider.value
            )
            spec_sources[cloudy].data["flux"] = gaussian_filter1d(
                df_nir.flux.values / scalar_norm, smoothing_slider.value
            )

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
    )
    doc.add_root(widgets_and_figures)
