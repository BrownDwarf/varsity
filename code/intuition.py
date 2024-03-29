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
band = "H"  # "H" or "K"
end_order_dict = {"H": 27, "K": 25}
data_path = "../../data/IGRINS/originals/GS-2021A-DD-104/*/reduced/SDC{}*.spec_a0v.fits".format(
    band
)
reduced_fns = glob.glob(data_path)
reduced_fns = sorted(reduced_fns)
reduced_fns = reduced_fns[slice(0, 8, 2)]
# reduced_fns = reduced_fns[slice(1, 8, 2)]

spec1 = (
    IGRINSSpectrum(file=reduced_fns[3], order=13).remove_nans().normalize().trim_edges()
)

## The subset with overlapping points
teff_points = np.array([1200, 1300, 1400, 1500, 1600])
logg_points = np.arange(3.75, 4.51, 0.25)
logg_par_dict = {4.5: "316", 5.0: "1000"}
teff_dict = OrderedDict()
for i, teff in enumerate(teff_points):
    teff_dict[i] = str(teff)

band_suffix_dict = {
    "K": "_m0.0_co1.0_4100_5000_0.005.spec",
    "H": "_m0.0_co1.0_5000_7000_0.005.spec",
    "low": "_m0.0_co1.0.spec",
}
models_path = os.path.expandvars("$HOME/libraries/raw/morley_clouds_20210805/")
fns_morley = sorted(glob.glob(models_path + "/t*.spec"))


def calcBrightnessTemp_model_CVM(
    wl, flux, cgs=False
):  # wl in microns, flux in Watts/m^2/micron
    if cgs == False:  # we're in mks units, and need to convert back in ergs/s/cm^2/Hz
        for i in range(len(wl)):  # converts back from W/m2/micron to ergs/s/cm^2/Hz
            flux[i] = flux[i] * wl[i] ** 2 * 1000 / 2.99792458e14

    wn = np.empty(len(wl))
    T_B = np.empty(len(wl))
    for i in range(len(wl)):
        wn[i] = 1 / (wl[i] * 1e-4)  # gives wavenumber in 1/cm
        T_B[i] = (
            1.4388
            * wn[i]
            / np.log(1.191e-5 * wn[i] ** 3 * np.pi / (2.99792458e10 * flux[i]) + 1)
        )  # Equation from EGP code, prtout.mod.f lines

    return T_B


def calcBrightnessTemp_model_MGS(
    wl, flux, cgs=False
):  # wl in microns, flux in Watts/m^2/micron
    if cgs == False:  # we're in mks units, and need to convert back in ergs/s/cm^2/Hz
        flux = flux * wl ** 2 * 1000 / 2.99792458e14

    wn = 1 / (wl * 1e-4)  # gives wavenumber in 1/cm
    T_B = (
        1.4388 * wn / np.log(1.191e-5 * wn ** 3 * np.pi / (2.99792458e10 * flux) + 1)
    )  # Equation from EGP code, prtout.mod.f lines

    return T_B


def basename_constructor(teff, logg, fsed, band):
    """Construct the model basename base on Teff, logg, fsed and band.
    
    Note: band must be specified as 'H', 'K', or 'low'
    """
    basename = (
        "t{0:0>.0f}g{1:}f{2:01d}".format(np.float(teff), logg_par_dict[logg], int(fsed))
        + band_suffix_dict[band]
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


def load_and_prep_spectrum(fn, wl_low, wl_high, downsample=4):
    df_native = (
        pd.read_csv(
            fn,
            skiprows=[0, 1, 2],
            delim_whitespace=True,
            names=["wavelength", "flux_raw"],
        )
        .sort_values("wavelength")
        .reset_index(drop=True)
    )

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
    logg = 4.5
    fsed = 2.0
    ff = 0.3
    basename = basename_constructor(np.float(teff), logg, int(fsed), band)

    # Cloud-free
    wl_low = np.nanmin(spec1.wavelength.value / 10_000) * 0.995
    wl_high = np.nanmax(spec1.wavelength.value / 10_000) * 1.005

    df_nir = load_and_prep_spectrum(
        models_path + basename, downsample=4, wl_low=wl_low, wl_high=wl_high
    )

    # Fixed normalization constant for all spectra
    scalar_norm = np.percentile(df_nir.flux.values, 50)

    spec_source_net = ColumnDataSource(
        data=dict(
            wavelength=df_nir.wavelength.values,
            flux=gaussian_filter1d(df_nir.flux.values / scalar_norm, 0.1),
            native_flux=df_nir.flux.values / scalar_norm,
            native_wavelength=df_nir.wavelength.values,
        )
    )

    spec_source_data = ColumnDataSource(
        data=dict(wavelength=spec1.wavelength.value / 10_000, flux=spec1.flux.value)
    )

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
        legend_label="Cloudy model",
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
    fig.legend.orientation = "horizontal"
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
        start=4.5, end=5.0, value=4.5, step=0.5, title="logg", width=490,
    )

    fsed_slider = Slider(start=1, end=3, value=2, step=1, title="fsed", width=490,)
    file_slider = Slider(
        start=0, end=3, value=0, step=1, title="Data File Index", width=490,
    )

    # TODO: change the end value to 28 if H band
    order_slider = Slider(
        start=0,
        end=end_order_dict[band],
        value=13,
        step=1,
        title="Echelle Order Index",
        width=490,
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
            wavelength=spec1.wavelength.value / 10_000, flux=spec1.flux.value
        )

        fig.x_range.start = np.min(spec_source_data.data["wavelength"]) * 0.9995
        fig.x_range.end = np.max(spec_source_data.data["wavelength"]) * 1.0005

        # We need to trigger the model update in Teff selection just to get the new model.
        teff_slider.value += 1.0e-9  # Hack, but it works!

    def update_upon_smooth(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        # spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec_source_net.data["flux"] = gaussian_filter1d(
            spec_source_net.data["native_flux"], new
        )
        spec_source_net.data["flux"] = spec_source_net.data["flux"] / np.median(
            spec_source_net.data["flux"]
        )

    def update_upon_vz(attr, old, new):
        """Callback to take action when vz slider changes"""
        spec_source_net.data["wavelength"] = (
            spec_source_net.data["native_wavelength"] - new
        )

    def update_upon_fsed_selection(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        teff = find_nearest(teff_points, teff_slider.value)

        wl_low = np.nanmin(spec_source_data.data["wavelength"])
        wl_high = np.nanmax(spec_source_data.data["wavelength"])

        basename = basename_constructor(np.float(teff), logg_slider.value, new, band)

        fn = models_path + basename

        df_nir = load_and_prep_spectrum(
            fn, wl_low=wl_low, wl_high=wl_high, downsample=4,
        )

        spec_source_net.data = dict(
            wavelength=df_nir.wavelength.values - vz_slider.value,
            flux=gaussian_filter1d(df_nir.flux.values, smoothing_slider.value),
            native_flux=df_nir.flux.values,
            native_wavelength=df_nir.wavelength.values,
        )

        spec_source_net.data["flux"] = spec_source_net.data["flux"] / np.median(
            spec_source_net.data["flux"]
        )

    def update_upon_teff_selection(attr, old, new):
        """Callback to take action when teff slider changes"""
        teff = find_nearest(teff_points, new)

        teff_message.text = "Closest grid point: {}".format(teff)

        wl_low = np.nanmin(spec_source_data.data["wavelength"])
        wl_high = np.nanmax(spec_source_data.data["wavelength"])

        basename = basename_constructor(
            np.float(teff), logg_slider.value, int(fsed_slider.value), band
        )

        fn = models_path + basename

        df_nir = load_and_prep_spectrum(
            fn, wl_low=wl_low, wl_high=wl_high, downsample=4,
        )

        spec_source_net.data = dict(
            wavelength=df_nir.wavelength.values - vz_slider.value,
            flux=gaussian_filter1d(df_nir.flux.values, smoothing_slider.value),
            native_flux=df_nir.flux.values,
            native_wavelength=df_nir.wavelength.values,
        )

        spec_source_net.data["flux"] = spec_source_net.data["flux"] / np.median(
            spec_source_net.data["flux"]
        )

    def update_upon_logg_selection(attr, old, new):
        """Callback to take action when logg slider changes"""
        teff = find_nearest(teff_points, teff_slider.value)

        wl_low = np.nanmin(spec_source_data.data["wavelength"])
        wl_high = np.nanmax(spec_source_data.data["wavelength"])

        basename = basename_constructor(
            np.float(teff), new, int(fsed_slider.value), band
        )
        fn = models_path + basename

        df_nir = load_and_prep_spectrum(
            fn, wl_low=wl_low, wl_high=wl_high, downsample=4,
        )

        spec_source_net.data = dict(
            wavelength=df_nir.wavelength.values - vz_slider.value,
            flux=gaussian_filter1d(df_nir.flux.values, smoothing_slider.value),
            native_flux=df_nir.flux.values,
            native_wavelength=df_nir.wavelength.values,
        )

        spec_source_net.data["flux"] = spec_source_net.data["flux"] / np.median(
            spec_source_net.data["flux"]
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
    fsed_slider.on_change("value", update_upon_fsed_selection)
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
        [sp4, fsed_slider],
        [sp4, file_slider],
        [sp4, order_slider],
    )
    doc.add_root(widgets_and_figures)
