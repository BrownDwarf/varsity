import numpy as np
import pandas as pd
import bokeh
import glob
import astropy.units as u
import tqdm

from bokeh.io import show, output_notebook, push_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import (
    Slider,
    Span,
    Range1d,
    Dropdown
)
from bokeh.layouts import layout, Spacer
from bokeh.models.widgets import Button, Div

from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict

## We can optionally cache the spectra
## This should be faster, but is only slightly faster
precache = False # Toggle for caching

## The published grid points of Sonora
teff_points = np.array([500, 525, 550, 575, 600, 650, 700, 750, 800,
            850, 900, 950,  1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
            1800,1900, 2000, 2100, 2200, 2300, 2400])
logg_points = np.arange(4.0, 5.51, 0.25)
logg_par_dict = {4.0:"100",4.25:"178",4.5:"316",4.75:"562",
                           5.0:"1000",5.25:"1780",5.5:"3160"}

## Caroline's custom cloudy models (only high-res in K-band)
fns_morley = sorted(glob.glob('/home/gully/GitHub/varsity/models/models_forGully/t*.spec'))

def load_and_prep_spectrum(fn, wl_low=2.1, wl_high=2.13, downsample=4, cloudy=False):
    df_native = pd.read_csv(fn,
                            skiprows=[0, 1],
                            delim_whitespace=True,
                            names=['wavelength', 'flux_raw']
                           ).sort_values('wavelength').reset_index(drop=True)

    ## Standardize flux density units to cgs
    if cloudy:
        morley_flux_w_units = (df_native.flux_raw.values*u.Watt/u.m**2/u.m)
        flux_cgs_units = morley_flux_w_units.to(u.erg/u.cm**2/u.s/u.Hz,
                                equivalencies=u.spectral_density(df_native.wavelength.values*u.micron))
    else:
        flux_cgs_units = df_native['flux_raw'].values

    df_native['flux'] = flux_cgs_units

    ## Trim to the wavelength bounds
    nir_mask = (df_native.wavelength > wl_low) & (df_native.wavelength < wl_high)

    ## Decimate the data:
    df_nir = df_native[['wavelength', 'flux']][nir_mask]\
                                .rolling(int(downsample*2.5), win_type='gaussian')\
                                .mean(std=downsample/2)\
                                .iloc[::downsample, :]\
                                .dropna()\
                                .reset_index(drop=True)

    return df_nir


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

teff_dict = OrderedDict()
for i, teff in enumerate(teff_points):
    teff_dict[i]=str(teff)


def create_interact_ui(doc):

    ## Make the spectrum sources
    # Initialize
    teff = 1300
    logg = 4.0
    base_name = "sp_t{0:0>.0f}g{1:}nc_m0.0".format(np.float(teff), logg_par_dict[logg])
    fn = '~/libraries/raw/marley/'+base_name

    # Cloud-free
    df_nir = load_and_prep_spectrum(fn, downsample=4)

    # Cloudy
    fn = fns_morley[38]
    df_cloud = load_and_prep_spectrum(fn, downsample=4, cloudy=True)

    # Fixed normalization constant for all spectra
    scalar_norm = np.percentile(df_nir.flux.values, 95)

    spec_source = ColumnDataSource(
        data=dict(
            wavelength=df_nir.wavelength.values,
            flux=gaussian_filter1d(df_nir.flux.values/scalar_norm, 0.1),
            native_flux = df_nir.flux.values / scalar_norm,
            native_wavelength = df_nir.wavelength.values
        )
    )

    spec_source_cloud = ColumnDataSource(
        data=dict(
            wavelength=df_cloud.wavelength.values,
            flux=gaussian_filter1d(df_cloud.flux.values/scalar_norm, 0.1),
            native_flux = df_cloud.flux.values / scalar_norm,
            native_wavelength = df_cloud.wavelength.values
        )
    )

    fig = figure(
        title="Sonora Bobcat in Bokeh",
        plot_height=340,
        plot_width=600,
        tools="pan,wheel_zoom,box_zoom,tap,reset",
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )
    fig.title.offset = -10
    fig.yaxis.axis_label = "Flux "
    fig.xaxis.axis_label = "Wavelength (micron)"
    fig.y_range = Range1d(start=0, end=1.5)
    xmin, xmax = df_nir.wavelength.min()*0.995, df_nir.wavelength.max()*1.005
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
        color="slategray",
        source=spec_source_cloud,
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
            width=490
        )

    vz_slider = Slider(
            start=-0.009,
            end=0.009,
            value=0.00,
            step=0.0005,
            title="Radial Velocity",
            width=490,
        format='0.000f'
        )


    teff_slider = Slider(
            start=min(teff_points),
            end=max(teff_points),
            value=1300,
            step=25,
            title="Teff",
            width=490
        )
    teff_message = Div(text="Closest grid point: {}".format(teff), width=100, height=10)
    logg_slider = Slider(
            start=min(logg_points),
            end=max(logg_points),
            value=4.0,
            step=0.25,
            title="logg",
            width=490
        )
    r_button = Button(label=">", button_type="default", width=30)
    l_button = Button(label="<", button_type="default", width=30)

    def update_upon_smooth(attr, old, new):
        """Callback to take action when smoothing slider changes"""
        #spec_source.data["wavelength"] = df_nir.wavelength.values[::new]
        spec_source.data["flux"] = gaussian_filter1d(spec_source.data["native_flux"], new)

    def update_upon_vz(attr, old, new):
        """Callback to take action when vz slider changes"""
        spec_source.data["wavelength"] = spec_source.data["native_wavelength"] - new
        #spec_source.data["flux"] = gaussian_filter1d(df_nir.flux.values, new)

    def update_upon_teff_selection(attr, old, new):
        """Callback to take action when teff slider changes"""
        teff = find_nearest(teff_points, new)
        if teff != old:
            teff_message.text = "Closest grid point: {}".format(teff)
            base_name = "sp_t{0:0>.0f}g{1:}nc_m0.0".format(np.float(teff), logg_par_dict[logg])

            fn = '~/libraries/raw/marley/'+base_name
            if precache:
                df_nir = precached_grid[base_name]
            else:
                df_nir = load_and_prep_spectrum(fn, downsample=4)
            scalar_norm = np.percentile(df_nir.flux.values, 95)
            spec_source.data["native_wavelength"] = df_nir.wavelength.values
            spec_source.data["native_flux"] = df_nir.flux.values / scalar_norm
            spec_source.data["wavelength"] = df_nir.wavelength.values - vz_slider.value
            spec_source.data["flux"] = gaussian_filter1d(df_nir.flux.values/ scalar_norm, smoothing_slider.value)

        else:
            pass

    def update_upon_logg_selection(attr, old, new):
        """Callback to take action when logg slider changes"""
        teff = find_nearest(teff_points, teff_slider.value)
        base_name = "sp_t{0:0>.0f}g{1:}nc_m0.0".format(np.float(teff), logg_par_dict[new])

        fn = '~/libraries/raw/marley/'+base_name
        if precache:
            df_nir = precached_grid[base_name]
        else:
            df_nir = load_and_prep_spectrum(fn, downsample=4)
        scalar_norm = np.percentile(df_nir.flux.values, 95)
        spec_source.data["native_wavelength"] = df_nir.wavelength.values
        spec_source.data["native_flux"] = df_nir.flux.values / scalar_norm
        spec_source.data["wavelength"] = df_nir.wavelength.values - vz_slider.value
        spec_source.data["flux"] = gaussian_filter1d(df_nir.flux.values/ scalar_norm, smoothing_slider.value)

    def go_right_by_one():
        """Step forward in time by a single cadence"""
        current_index = np.abs(teff_points - teff_slider.value).argmin()
        new_index = current_index + 1
        if new_index <= (len(teff_points)-1):
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
            [sp4, vz_slider]
        )
    doc.add_root(widgets_and_figures)
