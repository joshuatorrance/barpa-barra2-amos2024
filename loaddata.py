"""
  NAME
      loaddata

  DESCRIPTION
      loaddata is a Python module for interfacing with the BARRA2 and BARPA
      data sets in the NCI data collection.

  PREREQUISITE
      Users must join ob53 and py18 projects via,
        https://my.nci.org.au/mancini/project/ob53/join
        https://my.nci.org.au/mancini/project/py18/join

  AUTHOR
      Chun-Hsu Su, chunhsu.su@bom.gov.au, Bureau of Meteorology
"""
import os
from glob import glob
from datetime import datetime
from datapaths import make_barpa_dirpath, make_barra2_dirpath, screen_files, str2datetime
import xarray as xr
import pandas as pd
import iris
import numpy as np
import cftime


def list_experiments(collection):
    """
    Prints listing of experiments

    Parameters:
        collection (str): Collection name, either BARRA2 or BARPA
    """

    if collection == 'BARRA2':
        rootdir = "/g/data/ob53/BARRA2/output/reanalysis/{domain_id}/BOM/" \
            "{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}"
        expts = glob("/g/data/ob53/BARRA2/output/reanalysis/*/BOM/*/*/*/*")
    else:
        rootdir = "/g/data/py18/BARPA/output/CMIP6/DD/{domain_id}/BOM/" \
            "{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{soruce_id}"
        expts = glob("/g/data/py18/BARPA/output/CMIP6/DD/*/BOM/*/*/*/*")

    toks = rootdir.split("/")
    columns = []
    indexes = []
    for i, tok in enumerate(toks):
        if tok.startswith("{") and tok.endswith("}"):
            columns.append(tok)
            indexes.append(i)

    columns_formatted = ['{0: <23}'.format(s) for s in columns]
    print(" ".join(columns_formatted))
    for expt in expts:
        toks = expt.split("/")
        toks_formatted = ['{0: <23}'.format(s) for s in np.array(toks)[indexes]]
        print(" ".join(toks_formatted))


def get_barra2_files(id_in, freq_in, variable_id_in,
                     version='v*',
                     tstart='197901',
                     tend='203001'):
    """
    Returns all matching BARRA-R2 files in the NCI data collection.

    Parameters:
       id_in (str): Model name or the domain id, e.g., BARRA-R2 (either AUS-11 or AUST-11),
                            BARRA-RE2 (AUS-22 or AUST-22), BARRA-C2 (AUST-04), or
                            AUS-11, AUST-11, AUS-22, AUST-22, and AUST-04
       freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
       variable_id_in (str): Variable name, e.g., tas, uas, pr
       version (str): Data release version if multiple available
       tstart (str): Start of the time range, in yyyymmddHH
       tend (str): End of the time range, in yyyymmddHH
       use_thredds (boolean): True to read the file listing from THREDDS server
                   instead of gdata lustre filesystem

    Returns:
       list of str: List of full paths to the files
    """
    rootdir = os.path.join(make_barra2_dirpath(id_in, freq_in), variable_id_in, version)

    files = []
    if freq_in == 'fx':
        # static data
        files = glob(os.path.join(rootdir, f'{variable_id_in}_**.nc'))
    else:
        tstart = str2datetime(tstart, start=True)
        tend = str2datetime(tend, start=False)

        # non-static data
        # data organised as monthly files
        tspan = pd.date_range(datetime(tstart.year, tstart.month, 1),
                              datetime(tend.year, tend.month, 1), freq='MS')

        if len(tspan)==0:
            tspan = pd.date_range(tstart, tend, freq='D')

        for time in tspan:
            files += glob(os.path.join(rootdir, f'{variable_id_in}_*_{time.strftime("%Y%m")}-*.nc'))

    files = list(set(files))
    files.sort()

    return files


def load_barra2_data(id_in, freq_in, variable_id_in,
                     version="v*",
                     tstart='197901', tend='203001',
                    loc=None,
                    latrange=None, lonrange=None,
                    **read_kwargs):
    """
    Returns the BARRA2 data

    Parameters:
       id_in (str): Model name or the domain id, e.g., BARRA-R2 (either AUS-11 or AUST-11),
                            BARRA-RE2 (AUS-22 or AUST-22), BARRA-C2 (AUST-04), or
                            AUS-11, AUST-11, AUS-22, AUST-22, and AUST-04
       freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
       variable_id_in (str): Variable name, e.g., tas, uas, pr
       version (str): Data release version if multiple available
       tstart (str): Start of the time range, in yyyymmddHH format
       tend (str): End of the time range, in yyyymmddHH format
       loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
       latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
       lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
       read_kwargs (dict): Arguments to pass to xarray.open_mfdataset

    Returns:
        xarray.Dataset: Extracted data
    """
    files = get_barra2_files(id_in, freq_in, variable_id_in,
                             version=version, tstart=tstart, tend=tend)
    assert len(files) > 0, "Cannot find data files"

    # Define some default keys to pass to mf_dataset
    # If key appears in read_kwargs it will override the default
    read_kwargs_default = {
        "combine": "nested",
        "concat_dim": "time",
        "parallel": True,
        "coords": "minimal",
        "data_vars": "minimal",
        "compat": "override",
    }
    for key, val in read_kwargs_default.items():
        if not key in read_kwargs:
            read_kwargs[key] = val

    ds = xr.open_mfdataset(files, **read_kwargs)

    if loc is not None:
        lat0 = loc[0]
        lon0 = loc[1]
        ds = ds.sel(lat=lat0, method='nearest')
        ds = ds.sel(lon=lon0, method='nearest')

    if latrange is not None:
        ds = ds.sel(lat=slice(latrange[0], latrange[1]))
    if lonrange is not None:
        ds = ds.sel(lon=slice(lonrange[0], lonrange[1]))

    if freq_in == 'fx':
        out = ds
    else:
        tstart = str2datetime(tstart, start=True)
        tend = str2datetime(tend, start=False)
        out = ds.sel(time=slice(tstart, tend))

    return out


def get_barpa_files(id_in, driving_source_id_in,
                    driving_experiment_id_in, freq_in,
                    variable_id_in,
                    version="v*",
                    tstart='196001',
                    tend='210101'):
    """
    Returns all the matching BARPA files in the NCI data collection.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARPA-R (AUS-15) or BARPA-C (AUST-04), or
                            AUS-15, AUST-15, AUS-20i, AUST-04
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon
        variable_id_in (str): Variable name, e.g. tas, uas, pr
        version (str): Data release version if multiple available
        tstart (str): Start of the time period, in yyyymmddHH format
        tend (str): End of the time period, in yyyymmddHH format

    Returns:
       list of str: List of full paths to the files
    """
    datadir = os.path.join( make_barpa_dirpath(id_in, driving_source_id_in,
                                               driving_experiment_id_in, freq_in),
                           variable_id_in, version)

    # Find all the files within the time range
    files = glob(os.path.join(datadir, f'{variable_id_in}_*.nc'))
    files.sort()

    if freq_in == 'fx':
        return files

    files = screen_files(files, tstart=tstart, tend=tend)

    return files


def _get_calendar(file):
    """
    Returns the calendar name in the given netcdf file.

    Parameters:
        file (str): Path to the netcdf file

    Returns:
        calendar_name (str): Name of the calendar type in this file
    """
    cube = iris.load(file)
    return cube[0].coords('time')[0].units.calendar


def load_barpa_data(id_in,
                    driving_source_id_in,
                    driving_experiment_id_in,
                    freq_in,
                    variable_id_in,
                    version="v*",
                    tstart='190001', tend='210101',
                    loc=None,
                    latrange=None,
                    lonrange=None,
                    **read_kwargs):
    """
    Returns the BAPRA data.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARPA-R (AUS-15) or BARPA-C (AUST-04), or
                            AUS-15, AUST-15, AUS-20i, AUST-04
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon
        variable_id_in (str): Variable name, e.g. tas, uas, pr
        version (str): Data release version if multiple available
        tstart (str): Start of the time period, in yyyymmddHH format
        tend (str): End of the time period, in yyyymmddHH format
        loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
        latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
        lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
        read_kwargs (dict): Arguments to pass to xarray.open_mfdataset

    Returns:
       xarray.Dataset: Extracted data
    """
    files = get_barpa_files(id_in, driving_source_id_in,
                            driving_experiment_id_in,
                            freq_in, variable_id_in,
                            version=version,
                            tstart=tstart, tend=tend)
    assert len(files) > 0, "Cannot find data files"

    # Define some default keys to pass to mf_dataset
    # If key appears in read_kwargs it will override the default
    read_kwargs_default = {
        "combine": "nested",
        "concat_dim": "time",
        "parallel": True,
        "coords": "minimal",
        "data_vars": "minimal",
        "compat": "override",
    }
    for key, val in read_kwargs_default.items():
        if not key in read_kwargs:
            read_kwargs[key] = val

    ds = xr.open_mfdataset(files, **read_kwargs)

    if freq_in == 'fx':
        out = ds
    else:
        tstart = str2datetime(tstart, start=True)
        tend = str2datetime(tend, start=False)

        # To accommodate for non-gregorian calendars
        cal = _get_calendar(files[0])
        if '360' in cal:
            tstart = cftime.Datetime360Day(tstart.year, tstart.month, tstart.day, tstart.hour)
            tend = cftime.Datetime360Day(tend.year, tend.month, tend.day, tend.hour)
        elif '365' in cal:
            tstart = cftime.DatetimeNoLeap(tstart.year, tstart.month, tstart.day, tstart.hour)
            tend = cftime.DatetimeNoLeap(tend.year, tend.month, tend.day, tend.hour)

        out = ds.sel(time=slice(tstart, tend))

    if loc is not None:
        lat0 = loc[0]
        lon0 = loc[1]
        out = out.sel(lat=lat0, method='nearest')
        out = out.sel(lon=lon0, method='nearest')

    if latrange is not None:
        out = out.sel(lat=slice(latrange[0], latrange[1]))
    if lonrange is not None:
        out = out.sel(lon=slice(lonrange[0], lonrange[1]))

    return out


def whatis(freq_in, variable_id_in, collection='BARRA2'):
    """
    Prints the metadata for this variable

    Parameters:
        freq_in (str): Time frequency of the variable, e.g., 1hr, day
        variable_id_in (str): Variable name
        collection (str): Collection name, either BARRA2 or BARPA

    Returns
        dict: Dictionary containing the variable attributes
    """
    files = []
    if collection == 'BARRA2':
        for id_in in ["BARRA-R2", "BARRA-RE2", "BARRA-C2", "AUST-11", "AUST-22"]:
            files += get_barra2_files(id_in, freq_in, variable_id_in,
                                 tstart='20100101', tend='20100101')
            if len(files) > 0:
                break
    else:
        for id_in in ["BARPA-R", "BARPA-C", "AUST-15"]:
            files += get_barpa_files(id_in, 'ERA5', 'evaluation', freq_in, variable_id_in,
                                tstart='20100101', tend='20100101')
            if len(files) > 0:
                break

    if len(files) == 0:
        print(f"Information not found for {collection}/{freq_in}/{variable_id_in}")
        return

    ds = xr.open_dataset(files[0])
    print(f"Short name: {variable_id_in}")

    attrs_dict = ds[variable_id_in].attrs
    for attr in attrs_dict:
        print(f"{attr}: {attrs_dict[attr]}")

    return attrs_dict
