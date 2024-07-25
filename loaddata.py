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
import calendar
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
        rootdir = "/g/data/ob53/BARRA2/output/reanalysis/{domain_id}/BOM/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}"
        expts = glob("/g/data/ob53/BARRA2/output/reanalysis/*/BOM/*/*/*/*")
    else:
        rootdir = "/g/data/py18/BARPA/output/CMIP6/DD/{domain_id}/BOM/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{soruce_id}"
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

def make_barra2_dirpath(source_id_in, freq_in):
    """
    Returns path to the BARRA2 data directory.

    Parameters:
        source_id_in (str): Model name, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
        freq_in (str): Time frequency of the data, e.g. 1hr, day, mon

    Returns:
        str: Directory path
    """
    basepath = '/g/data/ob53/BARRA2/output'
    
    rootdir_templ = "{basepath}/reanalysis/{domain_id}/BOM/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}/v1/{freq}"
    model_dict = {'BARRA-R2': ("hres", "AUS-11"),
                  'BARRA-RE2': ("eda", "AUS-22"),
                  'BARRA-C2': ("hres", 'AUS-04')}

    driving_source_id = "ERA5"
    driving_experiment_id = "historical"
    driving_variant_label = model_dict[source_id_in][0]
    domain_id = model_dict[source_id_in][1]
    return rootdir_templ.format(basepath=basepath, 
                                domain_id=domain_id,
                                driving_source_id=driving_source_id,
                                driving_experiment_id=driving_experiment_id,
                                driving_variant_label=driving_variant_label, 
                                source_id=source_id_in, 
                                freq=freq_in)

    
def get_barra2_files(source_id_in, freq_in, variable_id_in,
                     version='v*',
                     tstart='197901',
                     tend='203001'):
    """
    Returns all matching BARRA-R2 files in the NCI data collection.

    Parameters:
       source_id_in (str): Model name, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
       freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
       variable_id_in (str): Variable name, e.g., tas, uas, pr
       version (str): Data release version if multiple available
       tstart (str): Start of the time range, in yyyymmddHH
       tend (str): End of the time range, in yyyymmddHH

    Returns:
       list of str: List of full paths to the files
    """
    rootdir = os.path.join(make_barra2_dirpath(source_id_in, freq_in), variable_id_in, version)

    files = []
    if freq_in == 'fx':
        # static data
        files = glob(os.path.join(rootdir, f'{variable}_**.nc'))
    else:
        tstart = _str2datetime(tstart, start=True)
        tend = _str2datetime(tend, start=False)
        
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

def list_barra2_variables(source_id_in, freq_in):
    """
    Prints a listing of the variables available for BARRA2 model.

    Parameters:
        source_id_in (str): Model name, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
        freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
    """
    rootdir = make_barra2_dirpath(source_id_in, freq_in)

    varlist = os.listdir(rootdir)
    print(", ".join(varlist))

    return varlist

def list_barra2_freqs(model):
    """
    Prints a listing of the time frequency available for BARRA2 model data.

    Parameters:
        model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
    """
    rootdir = make_barra2_dirpath(model, 'fx')

    freqlist = os.listdir(rootdir+'/..')
    print(", ".join(freqlist))

    return freqlist

def load_barra2_data(source_id_in, freq_in, variable_id_in,
                     version="v*",
                     tstart='197901', tend='203001',
                    loc=None,
                    latrange=None, lonrange=None,
                    **read_kwargs):
    """
    Returns the BARRA2 data

    Parameters:
       source_id_in (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
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
    files = get_barra2_files(source_id_in, freq_in, variable_id_in, 
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
    for key in read_kwargs_default:
        if not key in read_kwargs:
            read_kwargs[key] = read_kwargs_default[key]
            
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
        tstart = _str2datetime(tstart, start=True)
        tend = _str2datetime(tend, start=False)
        out = ds.sel(time=slice(tstart, tend))

    return out

def _str2datetime(t, start=True):
    """
    Convert the datetime string to datetime object.

    Parameters:
        datetime (str): Datetime in string, either y, ym, ymd, ymdH format
        start (boolean):  True to return the earliest time for that year, month, day or hour
             else return the latest time for that year, month, day or hour

    Returns:
        datetime (datetime.datetime): datetime object of the datetime matching t
    """
    assert len(t) in [4, 6, 8, 10], f"Undefined time range information: {t}"
    if len(t) == 4:
        # Assume yyyy
        y = int(t)
        if start:
            return datetime(y, 1, 1, 0, 0)
        else:
            return datetime(y, 12, 31, 23, 59, 59)

    elif len(t) == 6:
        # Assume yyyymm
        y = int(t[:4])
        m = int(t[4:])
        if start:
            return datetime(y, m, 1, 0, 0)
        else:
            return datetime(y, m, calendar.monthrange(y, m)[1], 23, 59, 59)
    elif len(t) == 8:
        # Assume yyyymmdd
        y = int(t[:4])
        m = int(t[4:6])
        d = int(t[6:])
        if start:
            return datetime(y, m, d, 0, 0)
        else:
            return datetime(y, m, d, 23, 59, 59)
    elif len(t) == 10:
        # Assume yyyymmddHH
        y = int(t[:4])
        m = int(t[4:6])
        d = int(t[6:8])
        H = int(t[8:])
        return datetime(y, m, d, H, 0)
    
    return

def _screen_files(files, tstart='196001', tend='210101'):
    """
    Filters the list of files based on prescribed time range.

    Parameters:
        files (list of str): A list of filenames, assumes that the time information in filename
            exists in *_<t0>-<t1>.nc
        tstart (datetime.datetime): Time range, earliest time
        tend (datetime.datetime): Time range, latest time

    Returns:
        files (list of str): A list of filenames that match the time range.
    """
    tstart = _str2datetime(tstart, start=True)
    tend = _str2datetime(tend, start=False)

    files_filt = []
    for file in files:
        bn = os.path.basename(file)
        timerange = os.path.splitext(bn)[0].split("_")[-1]
        t0 = _str2datetime(timerange.split("-")[0], start=True)
        t1 = _str2datetime(timerange.split("-")[1], start=False)

        if t1 < tstart:
            #print("{:} < {:}".format(t1, tstart))
            continue
        if t0 > tend:
            continue
        files_filt.append(file)

    files_filt.sort()

    return files_filt

def make_barpa_dirpath(source_id_in, driving_source_id_in, driving_experiment_id_in, freq_in):
    """
    Returns path to the BARPA data directory.

    Parameters:
        source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon

    Returns:
        str: Directory path
    """
    model_dict = {'BARPA-R': ('AUS-15'),
                  'BARPA-C': ('AUS-04')}
    gcm_ens = {'ACCESS-CM2': 'r4i1p1f1',
         'ACCESS-ESM1-5': 'r6i1p1f1',
         'ERA5': 'r1i1p1f1',
           'NorESM2-MM':'r1i1p1f1',
           'EC-Earth3':'r1i1p1f1',
           'CNRM-ESM2-1':'r1i1p1f2',
           'CESM2':'r11i1p1f1',
           'CMCC-ESM2':'r1i1p1f1',
           'MPI-ESM1-2-HR': 'r1i1p1f1'}

    basepath = '/g/data/py18/BARPA/output'
    rootdir_templ = "{basepath}/CMIP6/DD/{domain_id}/BOM/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}/v1-r1/{freq}"
    domain_id = model_dict[source_id_in]
    driving_variant_label = gcm_ens[driving_source_id_in]

    return rootdir_templ.format(basepath=basepath, domain_id=domain_id, 
                                driving_source_id=driving_source_id_in,
                                driving_experiment_id=driving_experiment_id_in, 
                                driving_variant_label=driving_variant_label, 
                                source_id=source_id_in, 
                                freq=freq_in)

def get_barpa_files(source_id_in, driving_source_id_in, 
                    driving_experiment_id_in, freq_in, 
                    variable_id_in,
                    version="v*",
                    tstart='196001', 
                    tend='210101'):
    """
    Returns all the matching BARPA files in the NCI data collection.

    Parameters:
        source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
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
    datadir = os.path.join( make_barpa_dirpath(source_id_in, driving_source_id_in, 
                                               driving_experiment_id_in, freq_in), 
                           variable_id_in, version)

    # Find all the files within the time range
    files = glob(os.path.join(datadir, f'{variable_id_in}_*.nc'))
    files.sort()

    if freq_in == 'fx':
        return files
    
    files = _screen_files(files, tstart=tstart, tend=tend)

    return files

def list_barpa_variables(source_id_in, 
                         driving_source_id_in, 
                         driving_experiment_id_in, 
                         freq_in):
    """
    Prints a listing of the variables available for BARPA model.

    Parameters:
        source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon
        
    Returns:
        list of str: List of variable_id values.
    """
    rootdir = make_barpa_dirpath(source_id_in, 
                                 driving_source_id_in, 
                                 driving_experiment_id_in, 
                                 freq_in)

    varlist = os.listdir(rootdir)
    print(", ".join(varlist))

    return varlist

def list_barpa_freqs(source_id_in, driving_source_id_in, driving_experiment_id_in):
    """
    Prints a listing of the time frequency available for BARPA model data.

    Parameters:
        source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        
    Returns:
        list of str: list of freq values.
    """
    rootdir = make_barpa_dirpath(source_id_in, driving_source_id_in, driving_experiment_id_in, 'fx')

    freqlist = os.listdir(rootdir+'/..')
    print(", ".join(freqlist))

    return freqlist

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

def load_barpa_data(source_id_in, 
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
        source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
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
    files = get_barpa_files(source_id_in, driving_source_id_in, 
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
    for key in read_kwargs_default:
        if not key in read_kwargs:
            read_kwargs[key] = read_kwargs_default[key]
    ds = xr.open_mfdataset(files, **read_kwargs)

    if freq_in == 'fx':
        out = ds
    else:
        tstart = _str2datetime(tstart, start=True)
        tend = _str2datetime(tend, start=False)
        
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
    if collection == 'BARRA2':
        files = get_barra2_files('BARRA-R2', freq_in, variable_id_in,
                                 tstart='20100101', tend='20100101')
    else:
        files = get_barpa_files('BARPA-R', 'ERA5', 'evaluation', freq_in, variable_id_in,
                                tstart='20100101', tend='20100101')

    ds = xr.open_dataset(files[0])
    print(f"Short name: {variable_id_in}")

    attrs_dict = ds[variable_id_in].attrs
    for attr in attrs_dict:
        print(f"{attr}: {attrs_dict[attr]}")

    return attrs_dict
