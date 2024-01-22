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
import xarray as xr
import pandas as pd
from datetime import datetime as dt
import calendar 
import iris
import numpy as np

def list_experiments(model):
    """
    Prints listing of experiments
    
    Parameters:
        model (str): Model, either BARRA2 or BARPA
    """
    
    if model == 'BARRA2':
        rootdir = "/g/data/ob53/BARRA2/output/reanalysis/{domain}/BOM/ERA5/historical/{era5_mem}/{model}"
        expts = glob("/g/data/ob53/BARRA2/output/reanalysis/*/BOM/ERA5/historical/*/*")
    else:
        rootdir = "/g/data/py18/BARPA/output/CMIP6/DD/{domain}/BOM/{gcm}/{scenario}/{ens}/{rcm}"
        expts = glob("/g/data/py18/BARPA/output/CMIP6/DD/*/BOM/*/*/*/*")

    toks = rootdir.split("/")
    columns = []
    indexes = []
    for i, tok in enumerate(toks):
        if tok.startswith("{") and tok.endswith("}"):
            columns.append(tok)
            indexes.append(i)
        
    print("   ".join(columns))
    for expt in expts:
        toks = expt.split("/")
        print("   ".join(np.array(toks)[indexes]))
        
    return
    
def make_barra2_dirpath(model, freq):
    """
    Returns path to the BARRA2 data directory.
    
    Parameters:
        model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
       freq (str): Time frequency of the data, e.g. 1hr, day, mon
       
    Returns:
       path (str): Directory path
    """
    basepath = '/g/data/ob53/BARRA2/output'
    rootdir_templ = "{basepath}/reanalysis/{domain}/BOM/ERA5/historical/{era5_mem}/{model}/v1/{freq}"
    model_dict = {'BARRA-R2': ("hres", "AUS-11"),
                  'BARRA-RE2': ("eda", "AUS-22"),
                  'BARRA-C2': ("hres", 'AUS-04')}
    
    era5_mem = model_dict[model][0]
    domain = model_dict[model][1]
    return rootdir_templ.format(basepath=basepath, domain=domain, era5_mem=era5_mem, model=model, freq=freq)
    
def get_barra2_files(model, freq, variable, 
                     version='*',
                     tstart=None, 
                     tend=None):
    """
    Returns all matching BARRA-R2 files in the NCI data collection.
    
    Parameters:
       model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
       freq (str): Time frequency of the data, e.g. 1hr, day, mon
       variable (str): Variable name, e.g., tas, uas, pr
       version (str): Data release version if multiple available
       tstart (datetime.datetime): Start of the time range
       tend (datetime.datetime): End of the time range
       
    Returns:
       files (list of str):List of full paths to the files 

    Note: model, freq, variable as per labels in
        /g/data/ob53/BARRA2/output/reanalysis/[domain]/BOM/ERA5/
        historical/*/[model]/v1/[freq]/[variable]/[version]
    """
    rootdir = os.path.join(make_barra2_dirpath(model, freq), variable, version)
    
    files = []
    if freq == 'fx':
        # static data
        files = glob(os.path.join(rootdir, '{:}_**.nc'.format(variable)))
    else:
        # non-static data
        # data organised as monthly files
        tspan = pd.date_range(dt(tstart.year, tstart.month, 1), dt(tend.year, tend.month, 1), freq='MS')
        
        if len(tspan)==0:
            tspan = pd.date_range(tstart, tend, freq='D')

        for time in tspan:
            files += glob(os.path.join(rootdir, '{:}_*_{:}-*.nc'.format(variable, time.strftime("%Y%m"))))
    
    files = list(set(files))
    files.sort()
    
    return files

def list_barra2_variables(model, freq):
    """
    Prints a listing of the variables available for BARRA2 model.
    
    Parameters:
        model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
       freq (str): Time frequency of the data, e.g. 1hr, day, mon
    """
    rootdir = make_barra2_dirpath(model, freq)
    
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
    
def load_barra2_data(model, freq, variable, 
                     version="*",
                    tstart=None, tend=None,
                    loc=None, 
                    latrange=None, lonrange=None):
    """
    Returns the BARRA2 data
    
    Parameters:
       model (str): Model, e.g., AUS-11, AUS-22
       freq (str): Time frequency of the data, e.g. 1hr, day, mon
       variable (str): Variable name, e.g., tas, uas, pr
       version (str): Data release version if multiple available
       tstart (datetime.datetime): Start of the time range
       tend (datetime.datetime): End of the time range
       loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
       latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
       lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range

    Returns:
       data (xarray.Dataset): Extracted data
       
    Note: model, freq, variable as per labels in
        /g/data/ob53/BARRA2/output/reanalysis/[model]/BOM/ERA5/
        historical/*/BARRA-R2/v1/[freq]/[variable]/[version]
    """
    files = get_barra2_files(model, freq, variable, version=version, tstart=tstart, tend=tend)
    assert len(files) > 0, "Cannot find data files"
    
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', parallel=True, coords='minimal', data_vars='minimal', compat='override')
    
    if loc is not None:
        lat0 = loc[0]
        lon0 = loc[1]
        ds = ds.sel(lat=lat0, method='nearest')
        ds = ds.sel(lon=lon0, method='nearest')
        
    if latrange is not None:
        ds = ds.sel(lat=slice(latrange[0], latrange[1]))
    if lonrange is not None:
        ds = ds.sel(lon=slice(lonrange[0], lonrange[1]))
    
    if freq == 'fx':
        out = ds
    else:
        out = ds.sel(time=slice(tstart, tend))

    return out

def _str2dt(t, start=True):
    """
    Convert the datetime string to datetime object.

    Parameters:
        datetime (str): Datetime in string, either y, ym, ymd, ymdH format
        start (boolean):  True to return the earliest time for that year, month, day or hour
             else return the latest time for that year, month, day or hour
    
    Returns:
        datetime (datetime.datetime): datetime object of the datetime matching t
    """
    assert len(t) in [4, 6, 8], "Undefine time range information: {:}".format(t)
    if len(t) == 4:
        # Assume yyyy
        y = int(t)
        if start:
            return dt(y, m, 1, 0, 0)
        else:
            return dt(y, m, 12, 23, 59, 59)
        
    elif len(t) == 6:
        # Assume yyyymm
        y = int(t[:4])
        m = int(t[4:])
        if start:
            return dt(y, m, 1, 0, 0)
        else:
            return dt(y, m, calendar.monthrange(y, m)[1], 23, 59, 59)
    elif len(t) == 8:
        # Assume yyyymmdd
        y = int(t[:4])
        m = int(t[4:6])
        d = int(t[6:])
        if start:
            return dt(y, m, d, 0, 0)
        else:
            return dt(y, m, d, 23, 59, 59)
    return

def _screen_files(files, tstart=None, tend=None):
    """
    Filters the list of files based on prescribed time range.

    Parameters:
        files (list of str): A list of filenames, assumes that the time information in filename
            exists in *_<t0>-<t1>.nc
        trange tuple of datetime.datetime: (tstart, tend) Time range, earliest time and latest time

    Returns:
        files (list of str): A list of filenames that match the time range.
    """
    if tstart is None:
        tstart = dt(1900, 1, 1)
    if tend is None:
        tend = dt(2200, 1, 1)
    
    files_filt = []
    for file in files:
        bn = os.path.basename(file)
        timerange = os.path.splitext(bn)[0].split("_")[-1]
        t0 = _str2dt(timerange.split("-")[0], start=True)
        t1 = _str2dt(timerange.split("-")[1], start=False)
    
        if t1 < tstart:
            #print("{:} < {:}".format(t1, tstart))
            continue
        if t0 > tend:
            continue
        files_filt.append(file)
    
    files_filt.sort()
    
    return files_filt

def make_barpa_dirpath(rcm, gcm, scenario, freq):
    """
    Returns path to the BARPA data directory.
    
    Parameters:
        rcm (str): Regional model, e.g. BARPA-R, BARPA-C
        gcm (str): Driving GCM name, e.g. ACCESS-CM2
        scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq (str): Time frequency of data, e.g., 1hr, day, mon
        
    Returns:
        path (str): Directory path
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
    rootdir_templ = "{basepath}/CMIP6/DD/{domain}/BOM/{gcm}/{scenario}/{ens}/{rcm}/v1-r1/{freq}"
    domain = model_dict[rcm]
    ens = gcm_ens[gcm]
    
    return rootdir_templ.format(basepath=basepath, domain=domain, gcm=gcm, scenario=scenario, ens=ens, rcm=rcm, freq=freq)
    
def get_barpa_files(rcm, gcm, scenario, freq, variable, 
                    version="*",
                    tstart=None, tend=None):
    """
    Returns all the matching BARPA files in the NCI data collection.

    Parameters:
        rcm (str): Regional model, e.g. BARPA-R, BARPA-C
        gcm (str): Driving GCM name, e.g. ACCESS-CM2
        scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq (str): Time frequency of data, e.g., 1hr, day, mon
        variable (str): Variable name, e.g. tas, uas, pr
        version (str): Data release version if multiple available
        tstart (datetime.datetime): Start of the time period
        tend (datetime.datetime): End of the time period
        
    Returns:
       files (list of str): List of full paths to the files 

    Note: model, freq, variable as per labels in
        /g/data/py18/BARPA/output/CMIP6/DD/[domain]/BOM/[gcm]/[scenario]/        
        [ens]/[rcm]/v1-r1/[freq]/[variable]/[version]
    """
    datadir = os.path.join( make_barpa_dirpath(rcm, gcm, scenario, freq), variable, version)

    # Find all the files within the time range
    files = glob(os.path.join(datadir, '%s_*.nc' % variable))
    files.sort()
    
    if freq == 'fx':
        return files
    
    files = _screen_files(files, tstart=tstart, tend=tend)
    
    return files

def list_barpa_variables(rcm, gcm, scenario, freq):
    """
    Prints a listing of the variables available for BARPA model.
    
    Parameters:
        rcm (str): Regional model, e.g. BARPA-R, BARPA-C
        gcm (str): Driving GCM name, e.g. ACCESS-CM2
        scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq (str): Time frequency of data, e.g., 1hr, day, mon
    """
    rootdir = make_barpa_dirpath(rcm, gcm, scenario, freq)
    
    varlist = os.listdir(rootdir)
    print(", ".join(varlist))
    
    return varlist

def list_barpa_freqs(rcm, gcm, scenario):
    """
    Prints a listing of the time frequency available for BARPA model data.
    
    Parameters:
        rcm (str): Regional model, e.g. BARPA-R, BARPA-C
        gcm (str): Driving GCM name, e.g. ACCESS-CM2
        scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
    """
    rootdir = make_barpa_dirpath(rcm, gcm, scenario, 'fx')
    
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

def load_barpa_data(rcm, gcm, scenario, freq, variable, 
                    version="*",
                    tstart=None, tend=None,
                    loc=None,
                    latrange=None,
                    lonrange=None):
    """
    Returns the BAPRA data.

    Parameters:
        rcm (str): Regional model, e.g. BARPA-R, BARPA-C
        gcm (str): Driving GCM name, e.g. ACCESS-CM2
        scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq (str): Time frequency of data, e.g., 1hr, day, mon
        variable (str): Variable name, e.g. tas, uas, pr
        version (str): Data release version if multiple available
        tstart (datetime.datetime): Start of the time period
        tend (datetime.datetime): End of the time period
        loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
        latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
        lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range

    Returns:
       data (xarray.Dataset): Extracted data

    Note: model, freq, variable as per labels in
        /g/data/py18/BARPA/output/CMIP6/DD/[domain]/BOM/[gcm]/[scenario]/        
        [ens]/[rcm]/v1-r1/[freq]/[variable]/[version]
    """
    files = get_barpa_files(rcm, gcm, scenario, freq, variable, version=version, tstart=tstart, tend=tend)
    assert len(files) > 0, "Cannot find data files"
    
    cal = _get_calendar(files[0])
    if 'gregorian' in cal:
        tstart = dt(1900, 1, 1) if tstart is None else tstart
        tend = dt(2200, 1, 1) if tend is None else tend
    elif '360' in cal:
        tstart = cftime.Datetime360Day(1900, 1, 1) if tstart is None else cftime.Datetime360Day(tstart.year, tstart.month, tstart.day, tstart.hour)
        tend = cftime.Datetime360Day(2200, 1, 1) if tend is None else cftime.Datetime360Day(tend.year, tend.month, tend.day, tend.hour)
    elif '365' in cal:
        tstart = cftime.DatetimeAllLeap(1900, 1, 1) if tstart is None else cftime.DatetimeNoLeap(tstart.year, tstart.month, tstart.day, tstart.hour)
        tend = cftime.DatetimeAllLeap(2200, 1, 1) if tend is None else cftime.DatetimeNoLeap(tend.year, tend.month, tend.day, tend.hour)
        
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', parallel=True, coords='minimal', data_vars='minimal', compat='override')
    
    if freq == 'fx':
        out = ds
    else:
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

def whatis(freq, variable, model='BARRA2'):
    """
    Prints the metadata for this variable
    
    Parameters:
        freq (str): Time frequency of the variable, e.g., 1hr, day
        variable (str): Variable name
        model (str): Which model, either BARRA2 or BARPA
    
    Returns
        attributes (dict): Dictionary containing the variable attributes
    """ 
    if model == 'BARRA2':
        files = get_barra2_files('BARRA-R2', freq, variable, tstart=dt(2010, 1, 1), tend=dt(2010, 1, 1))
    else:
        files = get_barpa_files('BARPA-R', 'ERA5', 'evaluation', freq, variable, tstart=dt(2010, 1, 1), tend=dt(2010, 1, 1))
        
    ds = xr.open_dataset(files[0])
    print("Short name: {:}".format(variable))
    
    attrs_dict = ds[variable].attrs
    for attr in attrs_dict:
        print("{:}: {:}".format(attr, attrs_dict[attr]))
        
    return attrs_dict