"""
  NAME
      esmloader

  DESCRIPTION
      esmloader utilizes intake-esm catalog files to load BARRA2 and BARPA datasets in the NCI data collection. It has been revised with enhancements inspired by the loaddata module developed by Chun-Hsu Su.

  PREREQUISITE
      Users must join dk92, ob53 and py18 projects via,
        https://my.nci.org.au/mancini/project/dk92/join
        https://my.nci.org.au/mancini/project/ob53/join
        https://my.nci.org.au/mancini/project/py18/join

  AUTHOR
      Rui Yang,  rui.yang@anu.edu.au, National Computational Infrastructure
      Chun-Hsu Su, chunhsu.su@bom.gov.au, Bureau of Meteorology
"""

import intake
from datetime import datetime
import calendar
import xarray as xr
import pandas as pd
import iris
import numpy as np
import cftime


class EsmCat:
    def __init__(self, collection):
        self.collection=collection
        catalog_files={"BARPA":"/g/data/dk92/catalog/v2/esm/barpa-py18/catalog.json",
                       "BARRA2":"/g/data/dk92/catalog/v2/esm/barra2-ob53/catalog.json"}
        self.data_catalog = intake.open_esm_datastore(catalog_files[self.collection])
        
        self.keys=list(self.data_catalog.df.columns)
        self.keyvals={}
        for col in self.keys:
            self.keyvals[col]=self.data_catalog.df[col].unique()
            
    def get_values(self,key):
        return(list(self.data_catalog.df[key].unique()))
            
    def whatis(self,freq,var_name):
        query = dict(
            variable_id=[var_name],
            freq=[freq],
        )
        catalog_subset = self.data_catalog.search(**query)
        files=catalog_subset.df.path
        ds = xr.open_dataset(files[0])
        print(f"Short name: {var_name}")
        attrs_dict = ds[var_name].attrs
        for attr in attrs_dict:
            print(f"{attr}: {attrs_dict[attr]}")
        return

    def list_barra2_variables(self,source_id_in,freq_in):
        """
        Prints a listing of the variables available for BARRA2 model.

        Parameters:
            model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
           freq (str): Time frequency of the data, e.g. 1hr, day, mon
        """
            
        query = dict(
            source_id=[source_id_in],
            freq=[freq_in],
        )
        catalog_subset = self.data_catalog.search(**query)
        varlist=list(catalog_subset.df.variable_id.unique())
        print(", ".join(varlist))
        return(varlist)
    
    def get_barra2_files(self,source_id_in, freq_in, variable_in,
                         version='*',
                         tstart=19790101,
                         tend=20300101):  
        tstart_in=int(tstart)
        tend_in=int(tend)

        query = dict(
                source_id=[source_id_in],
                freq=[freq_in],
                variable_id=[variable_in],)
        
        if tstart_in != 19790101 or tend_in != 20300101:    
            full_time_range=self.keyvals["time_range"]
            time_range_list=self.get_time_range(tstart_in,tend_in,full_time_range)    
            query["time_range"]=time_range_list
        
        catalog_subset = self.data_catalog.search(**query)
        files=list(catalog_subset.df.path)
        files.sort()
        return(catalog_subset,files)    

    def get_time_range(self,tstart, tend,time_range):
        start_date = datetime.strptime(str(tstart)[:6], "%Y%m")
        end_date = datetime.strptime(str(tend)[:6], "%Y%m")
        out_time_range=[]    
        for curdate in time_range:
            if curdate == 'na':
                continue
            cur_time_range=curdate.split("-")
            for cur_time in cur_time_range:
                tmp=datetime.strptime(str(cur_time), "%Y%m")                                 
                if start_date <= tmp <= end_date:
#                   print("catch")
                   out_time_range.append(curdate)
        return(list(set(out_time_range)))


    def load_barra2_data(self,model, freq, variable,
                     version="*",
                     tstart=19790101, tend=20300101,
                    loc=None,
                    latrange=None, lonrange=None,
                    **read_kwargs):
        """
        Returns the BARRA2 data

        Parameters:
           model (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
           freq (str): Time frequency of the data, e.g. 1hr, day, mon
           variable (str): Variable name, e.g., tas, uas, pr
           version (str): Data release version if multiple available
           tstart (str): Start of the time range, in yyyymmddHH format
           tend (str): End of the time range, in yyyymmddHH format
           loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
           latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
           lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
           read_kwargs (dict): Arguments to pass to xarray.open_mfdataset
           
        Returns:
           data (xarray.Dataset): Extracted data

        Note: model, freq, variable as per labels in
            /g/data/ob53/BARRA2/output/reanalysis/[model]/BOM/ERA5/
            historical/*/BARRA-R2/v1/[freq]/[variable]/[version]
        """

        
        catalog_subset,files = self.get_barra2_files(model, freq, variable, version=version, tstart=tstart, tend=tend)
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

#        ds = catalog_subset.to_dataset_dict(cdf_kwargs=read_kwargs)                
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

        if freq == 'fx':
            out = ds
        else:
            tstart = self._str2datetime(tstart, start=True)
            tend = self._str2datetime(tend, start=False)
            out = ds.sel(time=slice(tstart, tend))

        return (out)
    
    def _str2datetime(self,t, start=True):
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
    
    def get_barpa_files(self,source_id_in, driving_source_id_in, driving_experiment_id_in, freq_in, variable_in,
                    version="*",
                    tstart=19000101, 
                    tend=21010101):
        tstart_in=int(tstart)
        tend_in=int(tend)
        query = dict(
            source_id=[source_id_in],
            driving_source_id=[driving_source_id_in],
            driving_experiment_id=[driving_experiment_id_in],
            freq=[freq_in],
            variable_id=[variable_in],
        )
        if tstart_in != 19900101 or tend_in != 21010101:
            full_time_range=self.keyvals["time_range"]
            time_range_list=self.get_time_range(tstart_in,tend_in,full_time_range)
            query["time_range"]=time_range_list

        catalog_subset = self.data_catalog.search(**query)
        files=list(catalog_subset.df.path)
        files.sort()
        return(catalog_subset,files)

    
    def load_barpa_data(self,rcm, gcm, scenario, freq, variable,
                    version="*",
                    tstart='19000101', tend='21010101',
                    loc=None,
                    latrange=None,
                    lonrange=None,
                    **read_kwargs):
        """
        Returns the BAPRA data.

        Parameters:
            rcm (str): Regional model, e.g. BARPA-R, BARPA-C
            gcm (str): Driving GCM name, e.g. ACCESS-CM2
            scenario (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
            freq (str): Time frequency of data, e.g., 1hr, day, mon
            variable (str): Variable name, e.g. tas, uas, pr
            version (str): Data release version if multiple available
            tstart (str): Start of the time period, in yyyymmddHH format
            tend (str): End of the time period, in yyyymmddHH format
            loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
            latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
            lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
            read_kwargs (dict): Arguments to pass to xarray.open_mfdataset
    
        Returns:
           data (xarray.Dataset): Extracted data

        Note: model, freq, variable as per labels in
            /g/data/py18/BARPA/output/CMIP6/DD/[domain]/BOM/[gcm]/[scenario]/
            [ens]/[rcm]/v1-r1/[freq]/[variable]/[version]
        """
        catalog_subset,files = self.get_barpa_files(rcm, gcm, scenario, freq, variable, version=version,tstart=tstart, tend=tend)
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
#        ds = catalog_subset.to_dataset_dict(cdf_kwargs=read_kwargs)                        

        if freq == 'fx':
            out = ds
        else:
            tstart = self._str2datetime(tstart, start=True)
            tend = self._str2datetime(tend, start=False)
        
        # To accommodate for non-gregorian calendars
            cal = self._get_calendar(files[0])
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
    
    def _get_calendar(self,file):
        """
        Returns the calendar name in the given netcdf file.

        Parameters:
            file (str): Path to the netcdf file

        Returns:
            calendar_name (str): Name of the calendar type in this file
        """
        cube = iris.load(file)
        return cube[0].coords('time')[0].units.calendar





