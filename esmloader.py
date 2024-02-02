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
        """
        Initialise the object.
        
        Parameters:
            collection (str): Either BARRA2 or BARPA
        
        Returns:
            EsmCat object
        """
        
        self.collection=collection
        
        catalog_files={"BARPA":"/g/data/dk92/catalog/v2/esm/barpa-py18/catalog.json",
                       "BARRA2":"/g/data/dk92/catalog/v2/esm/barra2-ob53/catalog.json"}
        
        self.data_catalog = intake.open_esm_datastore(catalog_files[self.collection])
        
        self.keys=list(self.data_catalog.df.columns)
        self.keyvals={}
        
        for col in self.keys:
            self.keyvals[col]=self.data_catalog.df[col].unique()
            
    def get_values(self, key):
        """
        Returns the list of possible values for a given key.
        
        Parameters:
            key (str): Key such as source_id, domain_id, freq, variable_id, etc
            
        Returns:
            list of str: All possible values for a given key.
        """
        return(list(self.data_catalog.df[key].unique()))
            
    def whatis(self, freq, variable_id):
        """
        Prints the metadata for this variable_id

        Parameters:
            freq (str): Time frequency of the variable, e.g., 1hr, day
            variable_id (str): Variable name

        Returns
            dict: Dictionary containing the variable attributes
        """
        query = dict(
            variable_id=[variable_id],
            freq=[freq],
        )
        
        catalog_subset = self.data_catalog.search(**query)
        files=catalog_subset.df.path
        ds = xr.open_dataset(files[0])
        
        print(f"Short name: {variable_id}")
        attrs_dict = ds[variable_id].attrs
        
        for attr in attrs_dict:
            print(f"{attr}: {attrs_dict[attr]}")
            
        return(attr)

    def list_barra2_variables(self, source_id_in, freq_in):
        """
        Prints a listing of the variables available for BARRA2 model.

        Parameters:
            source_id_in (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
            freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
            
        Returns:
            list of str: List of variable_id
        """
            
        query = dict(
            source_id=[source_id_in],
            freq=[freq_in],
        )
        
        catalog_subset = self.data_catalog.search(**query)
        varlist = list(catalog_subset.df.variable_id.unique())
        
        print(", ".join(varlist))
        
        return(varlist)
    
    def get_barra2_files(self, source_id_in, freq_in, variable_id_in,
                         version='*',
                         tstart=19790101,
                         tend=20300101):  
        """
        Returns all matching BARRA-R2 files in the NCI data collection.

        Parameters:
           source_id_in (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
           freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
           variable_id_in (str): Variable name, e.g., tas, uas, pr
           version (str): Data release version if multiple available
           tstart (integer): Start of the time range, in yyymmdd
           tend (integer): End of the time range, in yyyymmdd

        Returns:
           list of str: List of full paths to the files
        """
        query = dict(
                source_id=[source_id_in],
                freq=[freq_in],
                variable_id=[variable_id_in],)
        
        if tstart != 19790101 or tend != 20300101:    
            full_time_range = self.keyvals["time_range"]
            time_range_list = self.get_time_range(tstart, tend, full_time_range)
            query["time_range"] = time_range_list
        
        catalog_subset = self.data_catalog.search(**query)
        
        files = list(catalog_subset.df.path)
        files.sort()
        
        return(catalog_subset,files)    

    def get_time_range(self, tstart, tend, time_range_list):
        """
        Filter the given list of time_range based on tstart and tend.
        
        Parameters:
            tstart (integer): Start of the time range in yyyymmdd or yyyymm
            tend (integer): End of the time range in yyyymmdd or yyyymm
            time_range_list (list): List of time range, each time range as yyyymm-yyyymm
            
        Returns:
            list of str: List of time range filtered from time_range_list
        """
        
        start_date = datetime.strptime(str(tstart)[:6], "%Y%m")
        end_date = datetime.strptime(str(tend)[:6], "%Y%m")
        out_time_range_list = []
        
        for curdate in time_range_list:
            if curdate == 'na':
                continue
            cur_time_range = curdate.split("-")
            
            for cur_time in cur_time_range:
                tmp = datetime.strptime(str(cur_time), "%Y%m")
                
                if start_date <= tmp <= end_date:
                    out_time_range_list.append(curdate)
    
        return(list(set(out_time_range_list)))

    def load_barra2_data(self, source_id_in, freq_in, variable_id,
                     version="*",
                     tstart=19790101, tend=20300101,
                    loc=None,
                    latrange=None, lonrange=None,
                    **read_kwargs):
        """
        Returns the BARRA2 data

        Parameters:
           source_id_in (str): Model, e.g., BARRA-R2, BARRA-RE2, BARRA-C2
           freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
           variable_id (str): Variable id, e.g., tas, uas, pr
           version (str): Data release version if multiple available
           tstart (integer): Start of the time range, in yyyymmddHH format
           tend (integer): End of the time range, in yyyymmddHH format
           loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
           latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
           lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
           read_kwargs (dict): Arguments to pass to xarray.open_mfdataset
           
        Returns:
           xarray.Dataset: Extracted data
        """
        catalog_subset,files = self.get_barra2_files(source_id_in, freq_in, variable_id,
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

        if freq_in == 'fx':
            out = ds
        else:
            tstart = self._str2datetime(str(tstart), start=True)
            tend = self._str2datetime(str(tend), start=False)
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
            datetime.datetime: datetime matching t
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
    
    def get_barpa_files(self, source_id_in, driving_source_id_in, 
                    driving_experiment_id_in, freq_in, variable_id_in,
                    version="*",
                    tstart=19000101, 
                    tend=21010101):
        """
        Returns all the matching BARPA files in the NCI data collection.

        Parameters:
            source_id_in (str): Regional model, e.g. BARPA-R, BARPA-C
            driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
            driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
            freq_in (str): Time frequency of data, e.g., 1hr, day, mon
            variable_id_in (str): Variable name, e.g. tas, uas, pr
            version (str): Data release version if multiple available
            tstart (integer): Start of the time period, in yyyymmdd format
            tend (integer): End of the time period, in yyyymmdd format

        Returns:
           list of str: List of full paths to the files
    """ 
        query = dict(
            source_id=[source_id_in],
            driving_source_id=[driving_source_id_in],
            driving_experiment_id=[driving_experiment_id_in],
            freq=[freq_in],
            variable_id=[variable_id_in],
        )
        
        if tstart != 19900101 or tend != 21010101:
            full_time_range=self.keyvals["time_range"]
            time_range_list=self.get_time_range(tstart, tend, full_time_range)
            time_range_list.sort()
            query["time_range"] = time_range_list

        catalog_subset = self.data_catalog.search(**query)
        
        files=list(catalog_subset.df.path)
        files.sort()
        
        return(catalog_subset,files)

    
    def load_barpa_data(self, source_id_in, driving_source_id_in, 
                        driving_experiment_id_in, freq_in, variable_id_in,
                        version="*",
                        tstart=19000101,
                        tend=21010101,
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
            variable_id_in (str): Variable id, e.g. tas, uas, pr
            version (str): Data release version if multiple available
            tstart (integer): Start of the time period, in yyyymmddHH format
            tend (integer): End of the time period, in yyyymmddHH format
            loc (tuple of float), (latitude, longitude) if requesting data closest to a point location
            latrange (tuple of float), (latmin, latmax) if requesting data over a latitude range
            lonrange (tuple of float), (lonmin, lonmax) if requesting data over a longitude range
            read_kwargs (dict): Arguments to pass to xarray.open_mfdataset
    
        Returns:
           xarray.Dataset: Extracted data
        """
        catalog_subset,files = self.get_barpa_files(source_id_in, driving_source_id_in, 
                                                    driving_experiment_id_in, freq_in, 
                                                    variable_id_in, 
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
#        ds = catalog_subset.to_dataset_dict(cdf_kwargs=read_kwargs)                        

        if freq_in == 'fx':
            out = ds
        else:
            tstart = self._str2datetime(str(tstart), start=True)
            tend = self._str2datetime(str(tend), start=False)
        
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
            str: Name of the calendar type in this file
        """
        cube = iris.load(file)
        return cube[0].coords('time')[0].units.calendar





