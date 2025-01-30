"""
    NAME
        datapaths

    DESCRIPTION
        datapaths is a Python module for interfacing with the BARRA2 and BARPA
        data sets in the NCI data collection that handles paths and other
        functionality without any dependance on packages such as xarray and
        iris.

    AUTHORS
        Chun-Hsu Su, chunhsu.su@bom.gov.au, Bureau of Meteorology
        Joshua Torrance, joshua.torrance@bom.gov.au, Bureau of Meteorology
"""
import calendar
from datetime import datetime
import os


### Shared Functions
def str2datetime(t, start=True):
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
        h = int(t[8:])
        return datetime(y, m, d, h, 0)

    return


def screen_files(files, tstart='196001', tend='210101'):
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
    tstart = str2datetime(tstart, start=True)
    tend = str2datetime(tend, start=False)

    files_filt = []
    for file in files:
        bn = os.path.basename(file)
        timerange = os.path.splitext(bn)[0].split("_")[-1]
        t0 = str2datetime(timerange.split("-")[0], start=True)
        t1 = str2datetime(timerange.split("-")[1], start=False)

        if t1 < tstart:
            #print("{:} < {:}".format(t1, tstart))
            continue
        if t0 > tend:
            continue
        files_filt.append(file)

    files_filt.sort()

    return files_filt


### BARRA2 Functions
def make_barra2_dirpath(id_in, freq_in):
    """
    Returns path to the BARRA2 data directory.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARRA-R2 (either AUS-11 or AUST-11),
                            BARRA-RE2 (AUS-22), BARRA-C2 (AUST-04), or
                            AUS-11, AUST-11, AUS-22, and AUST-04
        freq_in (str): Time frequency of the data, e.g. 1hr, day, mon

    Returns:
        str: Directory path
    """
    basepath = '/g/data/ob53/BARRA2/output'

    rootdir_templ = "{basepath}/reanalysis/{domain_id}/BOM/{driving_source_id}/" \
        "{driving_experiment_id}/{driving_variant_label}/{source_id}/v1/{freq}"

    # default
    model_dict = {'BARRA-R2': ("hres", "AUS-11"),
                  'BARRA-RE2': ("eda", "AUS-22"),
                  'BARRA-C2': ("hres", 'AUST-04')}
    domain_id_dict = {"AUS-11": ("hres", "BARRA-R2"),
                      "AUST-11": ("hres", "BARRA-R2"),
                      "AUS-22": ("eda", "BARRA-RE2"),
                      "AUST-22": ("eda", "BARRA-RE2"),
                      "AUST-04": ("hres", "BARRA-C2")}

    if id_in.startswith("BARRA-"):
        source_id = id_in
        driving_variant_label = model_dict[id_in][0]
        domain_id = model_dict[id_in][1]
    elif id_in.startswith("AUS"):
        domain_id = id_in
        driving_variant_label = domain_id_dict[id_in][0]
        source_id = domain_id_dict[id_in][1]
    else:
        assert False, f"Unknown {id_in}. Permittable values: BARPA-R, BARPA-C, " \
        "AUS-15, AUST-15, AUS-20i or AUST-04"

    driving_source_id = "ERA5"
    driving_experiment_id = "historical"

    path = rootdir_templ.format(basepath=basepath,
                                domain_id=domain_id,
                                driving_source_id=driving_source_id,
                                driving_experiment_id=driving_experiment_id,
                                driving_variant_label=driving_variant_label,
                                source_id=source_id,
                                freq=freq_in)

    #print(f"path={path}")
    return path


def list_barra2_variables(id_in, freq_in):
    """
    Prints a listing of the variables available for BARRA2 model.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARRA-R2 (either AUS-11 or AUST-11),
                            BARRA-RE2 (AUS-22 or AUST-22), BARRA-C2 (AUST-04), or
                            AUS-11, AUST-11, AUS-22, AUST-22, and AUST-04
        freq_in (str): Time frequency of the data, e.g. 1hr, day, mon
    """
    rootdir = make_barra2_dirpath(id_in, freq_in)

    varlist = os.listdir(rootdir)
    print(", ".join(varlist))

    return varlist


def list_barra2_freqs(id_in):
    """
    Prints a listing of the time frequency available for BARRA2 model data.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARRA-R2 (either AUS-11 or AUST-11),
                            BARRA-RE2 (AUS-22 or AUST-22), BARRA-C2 (AUST-04), or
                            AUS-11, AUST-11, AUS-22, AUST-22, and AUST-04
    """
    rootdir = make_barra2_dirpath(id_in, 'fx')

    freqlist = os.listdir(rootdir+'/..')
    print(", ".join(freqlist))

    return freqlist


### BARPA Functions
def make_barpa_dirpath(id_in, driving_source_id_in, driving_experiment_id_in, freq_in):
    """
    Returns path to the BARPA data directory.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARPA-R (AUS-15) or BARPA-C (AUST-04), or
                            AUS-15, AUST-15, AUS-20i, AUST-04
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon

    Returns:
        str: Directory path
    """
    # default to use AUS-15
    model_dict = {'BARPA-R': ('AUS-15'),
                  'BARPA-C': ('AUS-04')}

    domain_id_dict = {"AUS-15": ('BARPA-R'),
                      "AUST-15": ('BARPA-R'),
                      "AUS-20i": ("BARPA-R1-NN"),
                      "AUST-04": ("BARPA-C")}

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
    rootdir_templ = "{basepath}/CMIP6/DD/{domain_id}/BOM/{driving_source_id}/" \
        "{driving_experiment_id}/{driving_variant_label}/{source_id}/v1-r1/{freq}"

    driving_variant_label = gcm_ens[driving_source_id_in]

    if id_in.startswith("BARPA-"):
        source_id = id_in
        domain_id = model_dict[id_in]
    elif id_in.startswith("AUS"):
        domain_id = id_in
        source_id = domain_id_dict[id_in]
    else:
        assert False, f"Unknown {id_in}. Permittable values: BARPA-R, BARPA-C, " \
        "AUS-15, AUST-15, AUS-20i or AUST-04"

    path = rootdir_templ.format(basepath=basepath, domain_id=domain_id,
                                driving_source_id=driving_source_id_in,
                                driving_experiment_id=driving_experiment_id_in,
                                driving_variant_label=driving_variant_label,
                                source_id=source_id,
                                freq=freq_in)

    #print(f"path={path}")
    return path


def list_barpa_variables(id_in,
                         driving_source_id_in,
                         driving_experiment_id_in,
                         freq_in):
    """
    Prints a listing of the variables available for BARPA model.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARPA-R (AUS-15) or BARPA-C (AUST-04), or
                            AUS-15, AUST-15, AUS-20i, AUST-04
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation
        freq_in (str): Time frequency of data, e.g., 1hr, day, mon

    Returns:
        list of str: List of variable_id values.
    """
    rootdir = make_barpa_dirpath(id_in,
                                 driving_source_id_in,
                                 driving_experiment_id_in,
                                 freq_in)

    varlist = os.listdir(rootdir)
    print(", ".join(varlist))

    return varlist


def list_barpa_freqs(id_in, driving_source_id_in, driving_experiment_id_in):
    """
    Prints a listing of the time frequency available for BARPA model data.

    Parameters:
        id_in (str): Model name or the domain id, e.g., BARPA-R (AUS-15) or BARPA-C (AUST-04), or
                            AUS-15, AUST-15, AUS-20i, AUST-04
        driving_source_id_in (str): Driving GCM name, e.g. ACCESS-CM2
        driving_experiment_id_in (str): GCM experiment, e.g. historical, ssp370, ssp126, evaluation

    Returns:
        list of str: list of freq values.
    """
    rootdir = make_barpa_dirpath(id_in, driving_source_id_in, driving_experiment_id_in, 'fx')

    freqlist = os.listdir(rootdir+'/..')
    print(", ".join(freqlist))

    return freqlist
