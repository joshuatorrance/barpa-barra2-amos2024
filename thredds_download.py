'''
  DESCRIPTION
    Python script for downloading data sets:
    - BARRA2 regional reanalysis,         https://dx.doi.org/10.25914/1x6g-2v48
    - BARPA regional climate projections, https://dx.doi.org/10.25914/z1x6-dq28
    from National Computing Infrastructure (NCI) THREDDS server.
    These data sets are part of the NCI data collections.

    Extended documentations:
    - BARRA2 reanalysis:  https://opus.nci.org.au/spaces/NDP/pages/264241166/BOM+BARRA2+ob53
    - BARPA projections:  https://opus.nci.org.au/spaces/NDP/pages/264241161/BOM+BARPA+py18

  PREREQUISITE
      - Python3.X
      - Extra libraries:
          https://github.com/joshuatorrance/barpa-barra2-amos2024
          https://github.com/bird-house/threddsclient

  AUTHOR
      Chun-Hsu Su, chunhsu.su@bom.gov.au, Bureau of Meteorology
      Joshua Torrance, joshua.torrance@bom.gov.au, Bureau of Meteorology
'''

import os
import argparse
import urllib
import loaddata  # https://github.com/joshuatorrance/barpa-barra2-amos2024
import threddsclient  # https://github.com/bird-house/threddsclient


def main():
    collections = ['BARRA2', 'BARPA']
    domain_ids = ['AUS-11', 'AUS-15',
                  'AUS-20i', 'AUS-22',
                  'AUST-11', 'AUST-15',
                  'AUST-22', 'AUST-04']
    driving_source_ids = ['ACCESS-CM2', 'ACCESS-ESM1-5',
                          'ERA5', 'NorESM2-MM', 'EC-Earth3',
                          'CESM2', 'CMCC-ESM2',
                          'MPI-ESM1-2-HR']
    driving_experiment_ids = ['historical',
                              'evaluation',
                              'ssp370',
                              'ssp126']

    parser = argparse.ArgumentParser(
        description="Downloads BARRA2 regional reanalysis or BARPA regional "
        "projections from NCI THREDDS"
    )

    # Adding arguments
    # Data selection arguments
    parser.add_argument('-C', '--collection', type=str, choices=collections,
                        required=True,
                        help='Name of the data collection')
    parser.add_argument('-d', '--domain', type=str, choices=domain_ids,
                        required=True,
                        help='Domain id. Available domains for BARRA2 are AUS-11, AUS-22, '
                        'AUST-11, AUST-22 and AUST-04. Available domains for BARPA are '
                        'AUS-15, AUS-20i, AUST-15 and AUST-04')
    parser.add_argument('-f', '--freq', type=str,
                        required=True,
                        help='Time frequency of the data, e.g., 1hr, day, mon')
    parser.add_argument('-n', '--variable', type=str,
                        required=True,
                        help='Variable name, e.g., tas, uas, pr')
    parser.add_argument('--start', type=str, default='190001',
                        help='Start of the time range, in yyyymm')
    parser.add_argument('--end', type=str, default='210101',
                        help='End of the time range, in yyyymm')

    # Output arguments
    parser.add_argument('-l', '--create_list', action='store_true',
                        help='Do not download, only create a text file listing '
                        'files to be downloaded')
    parser.add_argument('-o', '--out_dir', type=str, default=os.getcwd(),
                        help='Output directory to save the downloaded data files')

    # Only used for BARPA
    parser.add_argument('-g', '--driving_source', type=str,
                        choices=driving_source_ids,
                        help='Driving GCM name for BARPA only')
    parser.add_argument('-s', '--driving_experiment', type=str,
                        choices=driving_experiment_ids,
                        help='GCM experiment BARPA only')

    # Parsing arguments
    args = parser.parse_args()

    #
    # Construct the THREDDS server path for this data subset
    #
    if args.collection == 'BARRA2':
        data_project = 'ob53'
        root = loaddata.make_barra2_dirpath(args.domain,
                                            args.freq)
        dirpath = os.path.join(root, args.variable, 'latest')
        thredds_url = dirpath.replace(f'/g/data/{data_project}/{args.collection}/',
                                      f'https://thredds.nci.org.au/thredds/catalog/{data_project}/')
    else:
        data_project = 'py18'
        root = loaddata.make_barpa_dirpath(args.domain,
                                         args.driving_source,
                                         args.driving_experiment,
                                         args.freq)
        dirpath = os.path.join(root, args.variable, 'latest')
        thredds_url = dirpath.replace(f'/g/data/{data_project}/',
                                      f'https://thredds.nci.org.au/thredds/catalog/{data_project}/')

    thredds_url = os.path.join(thredds_url, 'catalog.html')

    print(f"INFO: thredds URL: {thredds_url}")

    #
    # Get a listing of the files, from the server, to be downloaded
    #
    data_pairs = []
    for ds in threddsclient.crawl(thredds_url, depth=1):
        filepath = ds.url.split("=")[1]

        # check if it is within time range
        if len(loaddata.screen_files([filepath], tstart=args.start, tend=args.end)) == 0:
            continue

        src_dir = os.path.dirname(filepath)
        basename = os.path.basename(filepath)

        # new destination directory
        new_dst_dir = os.path.join(args.out_dir, src_dir)
        # new file to be created in the new destination directory
        new_file = os.path.join(new_dst_dir, basename)

        data_pairs.append( [ds.download_url(), new_file] )

    n = len(data_pairs)
    print(f"INFO: {n} files found")

    #
    # Write the file listing to file only
    #
    if args.create_list:
        outfile = os.path.join(args.out_dir, 'file_list.txt')
        with open(outfile, 'w', encoding="utf-8") as fout:
            for pair in data_pairs:
                print(pair[0], file=fout)

        print(f"INFO: File listing written to {outfile}")
        return

    #
    # Download the data to args.out_dir
    #
    error_counter = 0
    for i, pair in enumerate(data_pairs):
        src_file = pair[0]
        new_file = pair[1]

        dst_dir = os.path.dirname(new_file)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        try:
            print(f'INFO: Downloading {i+1} of {n}: {src_file} -> {new_file}\n')
            urllib.request.urlretrieve(src_file, new_file)
        except (SystemExit, KeyboardInterrupt):
            print("\nInterrupt detected, aborting.")
            break
        except (urllib.error.HTTPError, urllib.error.ContentTooShortError):
            print(f'ERROR: Unable to retrieve: {src_file} -> {new_file}')
            error_counter += 1

    #
    # Finish
    #
    n_success = n - error_counter
    print(f"INFO: {n_success}/{n} of files downloaded, and {error_counter} errors reported")


if __name__ == '__main__':
    main()
