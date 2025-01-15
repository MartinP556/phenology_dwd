import argparse

def define_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for downloading satellite data from earth engine"
    )

    parser.add_argument(
        "-b", "--basedir", dest="basedir", default="", help="path to files"
    )

    parser.add_argument(
        "-s",
        "--savename",
        dest="savename",
        default="",
        help="Name of file to save scraped timeseries",
    )

    parser.add_argument(
        "-start",
        "--start",
        dest="start",
        default="",
        help="start point to read in coord list",
    )

    parser.add_argument(
        "-end",
        "--end",
        dest="end",
        default="",
        help="end point to read in list",
    )

    return parser