import ee
from google.oauth2 import service_account
import numpy as np
import pandas as pd
import argparse
import argument_names

def mask_MODIS_clouds(image):
    """Masks clouds in a Sentinel-2 image using the stateQA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select('state_1km')
    # Bits 0-1 are cloud, 2 cloud shadow, 8-9 cirrus
    cloud_bit_mask = 3 << 0
    #cloud_bit_mask2 = 1 << 1
    cloud_shadow_bit_mask = 1 << 2
    cirrus_bit_mask = 3 << 8
    #cirrus_bit_mask2 = 1 << 9
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)
             .And(qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0)
                 )
            )
    )
    return image.updateMask(mask)

def MODIS_Mask_QC(image):
    """Masks clouds in a Sentinel-2 image using the stateQA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select('QC_500m')
    # Bits 0-1 are cloud, 2 cloud shadow, 8-9 cirrus
    b1_mask = 15 << 2
    b2_mask = 15 << 6
    b3_mask = 15 << 10
    b4_mask = 15 << 14
    mask = (
        qa.bitwiseAnd(b1_mask).eq(0)
        .And(qa.bitwiseAnd(b2_mask).eq(0)
             .And(qa.bitwiseAnd(b3_mask).eq(0)
                  .And(qa.bitwiseAnd(b4_mask).eq(0)
                      )
                 )
            )
    )
    return image.updateMask(mask)

def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask)

def mask_s2_clouds_collection(image_collection):
    """Masks clouds in a Sentinel-2 image collection using the SLC band.

    Args:
        image (ee.ImageCollection): A Sentinel-2 image collection.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image collection.
    """
    return image_collection.map(mask_s2_clouds)

def satellite_data_at_coords(coords, start_date = '2000-01-01', end_date = '2022-12-31', instrument = "COPERNICUS/S2_SR_HARMONIZED", bands = ['B4', 'B8'], box_width = 0.002, pixel_scale = 500, QC_function = mask_s2_clouds_collection):
    for coord_index, coord in enumerate(coords):
        print(coord_index)
        location = ee.Geometry.BBox(coord[1] - box_width, coord[0] - box_width, coord[1] + box_width, coord[0] + box_width) #ee.Geometry.Point(coord[:2])#
        dataset = ee.ImageCollection(instrument)
        dataset = QC_function(dataset)
        filtered_dataset = dataset.filterDate(start_date, end_date).filterBounds(location)
        time_series_data = filtered_dataset.select(*bands).map(lambda img: img.set('median', img.reduceRegion(ee.Reducer.median(), location , pixel_scale)))
        timelist = time_series_data.aggregate_array('system:time_start').getInfo()
        bandlist = time_series_data.aggregate_array('median').getInfo()
        dataset = {'Time': timelist,
                   'lat': [coord[0] for count in range(len(timelist))],
                   'lon': [coord[1] for count in range(len(timelist))],
                   'Stations_Id': [np.int64(coord[2]) for count in range(len(timelist))]
                   }
        for band in bands:
            dataset[f'Median {band}'] = [band_data[band] for band_data in bandlist]
        df = pd.DataFrame(dataset)
        df['formatted_time'] = pd.to_datetime(df['Time'], unit='ms').dt.strftime('%Y-%m-%d-%H-%M-%S')
        if coord_index == 0:
            df_full = df
        else:
            df_full = pd.concat([df_full, df])
    return df_full

parser = argument_names.define_parser()
args = parser.parse_args()

startpoint = np.int64(args.start)
endpoint = np.int64(args.end)
savename = args.savename
print(startpoint, endpoint, savename)
# Path to the private key file
key_path = 'Access/ee-martinparker637-e68fde65abb4.json'

# Load the service account credentials
credentials = service_account.Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/earthengine'])

# Initialize Earth Engine with the service account credentials
ee.Initialize(credentials)

root_directory = ''#'home/users/wlwc1989/phenology_dwd/'
#root_directory = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\'

coords = np.loadtxt(root_directory + "Saved_files/station_coords.csv", delimiter=',')

#df = satellite_data_at_coords(coords[startpoint:endpoint], start_date='2000-10-01')
import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile()
pr.enable()
# ... do something ...
df = satellite_data_at_coords(coords[startpoint:endpoint], start_date='2000-01-01', QC_function = lambda IC: IC.map(mask_MODIS_clouds).map(MODIS_Mask_QC), bands = [f'sur_refl_b0{n}' for n in range(1, 5)])

df.dropna().to_csv(root_directory + f"Saved_files/{savename}.csv")

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())