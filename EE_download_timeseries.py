import ee
from google.oauth2 import service_account
import numpy as np
import pandas as pd
import argparse
import argument_names

def satellite_data_at_coords(coords, start_date = '2000-01-01', end_date = '2022-12-31', instrument = "COPERNICUS/S2_SR_HARMONIZED", bands = ['B4', 'B8'], box_width = 0.002, pixel_scale = 500):
    for coord_index, coord in enumerate(coords):
        print(coord_index)
        location = ee.Geometry.BBox(coord[1] - box_width, coord[0] - box_width, coord[1] + box_width, coord[0] + box_width) #ee.Geometry.Point(coord[:2])#
        dataset = ee.ImageCollection(instrument)
        filtered_dataset = dataset.filterDate(start_date, end_date).filterBounds(location)
        time_series_data = filtered_dataset.select(*bands).map(lambda img: img.set('mean', img.reduceRegion(ee.Reducer.median(), location , pixel_scale)))#.getInfo()#.getRegion(location, pixel_scale).getInfo()#.getRegion(location, 30).getInfo()#('B4')#.median()#.get('B4')#reduceColumns(ee.Reducer.median().setOutputs(['median']), [''])#.get('B4')
        timelist = time_series_data.aggregate_array('system:time_start').getInfo()
        bandlist = time_series_data.aggregate_array('mean').getInfo()
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

df = satellite_data_at_coords(coords[startpoint:endpoint], start_date='2000-10-01')

df.to_csv(root_directory + f"Saved_files/{savename}.csv")

