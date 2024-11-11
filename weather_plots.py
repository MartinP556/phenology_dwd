import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def read_hyras(file_start, file_end, start_year, end_year):
    years = [str(yr) for yr in np.arange(start_year, end_year + 1, 1)]
    locations = [file_start + s_yr + file_end + '#mode=bytes' for s_yr in years]
    full_dataset = xr.load_dataset(locations[0])
    print('opened first set')
    #for i, file_name in enumerate(locations[1:]):
    #    print(i)
    #    ds = xr.load_dataset(file_name)
    #    full_dataset = xr.concat([ds, full_dataset], dim='time')
    full_dataset = xr.open_mfdataset(locations)
    full_dataset = 12
    return full_dataset

#hyras = read_hyras('https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/tas_hyras_5_', '_v5-0_de.nc', 1990, 2000)
#print(hyras['time'])

def plot_averages(dataset, variable, save_name, font_size = 18):
    fig, ax = plt.subplots(figsize = (10, 5))
    means = dataset.groupby('time.dayofyear').mean(dim=['time', 'x', 'y'])[variable]
    print('means obtained')
    stds = dataset.groupby('time.dayofyear').std(dim=['time']).mean(dim = ['x', 'y'])[variable]
    print('standard deviations obtained')
    ax.plot(pd.date_range('1999-01-01', '1999-12-31'), means)
    print('means plotted')
    ax.fill_between(pd.date_range('1999-01-01', '1999-12-31'), means - stds, means + stds)
    ax.set_xlabel('Day of year', fontsize = font_size)
    ax.set_ylabel(variable, fontsize = font_size)
    ax.set_title(variable + ' daily means and standard deviations over period')
    plt.savefig('C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\plots\\' + save_name)

ds = xr.open_dataset('https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_max/tasmax_hyras_5_1951_2020_v5-0_de.nc#mode=bytes', chunks={"time": 2}).thin({"time": 20 , "x": 20, "y": 20})#, chunks={"time": 3, "x": 5, "y": 5}, "x": 5, "y": 5

#print('dataset opened')
#plot_averages(ds, 'tasmax', 'maxes_time_series.png')