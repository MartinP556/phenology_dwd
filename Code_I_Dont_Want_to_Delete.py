#phen_data = phen_data.set_index(['Stations_id', 'Referenzjahr', 'Eintrittsdatum'])
#print(phen_data.head())
#print(phen_data[phen_data['Stations_id'] == 7521])
#print(phen_data.loc[(slice(None), 2021, slice(None)), :])
#print(phen_data.index)
#phen_data = phen_data.sort_index()

def read_hyras(file_start, file_end, start_year, end_year):
    years = [str(yr) for yr in np.arange(start_year, end_year + 1, 1)]
    locations = [file_start + s_yr + file_end + '#mode=bytes' for s_yr in years]
    full_dataset = xr.open_dataset(locations[0])
    for i, file_name in enumerate(locations[1:]):
        ds = xr.open_dataset(file_name)
        full_dataset = xr.concat([ds, full_dataset], dim='time')
    #full_dataset = xr.open_mfdataset(locations)
    full_dataset = 12
    return full_dataset

hyras = read_hyras('https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/air_temperature_mean/tas_hyras_5_', '_v5-0_de.nc', 1990, 1991)