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

font_size = 19
x = np.linspace(-20, 60, 200)
y = modelling_fctns.Wang_Engel_Temp_response(x, 8, 28, 36)
#y = Trapezoid_Temp_response(x, 0, 21.5, 32.5, 40)
fig, ax = plt.subplots(figsize = (7, 5))
ax.plot(x, y)
ax.set_xlabel('Temperature in degrees C', fontsize = font_size)
ax.set_ylabel('Response (scaled)', fontsize = font_size)
ax.set_title('Temperature response', fontsize = font_size + 1)
plt.xticks(fontsize = font_size - 4)
plt.yticks(fontsize = font_size - 4)
ax.set_xlim((0, 40))
ax.axvline(x = 28, linestyle= 'dashed', color = 'black')
devtime2020 = Maize_set.GDD_driver_data.where((Maize_set.GDD_driver_data['Referenzjahr'] == 2007)*(Maize_set.GDD_driver_data['Stations_id'] == 11193), drop=True)['Development Time']
driver2020 = Maize_set.GDD_driver_data.where((Maize_set.GDD_driver_data['Referenzjahr'] == 2007)*(Maize_set.GDD_driver_data['Stations_id'] == 11193), drop=True)['tas']
realtime2020 = Maize_set.GDD_driver_data.where((Maize_set.GDD_driver_data['Referenzjahr'] == 2007)*(Maize_set.GDD_driver_data['Stations_id'] == 11193), drop=True)['time']
font_size = 19
fig, ax = plt.subplots(figsize = (7, 5))
ax.plot(realtime2020, driver2020, label = 'Air temperature at site')
ax.plot(realtime2020, devtime2020*12.5, label = 'Development time (scaled for comparison)')
ax.set_xlim((realtime2020[0], realtime2020[120]))
ax.set_ylim((0, 25.5))
plt.xticks(rotation = 45, fontsize = font_size - 4)
plt.yticks(fontsize = font_size - 4)
ax.set_xlabel('Day of year', fontsize = font_size)
ax.set_ylabel('Temperature in degrees C', fontsize = font_size)
ax.set_title('Temperature compared to development time\nat one station', fontsize = font_size + 1)
Maize_set.ds_observed.xs(19732, level=1, drop_level=False)#[:, 19732]#19914]
#Maize_set.ds_observed.xs(1957, level=0, drop_level=False)