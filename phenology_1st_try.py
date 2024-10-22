import numpy as np
import pandas as pd
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
#import ee
#ee.Authenticate()
# Initialize the Earth Engine module.
#ee.Initialize()

# Print metadata for a DEM dataset.
#print(ee.Image('USGS/SRTMGL1_003').getInfo())
## READ IN DATA ##
#phen_data = pd.read_csv("https://opendata.dwd.de/climate_environment/CDC/observations_germany/phenology/annual_reporters/crops/recent/PH_Jahresmelder_Landwirtschaft_Kulturpflanze_Mais_akt.txt", engine='python', sep = r';\s+|;\t+|;\s+\t+|;\t+\s+|;|\s+;|\t+;|\s+\t+;|\t+\s+;')
#phen_data = phen_data.drop('Unnamed: 9', axis = 1)
phase_names = pd.read_csv("https://opendata.dwd.de/climate_environment/CDC/help/PH_Beschreibung_Phase.txt", encoding = "latin1", engine='python', sep = r';\s+|;\t+|;\s+\t+|;\t+\s+|;|\s+;|\t+;|\s+\t+;|\t+\s+;')
phen_data = pd.read_csv("https://opendata.dwd.de/climate_environment/CDC/observations_germany/phenology/annual_reporters/crops/historical/PH_Jahresmelder_Landwirtschaft_Kulturpflanze_Mais_1936_2023_hist.txt", sep = ';\s+|;\t+|;\s+\t+|;\t+\s+|;|\s+;|\t+;|\s+\t+;|\t+\s+;')
phen_data = phen_data.drop('Unnamed: 9', axis = 1)
station_data = pd.read_csv("https://opendata.dwd.de/climate_environment/CDC/help/PH_Beschreibung_Phaenologie_Stationen_Jahresmelder.txt",sep = ";\s+|;\t+|;\s+\t+|;\t+\s+|;|\s+;|\t+;|\s+\t+;|\t+\s+;", encoding='cp1252', on_bad_lines='skip')
station_data.index = station_data['Stations_id']
print(phen_data.columns)
station_data = station_data.drop('Unnamed: 12', axis = 1)
#print(station_data.head())
#print(phen_data.head(20))
#print(phen_data['Unnamed: 9'])
#print(phen_data.columns)
## READ IN DATA FOR PHASE NAMES IN ORDER ##
phen_data['Order of phase'] = np.nan
phen_data['Name of phase'] = np.nan
#print(phase_names['Phase_ID'])
def get_phase_name(phaseid):
    return phase_names['Phase_englisch'][phase_names['Phase_ID'] == str(phaseid)].values[0]
def get_station_locations(dataset):
    lat = [station_data._get_value(row, col) for row, col in zip(dataset['Stations_id'], ['geograph.Breite' for count in range(len(dataset))])] #station_data.lookup(row_labels = dataset['Stations_id'], col_labels = ['geograph.Breite'])
    lon = [station_data._get_value(row, col) for row, col in zip(dataset['Stations_id'], ['geograph.Laenge' for count in range(len(dataset))])] #station_data._lookup(dataset['Stations_id'], ['geograph.Laenge'])
    return lat, lon
LAT, LON = get_station_locations(phen_data)
phen_data['lat'] = LAT
phen_data['lon'] = LON
for i, phaseid in enumerate([10, 12, 67, 65, 5, 6, 19, 20, 21, 24, ]):
    #print(phase_names['Phase_englisch'][phase_names['Phase_ID'] == str(phaseid)])
    if len(phase_names['Phase_englisch'][phase_names['Phase_ID'] == str(phaseid)]) != 0:
        #print(i, phaseid)
        phen_data.loc[phen_data['Phase_id'] == phaseid, 'Order of phase'] = i
        phen_data.loc[phen_data['Phase_id'] == phaseid, 'Name of phase'] = get_phase_name(phaseid)
#print(station_data.columns)
#phen_data['Phase_id'] = phen_data['Phase_id'].replace(5, 50)
#phen_data = phen_data.set_index(['Stations_id', 'Referenzjahr', 'Eintrittsdatum'])
#print(phen_data.head())
#print(phen_data[phen_data['Stations_id'] == 7521])
#print(phen_data.loc[(slice(None), 2021, slice(None)), :])
#print(phen_data.index)
#phen_data = phen_data.sort_index()
## SORT BY TIME ##
phen_data = phen_data.sort_values(by = ['Stations_id', 'Referenzjahr', 'Eintrittsdatum', 'Order of phase'])
## CONVERT DATE TO DATETIME ##
phen_data['Eintrittsdatum'] = pd.to_datetime(phen_data['Eintrittsdatum'], format = '%Y%m%d')
## CALCULATE TIME TO NEXT STAGE ##
phen_data['Time to next stage'] = phen_data['Eintrittsdatum'].shift(-1) - phen_data['Eintrittsdatum']
phen_data['Next stage name'] = phen_data['Name of phase'].shift(-1)
## EXCLUDE CHANGES IN STATION ##
phen_data.loc[phen_data['Stations_id'] != phen_data['Stations_id'].shift(-1), 'Time to next stage'] = np.nan
phen_data.loc[phen_data['Stations_id'] != phen_data['Stations_id'].shift(-1), 'Next stage name'] = np.nan
print(phen_data.head(30))
#print(phen_data.isnull().sum())
## COUNT NUMBER AND AVERAGE LENGTH GOING FROM ONE PHASE TO ANOTHER ##
def count_transition_numbers():
    for phase_name in phen_data['Name of phase'].unique():
        for next_phase in phen_data['Next stage name'].unique():
            obs_this_change = phen_data.where((phen_data['Name of phase'] == phase_name)&(phen_data['Next stage name'] == next_phase)).dropna()
            if len(obs_this_change) != 0:
                print('Number going from ', phase_name, ' to ', next_phase, ': ', len(obs_this_change))
                print('Average time going from ', phase_name, ' to ', next_phase, ': ', obs_this_change['Time to next stage'].mean())
#count_transition_numbers()
## PLOT LOCATIONS OF STATIONS ##   
def plot_station_locations(font_size = 20):
    adm1_shapes = list(shpreader.Reader('Code/gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    fig,ax=plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines(resolution='10m')
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(), edgecolor='black', facecolor='none', alpha=1)
    ax.set_extent([5, 16, 47, 56], ccrs.PlateCarree())
    ax.set_title('Locations of stations', fontsize = font_size)
    for station_id in phen_data['Stations_id'].unique():
        station = station_data[station_data['Stations_id'] == station_id]
        ax.plot(station['geograph.Laenge'], station['geograph.Breite'], 'o', color = 'red', markersize = 5)
    fig.savefig('plots/station_locations.png')
#plot_station_locations()
## PLOT AVERAGE NUMBER OF OBSERVATIONS PER STATION ##
#print(phen_data.groupby('Stations_id').size())
def plot_avg_num_obs_grouped(grouping_variable, save_name, title, font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.hist(phen_data.groupby(grouping_variable).size(), bins = 30)
    ax.set_title('Average number of observations per station', fontsize = font_size)
    ax.set_xlabel('Number of observations', fontsize = font_size)
    ax.set_ylabel('Number of stations', fontsize = font_size)
    fig.savefig('plots/' + save_name + '.png')
#plot_avg_num_obs_grouped('Stations_id', 'avg_num_obs_per_station', 'Histogram number of obs per year')
#plot_avg_num_obs_grouped('Referenzjahr', 'avg_num_obs_per_year', 'Histogram number of obs per station')
#print(phen_data.groupby(['Stations_id', 'Referenzjahr']).size())
def plot_avg_num_yearly_obs(font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 10))
    obsnums_yearly = phen_data.groupby(['Stations_id', 'Referenzjahr']).size()
    ax.hist(obsnums_yearly.groupby('Stations_id').mean(), bins = 30)
    ax.set_title('Average number of observations per year', fontsize = font_size)
    ax.set_xlabel('Number of observations', fontsize = font_size)
    ax.set_ylabel('Average number of obs per year', fontsize = font_size)
    fig.savefig('plots/avg_num_yearly_obs.png')
#plot_avg_num_yearly_obs()
## PLOT AVERAGE LENGTH OF PHASE BY LOCATION ##
def length_phase_box_plot(phase_names, font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 10))
    phase_lengths = []
    for phase_name in phase_names:
        obs_this_phase = phen_data[phen_data['Name of phase'] == phase_name]
        phase_lengths.append(obs_this_phase['Time to next stage'].dropna().dt.days.values)
    #print(phase_lengths)
    ax.boxplot(phase_lengths, tick_labels=phase_names, widths = 0.5) #positions = [obs_this_phase['Stations_id'].unique()[0]],
    ax.set_ylim(0, 100)
    plt.xticks(rotation = 90)
    ax.tick_params(labelsize = font_size)
    ax.set_title('Length of phases', fontsize = font_size)
    #ax.set_xlabel('Phase', fontsize = font_size)
    ax.set_ylabel('Length of phase (days)', fontsize = font_size)
    fig.savefig('plots/length_phase_box_plot.png', bbox_inches='tight')

#length_phase_box_plot(['beginning of tilling sowing drilling', 
#                        'beginning of emergence', 
#                        'beginning of growth in height', 
#                        'tip of tassel visible', 
#                        'beginning of flowering', 
#                        'beginning of mil ripeness', 
#                        'beginning of wax-ripe stage'])

def day_of_emergence_map(binsize = 0.25, font_size = 20):
    adm1_shapes = list(shpreader.Reader('Code/gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    fig, ax = plt.subplots(figsize = (10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines(resolution='10m')
    adm1_shapes = list(shpreader.Reader('Code/gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(), edgecolor='black', facecolor='none', alpha=1)
    ax.set_extent([5, 16, 47, 56], ccrs.PlateCarree())
    lons = []
    lats = []
    emergence_days = []
    for LON in np.arange(5, 16, binsize):
        for LAT in np.arange(47, 56, binsize):
            print(phen_data['lat'])
            print(phen_data['lon'])
            mask = (phen_data['Name of phase'] == 'beginning of emergence')*np.greater_equal(phen_data['lat'], LAT)*np.less(phen_data['lat'], LAT + binsize)*np.greater_equal(phen_data['lon'], LON)*np.less(phen_data['lon'], LON + binsize)
            #mask = (phen_data['Name of phase'] == 'beginning of emergence')*(phen_data['lat'] >= LAT)*(phen_data['lat'] < LAT + binsize)*(phen_data['lon'] >= LON)*(phen_data['lon'] < LON + binsize)
            if len(phen_data[mask]) == 0:
                continue
            average_emergence_day = phen_data[mask]['Jultag'].dropna().mean()
            #print(phen_data[mask]['Jultag'])
            #print(average_emergence_day)
            lons.append(LON)
            lats.append(LAT)
            emergence_days.append(average_emergence_day)
    emergence_map = ax.scatter(lons, lats, s = 20, c = emergence_days, cmap = 'Purples') #c = average_emergence_day, cmap = 'Purples')
    #ax._colorbars()
    #plt.colorbar()
    cbar = fig.colorbar(emergence_map, shrink = 0.5)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Average day of emergence\n(from 1st Jan)', size = font_size)
    ax.set_title('Average day of emergence\n(from 1st Jan)', size = font_size)
    fig.savefig('plots/day_of_emergence_map.png')
day_of_emergence_map()
#print(phen_data[['Eintrittsdatum', 'Name of phase', 'Time to next stage']])
#temp_data = 12
#print(station_data['Unnamed: 2'])
#def next_observed_phase()