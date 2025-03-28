#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project1_main.py
Author: Reese Barrett
Date: 2023-10-31

Description: Main script for Project 1, calls functions written in project1.py
    for data analysis
"""

import project1 as p1
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
from scipy import stats
from scipy.io import loadmat
from sklearn import linear_model
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm
import cmocean
import cmocean.cm as cmo
import PyCO2SYS as pyco2
import xarray as xr


filepath = '/Users/Reese_1/Documents/Research Projects/project1/data/' # where GLODAP data is stored
gridded_filepath = '/Users/Reese_1/Documents/Research Projects/project2/GLODAPv2.2016b.MappedProduct/' # where GLODAP gridded data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)
input_mc_cruise_file = 'G2talk_mc_simulated.csv' # MC (per cruise) simulated data
input_mc_individual_file = 'G2talk_mc_individual_simulated.csv' # MC (per cruise) simulated data
coeffs_file = 'ESPER_LIR_coeffs.csv' # ESPER LIR coefficients saved from MATLAB
monthly_clim_file = 'monthlyclim.mat'
input_bats_file = 'bats_bottle.txt'
input_hot_file = 'niskin_v2.txt'

# %% import GLODAP data file
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

 # %% filter data for only GO-SHIP + associated cruises
go_ship, go_ship_cruise_nums_2023 = p1.go_ship_only(glodap)

# %% do quality control
go_ship = p1.glodap_qc(go_ship)

# %% convert time to decimal time and datetime for use in ESPERs
go_ship = p1.glodap_reformat_time(go_ship)

# %% call ESPERs
# this is done in MATLAB for now, will update when code is translated
# need to get Python ESPERs package and translate my MATLAB code calling ESPERs into Python
# for now: 1. save processed_glodap, 2. process in MATLAB (call_ESPERs.m), 3. upload again to here
# step 1: save processed_glodap

# select relevant columns
glodap_out = go_ship[['G2expocode','G2cruise','G2station','G2region','G2cast',
                 'dectime','datetime','G2latitude','G2longitude','G2depth',
                 'G2temperature','G2salinity','G2oxygen','G2nitrate',
                 'G2silicate','G2phosphate','G2talk','G2phtsinsitutp']]
        
# glodap.to_csv(filepath + 'GLODAPv2.2022_for_ESPERs.csv', index=False) # for 2022
glodap_out.to_csv(filepath + 'GLODAPv2.2023_for_ESPERs.csv', index=False) # for 2023

# %% output for Brendan to re-train LIR
# select relevant columns
glodap_out = go_ship[['G2expocode','G2cruise','G2station','G2region','G2cast',
                      'dectime','datetime','G2latitude','G2longitude',
                      'G2depth','G2temperature','G2salinity','G2salinityf',
                      'G2salinityqc','G2oxygen', 'G2oxygenf','G2oxygenqc',
                      'G2nitrate','G2nitratef','G2nitrateqc','G2silicate',
                      'G2silicatef','G2silicateqc','G2phosphate',
                      'G2phosphatef','G2phosphateqc','G2talk', 'G2talkf',
                      'G2talkqc','G2phtsinsitutp','G2phtsinsitutpf',
                      'G2phtsqc']]
        
# glodap.to_csv(filepath + 'GLODAPv2.2022_for_ESPERs.csv', index=False) # for 2022
glodap_out.to_csv(filepath + 'GLODAPv2.2023_for_Brendan.csv', index=False) # for 2023
 
# %% upload ESPERs outputs to here
espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv') # to do the normal ESPER
#espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA_GO-SHIP_LIR.csv') # to do the GO-SHIP trained ESPER
espers['datetime'] = pd.to_datetime(espers['datetime']) # recast datetime as datetime data type

# %% set depth to use as boundary between surface and deep ocean

# static depth boundary
espers['surface_depth'] = 25

# to use dynamic mixed layer depth
monthly_clim = loadmat(filepath + monthly_clim_file)
MLD_da_max = monthly_clim['mld_da_max']
MLD_da_mean = monthly_clim['mld_da_mean']
latm = monthly_clim['latm']
lonm = monthly_clim['lonm']

max_MLDs = p1.find_MLD(espers.G2longitude, espers.G2latitude, MLD_da_max, latm, lonm, 0)
mean_MLDs = p1.find_MLD(espers.G2longitude, espers.G2latitude, MLD_da_mean, latm, lonm, 1)

#espers['surface_depth'] = max_MLDs

#%% calculate percentage of the ocean where max MLD is greater than saturation horizon
OmegaA_data = xr.open_dataset(gridded_filepath + 'GLODAPv2.2016b.OmegaA.nc')
OmegaA_data = OmegaA_data.set_coords('Depth').rename({'Depth':'depth'}) # change depth from data variable to coordinate
OmegaA_data = OmegaA_data.rename({'depth_surface': 'depth'})

OmegaC_data = xr.open_dataset(gridded_filepath + 'GLODAPv2.2016b.OmegaC.nc')
OmegaC_data = OmegaC_data.set_coords('Depth').rename({'Depth':'depth'}) # change depth from data variable to coordinate
OmegaC_data = OmegaC_data.rename({'depth_surface': 'depth'})

lon_name = 'lon'

# convert glodap longitude to -180 to 180 (first, subtract 20 from everything greater than 360)
OmegaA_data.coords['lon'] = xr.where(OmegaA_data.coords['lon'] > 360, OmegaA_data.coords['lon'], OmegaA_data.coords['lon'] - 20)
OmegaA_data.coords['lon'] = (OmegaA_data.coords['lon'] + 180) % 360 - 180
OmegaA_data = OmegaA_data.sortby(OmegaA_data.lon)

OmegaC_data.coords['lon'] = xr.where(OmegaC_data.coords['lon'] > 360, OmegaC_data.coords['lon'], OmegaC_data.coords['lon'] - 20)
OmegaC_data.coords['lon'] = (OmegaC_data.coords['lon'] + 180) % 360 - 180
OmegaC_data = OmegaC_data.sortby(OmegaC_data.lon)

# find deepest depth of each Omega array where Omega is > 1
# where the water column has a value for saturation state, get the index of the shallowest depth level where saturation state < 1
OmegaA = OmegaA_data['OmegaA'].to_numpy() # convert from xarray to numpy
valid_mask = np.isnan(OmegaA) # mask out NaNs
undersaturated_mask = (OmegaA < 1) & ~valid_mask # find places where the water is undersaturated (skipping NaNs)
lowest_undersaturated_idx = np.argmax(undersaturated_mask, axis=0) # find the index of the shallowest undersaturated value
lowest_undersaturated_idx = np.where(np.any(undersaturated_mask, axis=0), lowest_undersaturated_idx, np.nan) # set NaNs in place of zero (array above returns 0 where values are NaN)

# convert depth indicies to actual depths
depths = OmegaA_data['depth'].to_numpy() # array relating indices to depths
nan_mask = np.isnan(lowest_undersaturated_idx)
shallowest_undersaturated_depth = np.copy(lowest_undersaturated_idx)
shallowest_undersaturated_depth[~nan_mask] = depths[lowest_undersaturated_idx[~nan_mask].astype(int)]  # Convert indices to depth levels

# loop through months, count number of instances MMMLD is deeper than saturation horizon
avg_percent_undersat = 0
for i in range(0, 12):
    monthly_MLD_da_max = MLD_da_max[i, :, :].T
    nan_mask = ~np.isnan(monthly_MLD_da_max) & ~np.isnan(shallowest_undersaturated_depth)
    
    # calculate number of instances where undersaturated water is within the max monthly MLD
    num_undersat = np.sum((monthly_MLD_da_max[nan_mask] > shallowest_undersaturated_depth[nan_mask]))
    
    # calculate number of instances where this is not true
    num_supersat = np.sum((monthly_MLD_da_max[nan_mask] < shallowest_undersaturated_depth[nan_mask]))
    
    # calculate percentage of lat/lon points where the mixed layer is deeper than the saturation horizon
    percent_undersat = num_undersat / (num_undersat + num_supersat) * 100
    print(str(round(percent_undersat, 4)) + ' %')
    
    avg_percent_undersat += percent_undersat

print('annual average percent undersat: ' + str(round(avg_percent_undersat/12, 4)) + ' %')

# %% use KL divergence to determine which equations predict best (lower KL divergence = two datasets are closer)
kl_div = p1.kl_divergence(espers)
#kl_div.to_csv('kl_div.csv')

# %% calculate ensemble mean TA for each data point
espers = p1.ensemble_mean(espers)

# %% calculate ensemble mean A_T prediction minus eqn. 16 A_T prediction
espers['TA_Ensemble_-_16_LIR'] = espers.Ensemble_Mean_TA_LIR - espers.LIRtalk16
espers['TA_Ensemble_-_16_NN'] = espers.Ensemble_Mean_TA_NN - espers.NNtalk16

# ensemble mean with only equations 1-8
espers = p1.ensemble_mean_1_8(espers)

# ensemble mean with only equations 9-16
espers = p1.ensemble_mean_9_16(espers)

# %% trim GO-SHIP + associated cruises to pick out data points on the standard transect
trimmed = p1.trim_go_ship(espers, go_ship_cruise_nums_2023)
#del trimmed['SR04'] # to delete SR04 for testing
all_trimmed = pd.concat(trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
all_trimmed = all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

# %% run (or upload) MC simulation to create array of simulated G2talk values (by cruise offset)
#num_mc_runs = 1000
#G2talk_mc = p1.create_mc_cruise_offset(all_trimmed, num_mc_runs)
# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#G2talk_mc = pd.DataFrame(G2talk_mc)
#G2talk_mc.to_csv(filepath + input_mc_cruise_file, index=False)

G2talk_mc = pd.read_csv(filepath + input_mc_cruise_file, na_values = -9999)
G2talk_std = G2talk_mc.std(axis=1)
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)

# %% run (or upload) MC simulation to create array of simulated G2talk values (individual point offset)
#num_mc_runs = 1000
#G2talk_mc = p1.create_mc_individual_offset(all_trimmed, num_mc_runs)
# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#G2talk_mc = pd.DataFrame(G2talk_mc)
#G2talk_mc.to_csv(filepath + input_mc_individual_file, index=False)

G2talk_mc = pd.read_csv(filepath + input_mc_individual_file, na_values = -9999)
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)

#%% drop indian ocean data for testing
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)
trimmed_mc['I01'] = trimmed_mc['I01'][trimmed_mc['I01'].G2latitude < -60]
trimmed_mc['I03'] = trimmed_mc['I03'][trimmed_mc['I03'].G2latitude < -60]
trimmed_mc['I05'] = trimmed_mc['I05'][trimmed_mc['I05'].G2latitude < -60]
trimmed_mc['I06'] = trimmed_mc['I06'][trimmed_mc['I06'].G2latitude < -60]
trimmed_mc['I07'] = trimmed_mc['I07'][trimmed_mc['I07'].G2latitude < -60]
trimmed_mc['I08N'] = trimmed_mc['I08N'][trimmed_mc['I08N'].G2latitude < -60]
trimmed_mc['I08S'] = trimmed_mc['I08S'][trimmed_mc['I08S'].G2latitude < -60]
trimmed_mc['I09N'] = trimmed_mc['I09N'][trimmed_mc['I09N'].G2latitude < -60]
trimmed_mc['I09S'] = trimmed_mc['I09S'][trimmed_mc['I09S'].G2latitude < -60]
trimmed_mc['I10'] = trimmed_mc['I10'][trimmed_mc['I10'].G2latitude < -60]

trimmed['I01'] = trimmed['I01'][trimmed['I01'].G2latitude < -60]
trimmed['I03'] = trimmed['I03'][trimmed['I03'].G2latitude < -60]
trimmed['I05'] = trimmed['I05'][trimmed['I05'].G2latitude < -60]
trimmed['I06'] = trimmed['I06'][trimmed['I06'].G2latitude < -60]
trimmed['I07'] = trimmed['I07'][trimmed['I07'].G2latitude < -60]
trimmed['I08N'] = trimmed['I08N'][trimmed['I08N'].G2latitude < -60]
trimmed['I08S'] = trimmed['I08S'][trimmed['I08S'].G2latitude < -60]
trimmed['I09N'] = trimmed['I09N'][trimmed['I09N'].G2latitude < -60]
trimmed['I09S'] = trimmed['I09S'][trimmed['I09S'].G2latitude < -60]
trimmed['I10'] = trimmed['I10'][trimmed['I10'].G2latitude < -60]

all_trimmed_mc = pd.concat(trimmed_mc.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
all_trimmed_mc = all_trimmed_mc.drop_duplicates(ignore_index=True) # drop duplicates

all_trimmed = pd.concat(trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
all_trimmed = all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

#%% show layered histograms of distance from ensemble mean for each equation

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,4), dpi=200, sharex=True)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

hist_data = np.zeros((all_trimmed.shape[0], 16))

for i in range(0,16):
    hist_data[:,i] = all_trimmed['LIRtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_LIR']

# plot ESPER-LIR results
binwidth = 0.25
labels = ['Eqn. 1', 'Eqn. 2', 'Eqn. 3', 'Eqn. 4', 'Eqn. 5', 'Eqn. 6', 'Eqn. 7',
          'Eqn. 8', 'Eqn. 9', 'Eqn. 10', 'Eqn. 11', 'Eqn. 12', 'Eqn. 13',
          'Eqn. 14', 'Eqn. 15', 'Eqn. 16']
ax1.hist(hist_data, bins=np.arange(np.nanmin(hist_data.flatten()), np.nanmax(hist_data.flatten()) + binwidth, binwidth), histtype ='step', stacked=True,fill=True, label=labels)
ax1.set_xlim([-5,5])
ax1.set_ylim([0,600000])
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
fig.text(0.14, 0.825, 'A: ESPER_LIR', fontsize=11)

# plot ESPER NN results
hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['NNtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_NN']

ax2.hist(hist_data, bins=np.arange(np.nanmin(hist_data.flatten()), np.nanmax(hist_data.flatten()) + binwidth, binwidth), histtype ='step', stacked=True,fill=True, label=labels)
ax2.set_xlim([-5,5])
ax2.set_ylim([0,600000])
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

# label axis, set up legend
plt.ylabel('Number of $∆A_\mathrm{T}$ Calculations ($x$ $10^{5}$)')
plt.xlabel('ESPER-Estimated $A_\mathrm{T}$ - Ensemble Mean $A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
fig.text(0.14, 0.41, 'B: ESPER_NN', fontsize=11)

# remove "1e5" from axis bc included in axis label
fig.patches.extend([plt.Rectangle((0.125,0.883),0.1,0.031, fill=True, color='w',
                                  zorder=1000, transform=fig.transFigure, figure=fig)])

fig.patches.extend([plt.Rectangle((0.125,0.48),0.1,0.031, fill=True, color='w',
                                  zorder=1000, transform=fig.transFigure, figure=fig)])

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], bbox_to_anchor = (1.05, 2.35), loc='upper left')

# %% plot cruises colored by measurement density
# set up map
# atlantic-centered view
#fig = plt.figure(figsize=(6.2,4.1))
fig = plt.figure(figsize=(10,3.5), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
#ax = plt.axes(projection=ccrs.Orthographic(0,90)) # arctic-centered view (turn off "extent" variable)

#ax.set_title('North Atlantic Coverage of TA (GLODAPv2.2023)')
#extent = [5, 15, -52.5, -52]
#extent = [-30, 30, -80, 10]
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# keep only unique years
# make dataframe of lat, lon, dectime, round dectime to year
# round lat and lon to nearest 2º
round_to = 4
lat = all_trimmed.G2latitude
lat = (lat / round_to).round().astype(int) * round_to
lon = all_trimmed.G2longitude
lon = (lon / round_to).round().astype(int) * round_to
positions = pd.DataFrame({'lat' : lat, 'lon' : lon, 'dectime' : all_trimmed.dectime.round(0)})

# deal with some longitudes needing to be transformed
# if lon = -180, make it = 180
positions.lon[positions.lon == -180] = 180

# if lon > 180, subtract 360
positions.lon[positions.lon > 180] -= 360

# only keep one cruise per station (I think just drop repeated rows?)
positions.drop_duplicates(inplace=True)

# count number of years there is a lat/lon observation in a place
count_positions = positions[['lat', 'lon']].value_counts().reset_index(name='counts')

# plot all trimmed transects
lon = count_positions.lon
lat = count_positions.lat
counts = count_positions.counts

cropped_cmap = cmocean.tools.crop_by_percent(cmo.dense, 10, which='min', N=None)
base = plt.cm.get_cmap(cropped_cmap)
color_list = base(np.linspace(0, 1, 15))
cmap_name = base.name + str(15)
cmap = base.from_list(cmap_name, color_list, 15)

im = ax.scatter(lon,lat,c=counts, cmap=cmap, transform=ccrs.PlateCarree(), marker='o', edgecolors='none', s=15)
fig.colorbar(im, label='Number of Unique Years an\n$A_\mathrm{T}$ Measurement Was Made', pad=0.02)

#cmap = cmo.dense
#cmap.set_under('w',1)
#h = ax.hist2d(lon, lat, bins=50, cmap=cmap, vmin=1, transform=ccrs.PlateCarree(), zorder=10)
#fig.colorbar(h[3],label='Number of Unique Years a\nMeasurement Was Made', pad=0.02)

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)

# plot one cruise colored
#df = trimmed['A17']
#df = df.loc[df.G2cruise == 1020]
#lon = df.G2longitude
#lat = df.G2latitude
#plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='crimson')

#%% plot normal distribution and one cruise colored
mean = 0
std_dev = 2
highlight = 0.43 # mark with dot for presentation

x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = stats.norm.pdf(x, mean, std_dev)

plt.figure(figsize=(6,1.5), dpi=200)
plt.plot(x, y, label='Normal Distribution')
plt.plot(highlight, stats.norm.pdf(highlight, mean, std_dev), 'o', c='#c1375b', label=f'Value = {highlight}')

# Move axes to the center
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# make custom axis spines
plt.gca().spines['left'].set_position('zero')
plt.gca().spines['bottom'].set_position('zero')

# hide the right and top spines
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# hide tick marks
plt.gca().set_yticklabels([])

# move x axis tick marks down slightly
for tick in plt.gca().get_xticklabels():
    tick.set_y(-0.008)  # Adjust this value to control the distance

# plot cruises with one cruise colored
fig = plt.figure(figsize=(5,3), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)

# plot all cruises 
df = all_trimmed
lon = df.G2longitude
lat = df.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=5,color='steelblue')

# plot one cruise colored
df = all_trimmed
df = df.loc[df.G2cruise == 1030]
lon = df.G2longitude
lat = df.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=15,color='#c1375b')

# create map of MC run 2 as an example
offsets = all_trimmed_mc['G2talk'] - all_trimmed_mc['2']

fig = plt.figure(figsize=(5,4), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)

# plot all cruises 
df = all_trimmed
lon = df.G2longitude
lat = df.G2latitude

im = ax.scatter(lon,lat,c=offsets, cmap=cmo.balance, transform=ccrs.PlateCarree(), marker='o', edgecolors='none', s=5)
fig.colorbar(im, label='Offset', pad=0.05)

# plot trend in AT for MC run 2
plt.figure(figsize=(3.9,2.5), dpi=200)
ax = plt.gca()
all_trimmed_mc['2']
# sort by time
esper_sel = all_trimmed.sort_values(by=['dectime'],ascending=True)

# calculate the difference in TA betwen GLODAP and ESPERS, store for regression
del_alk = all_trimmed_mc['2'] - esper_sel.loc[:,'Ensemble_Mean_TA_LIR']
x = esper_sel['dectime'].to_numpy()
y = del_alk.to_numpy()

# fit model and print summary
x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

ols_model = sm.OLS(y, x_model)
ols_results = ols_model.fit()

h = ax.hist2d(x, y, bins=100, norm='log', cmap=cmo.matter) # for 2d histogram
ax.plot(x_model[:,1], ols_results.fittedvalues, lw=2.5, ls='--', color='gainsboro', label='OLS')
ax.set_ylim([-60, 60]) # for LIR-trained only

# print equations & p values for each regression type
ax.text(1992.5, 45, str('OLS: $m={:+.3f}$'.format(ols_results.params[1]) + str(' $µmol$ $kg^{-1}$ $yr^{-1}$')), fontsize=12) # for LIR-trained only
ax.set_xlim([1991.66753234399, 2021.75842656012])
ax.set_yticklabels([])
ax.set_xticklabels([])

#%% plot difference between measured and espers on a map

# set up figure
fig = plt.figure(figsize=(10,4), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get espers data, exclude points with difference <5 or >-5 µmol/kg
surface = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth]
#surface = all_trimmed[(all_trimmed.G2depth < 4000) & (all_trimmed.G2depth >3500)]
lat = surface.G2latitude
lon = surface.G2longitude
diff = surface.G2talk - surface.Ensemble_Mean_TA_LIR
to_plot = pd.DataFrame(data={'G2latitude' : lat, 'G2longitude' : lon, 'del_alk' : diff, 'abs_del_alk' : np.abs(diff)})
to_plot = to_plot[(to_plot.del_alk > -30) & (to_plot.del_alk < 30)]
to_plot = to_plot.sort_values('abs_del_alk',axis=0,ascending=True)

# create colormap
cmap = cmocean.tools.crop(cmo.balance, to_plot.del_alk.min(), to_plot.del_alk.max(), 0)

# plot data
pts = ax.scatter(to_plot.G2longitude,to_plot.G2latitude,transform=ccrs.PlateCarree(),s=30,c=to_plot.del_alk,cmap=cmap,edgecolors='none')
plt.colorbar(pts, ax=ax, label='$∆A_\mathrm{T}$ \n($µmol\;kg^{-1}$)')

per_data = 100*(1-len(to_plot)/len(diff))
print('% data removed to improve colorbar scale:', per_data, '%')

print('average error:', diff.mean(skipna=True),'$µmol kg^{-1}$')

#%% 2D histogram for global ensemble mean regression for all trimmed GO-SHIP
# with robust regression (statsmodels rlm)

winter = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] < 10))]
spring = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] < 10))]
summer = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] > 10))]
fall = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] > 10))]

# espers was trained on GLODAPv2.2020, new cruises by GLODAPv2.2023 are cruise number > 2106 & < 9999
data_not_used_for_espers =  all_trimmed.loc[((all_trimmed['G2cruise'] > 2106) & (all_trimmed['G2cruise'] < 9999))]

# looking at four A05 cruises that start in winter
A05_test = all_trimmed[(all_trimmed.G2cruise == 341) | (all_trimmed.G2cruise == 699) | (all_trimmed.G2cruise == 1030) | (all_trimmed.G2cruise == 1109)]

# looking at subarctic waters
subarctic = all_trimmed[(all_trimmed.G2latitude > 45) & (all_trimmed.G2latitude  < 85)]

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_LIR_9-16' # LIR, NN, or Mixed
#esper_type = 'LIRtalk16'
#esper_sel = all_trimmed
esper_sel = all_trimmed
#esper_sel = subarctic
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
#esper_sel = esper_sel[(esper_sel.G2depth > 200) & (esper_sel.G2depth < 2000)]
#esper_sel = esper_sel[esper_sel['Ensemble_Mean_TA_LIR_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,0], 'A', 0)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[0,0], 'Surface (< 25 m), LIR', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_LIR_9-16' # LIR, NN, or Mixed
#esper_type = 'LIRtalk16'
#esper_sel = all_trimmed # full depth
esper_sel = all_trimmed
#esper_sel = subarctic
#esper_sel = esper_sel[esper_sel['Ensemble_Mean_TA_LIR_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,0], 'C', 0)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[1,0], 'Full Depth, LIR', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_NN_9-16' # LIR, NN, or Mixed
#esper_type = 'NNtalk16'
#esper_sel = all_trimmed
esper_sel = all_trimmed
#esper_sel = subarctic
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
#esper_sel = esper_sel[(esper_sel.G2depth > 200) & (esper_sel.G2depth < 2000)]
#esper_sel = esper_sel[esper_sel['Ensemble_Mean_TA_NN_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,1], 'B', 1)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[0,1], 'Surface (< 25 m), NN', 1)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_NN_9-16' # LIR, NN, or Mixed
#esper_type = 'NNtalk16'
#esper_sel = all_trimmed # full depth
esper_sel = all_trimmed
#esper_sel = subarctic
#esper_sel = esper_sel[esper_sel['Ensemble_Mean_TA_NN_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,1], 'D', 1)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[1,1], 'Full Depth, NN', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% calculate metrics

# number of TA measurements
print('Number of TA measurements:', len(all_trimmed.G2talk))

# number of cruises
print('Number of cruises:', len(all_trimmed.G2cruise.unique()))

# average ∆TA with ESPER LIR (± standard deviation)
del_alk = all_trimmed.loc[:,'G2talk'] - all_trimmed.loc[:,'Ensemble_Mean_TA_LIR']
print('Average ∆TA with ESPER LIR (± standard deviation):', del_alk.mean(skipna=True), '±', del_alk.std(skipna=True))

# average ∆TA with ESPER NN (± standard deviation)
del_alk = all_trimmed.loc[:,'G2talk'] - all_trimmed.loc[:,'Ensemble_Mean_TA_NN']
print('Average ∆TA with ESPER NN (± standard deviation):', del_alk.mean(skipna=True), '±', del_alk.std(skipna=True))

#%% test MLR idea --> TA = a * T + b * S + c * oxygen + d * nitrate + e * silicate + f * time
clf = linear_model.LinearRegression()

no_nans = all_trimmed[['G2temperature', 'G2salinity', 'G2oxygen', 'G2nitrate', 'G2silicate', 'dectime', 'G2talk']]
no_nans = no_nans.dropna()

x = no_nans[['G2salinity', 'G2silicate', 'dectime']]
y = no_nans['G2talk']

clf.fit(x, y)

#print('TA = ' + str(round(clf.coef_[0],4)) + '*T + ' + str(round(clf.coef_[1], 4)) + '*S + ' + str(round(clf.coef_[2], 4)) + '*O2 + ' + str(round(clf.coef_[3], 4)) + '*N + ' + str(round(clf.coef_[4], 4)) + '*Si + ' + str(round(clf.coef_[5], 4)) + '*time + ' + str(round(clf.intercept_, 4))) 
#print('TA = ' + str(round(clf.coef_[0],4)) + '*T + ' + str(round(clf.coef_[1], 4)) + '*S + ' + str(round(clf.coef_[2], 4)) + '*O2 + ' + str(round(clf.coef_[3], 4)) + '*N + ' + str(round(clf.coef_[4], 4)) + '*time + ' + str(round(clf.intercept_, 4)))
#print('TA = ' + str(round(clf.coef_[0],4)) + '*T + ' + str(round(clf.coef_[1], 4)) + '*S + ' + str(round(clf.coef_[2], 4)) + '*O2 + ' + str(round(clf.coef_[3], 4)) + '*time + ' + str(round(clf.intercept_, 4))) 
#print('TA = ' + str(round(clf.coef_[0],4)) + '*T + ' + str(round(clf.coef_[1], 4)) + '*S + ' + str(round(clf.coef_[2], 4)) + '*time + ' + str(round(clf.intercept_, 4))) 
#print('TA = ' + str(round(clf.coef_[0], 4)) + '*S + ' + str(round(clf.coef_[1], 4)) + '*time + ' + str(round(clf.intercept_, 4))) 

print('TA = ' + str(round(clf.coef_[0], 4)) + '*S + ' + str(round(clf.coef_[1], 4)) + '*Si + ' + str(round(clf.coef_[2], 4)) + '*time + ' + str(round(clf.intercept_, 4))) 

#%% plot data by season
# set up map
# atlantic-centered view
#fig = plt.figure(figsize=(6.2,4.1))
fig = plt.figure(figsize=(10,3.5), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
#ax = plt.axes(projection=ccrs.Orthographic(0,90)) # arctic-centered view (turn off "extent" variable)
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

#im = ax.scatter(fall.G2longitude,fall.G2latitude,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='peru')
#ax.set_title('Fall')

#im = ax.scatter(winter.G2longitude,winter.G2latitude,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='cornflowerblue')
#ax.set_title('Winter')

#im = ax.scatter(spring.G2longitude,spring.G2latitude,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='pink')
#ax.set_title('Spring')

im = ax.scatter(summer.G2longitude,summer.G2latitude,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='mediumseagreen')
ax.set_title('Summer')

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)

#plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='crimson')

#%% 2D histogram with 1D data count overlay by season
# with robust regression (statsmodels rlm)

winter = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] < 10))]
spring = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] < 10))]
summer = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] > 10))]
fall = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] > 10))]

# make figure
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(13,6), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# winter surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = winter
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[0,0], 'A: Winter (< 25 m)', 0, 1)

# winter full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = winter
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[1,0], 'E: Winter', 0, 0)

# spring surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = spring
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[0,1], 'B: Spring (< 25 m)', 0, 1)

# spring full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = spring
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[1,1], 'F: Spring', 0, 0)

# summer surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = summer
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[0,2], 'C: Summer (< 25 m)', 0, 1)

# summer full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = summer
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[1,2], 'G: Summer', 0, 0)

# fall surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = fall
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[0,3], 'D: Autumn (< 25 m)', 1, 1)

# fall full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = fall
p1.plot2dhist_1d(esper_sel, esper_type, fig, axs[1,3], 'H: Autumn', 1, 0)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.22,-0.62) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.52,0.28)

# remove "1e3" and "1e4" from axis bc included in axis label
fig.patches.extend([plt.Rectangle((0.8,0.974),0.1,0.02, fill=True, color='w',
                                  zorder=1000, transform=fig.transFigure, figure=fig)])

fig.patches.extend([plt.Rectangle((0.8,0.485),0.1,0.02, fill=True, color='w',
                                  zorder=1000, transform=fig.transFigure, figure=fig)])

#%% (∆TA/S) 2D histogram for global ensemble mean regression for all trimmed GO-SHIP
# with robust regression (statsmodels rlm)

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_S(esper_sel, esper_type, fig, axs[0,0], 'A', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot2dhist_S(esper_sel, esper_type, fig, axs[1,0], 'C', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist_S(esper_sel, esper_type, fig, axs[0,1], 'B', 1)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot2dhist_S(esper_sel, esper_type, fig, axs[1,1], 'D', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% 2D histogram for global ensemble mean regression with only GLODAPv2.2023 points used
# with robust regression (statsmodels rlm)

# make figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5,2.7), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_sel = all_trimmed # full depth
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
#esper_sel = all_trimmed[(all_trimmed.G2depth > 1000) & (all_trimmed.G2depth < 2000)] # subsurface only
p1.plot2dhist(esper_sel, esper_type, fig, axs[0], 'A: Surface (< 25 m)', 1)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
#esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
#esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
#esper_sel = all_trimmed[(all_trimmed.G2depth > 1000) & (all_trimmed.G2depth < 2000)] # subsurface only
p1.plot2dhist(esper_sel, esper_type, fig, axs[1], 'B: Full Depth', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.25,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
###ax.set_ylabel('Measured $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.57,0.28)

#%% data points colored by weight assigned by robust regression (statsmodels rlm)
# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7,4.5), dpi=400, sharex=True, sharey=True)
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[0,0], 'A: ESPER_LIR (< 25 m)', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[1,0], 'C: ESPER_LIR', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[0,1], 'B: ESPER_NN (< 25 m)', 0)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
pts = p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[1,1], 'D: ESPER_NN', 0)

# adjust figure
ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.4,-0.1)
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)', labelpad=15)

# add single colorbar
fig.colorbar(pts, ax=axs.ravel().tolist(), label='Weight Assigned by RLM')

# %% loop through monte carlo simulation-produced G2talk to do global ensemble mean regression

# create seasons
winter_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed_mc['G2latitude'] > 10)) | ((all_trimmed_mc.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed_mc['G2latitude'] < 10))]
spring_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed_mc['G2latitude'] > 10)) | ((all_trimmed_mc.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed_mc['G2latitude'] < 10))]
summer_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed_mc['G2latitude'] < 10)) | ((all_trimmed_mc.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed_mc['G2latitude'] > 10))]
fall_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed_mc['G2latitude'] < 10)) | ((all_trimmed_mc.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed_mc['G2latitude'] > 10))]

data_not_used_for_espers_mc =  all_trimmed_mc.loc[((all_trimmed_mc['G2cruise'] > 2106) & (all_trimmed_mc['G2cruise'] < 9999))]

# plot surface values and do regular linear regression
mc_sel = all_trimmed_mc
#mc_sel = fall_mc
#mc_sel = mc_sel[mc_sel['Ensemble_Mean_TA_LIR_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#mc_sel_surf = mc_sel[mc_sel.G2depth < mc_sel.surface_depth]
mc_sel_surf = mc_sel[(mc_sel.G2depth > 200) & (mc_sel.G2depth < 2000)]
#mc_sel_surf = mc_sel[(mc_sel.G2depth > 1000) & (mc_sel.G2depth < 2000)]
x = mc_sel.dectime
x_surf = mc_sel_surf.dectime

# preallocate arrays for storing slope and p-values
slopes_surf_LIR = np.zeros(G2talk_mc.shape[1])
pvalues_surf_LIR = np.zeros(G2talk_mc.shape[1])

slopes_LIR = np.zeros(G2talk_mc.shape[1])
pvalues_LIR = np.zeros(G2talk_mc.shape[1])

slopes_surf_NN = np.zeros(G2talk_mc.shape[1])
pvalues_surf_NN = np.zeros(G2talk_mc.shape[1])

slopes_NN = np.zeros(G2talk_mc.shape[1])
pvalues_NN = np.zeros(G2talk_mc.shape[1])

for i in range(0,G2talk_mc.shape[1]): 
    y_surf_LIR = mc_sel_surf[str(i)] - mc_sel_surf.Ensemble_Mean_TA_LIR
    y_LIR = mc_sel[str(i)] - mc_sel.Ensemble_Mean_TA_LIR
    y_surf_NN = mc_sel_surf[str(i)] - mc_sel_surf.Ensemble_Mean_TA_NN
    y_NN = mc_sel[str(i)] - mc_sel.Ensemble_Mean_TA_NN
    #y_surf_LIR = mc_sel_surf[str(i)] - mc_sel_surf['Ensemble_Mean_TA_LIR_9-16']
    #y_LIR = mc_sel[str(i)] - mc_sel['Ensemble_Mean_TA_LIR_9-16']
    #y_surf_NN = mc_sel_surf[str(i)] - mc_sel_surf['Ensemble_Mean_TA_NN_9-16']
    #y_NN = mc_sel[str(i)] - mc_sel['Ensemble_Mean_TA_NN_9-16']
    
    slope_surf_LIR, intercept_surf_LIR, _, pvalue_surf_LIR, _ = stats.linregress(x_surf, y_surf_LIR, alternative='two-sided')
    slope_LIR, intercept_LIR, _, pvalue_LIR, _ = stats.linregress(x, y_LIR, alternative='two-sided')
    slope_surf_NN, intercept_surf_NN, _, pvalue_surf_NN, _ = stats.linregress(x_surf, y_surf_NN, alternative='two-sided')
    slope_NN, intercept_NN, _, pvalue_NN, _ = stats.linregress(x, y_NN, alternative='two-sided')
    
    slopes_surf_LIR[i] = slope_surf_LIR
    pvalues_surf_LIR[i] = pvalue_surf_LIR
    slopes_LIR[i] = slope_LIR
    pvalues_LIR[i] = pvalue_LIR
    slopes_surf_NN[i] = slope_surf_NN
    pvalues_surf_NN[i] = pvalue_surf_NN
    slopes_NN[i] = slope_NN
    pvalues_NN[i] = pvalue_NN

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,5), dpi=200, sharex=True, sharey=True)
fig.add_subplot(111,frameon=False)
fig.tight_layout(pad=1.5)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

axs[0,0].hist(slopes_surf_LIR, bins=100)
axs[0,0].set_xlim([-0.15, 0.15]) # per cruise offset
axs[0,0].set_xlim([-0.075, 0.075]) # individual offset
mu = slopes_surf_LIR.mean()
sigma = slopes_surf_LIR.std()
axs[0,0].text(-0.145, 31.5, 'A: ESPER_LIR (< 25 m)', fontsize=11) # per cruise offset
axs[0,0].text(-0.145, 23.5,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # per cruise offset
#axs[0,0].text(-0.071, 34.2, 'A: ESPER_LIR (< 25 m)', fontsize=11) # individual offset
#axs[0,0].text(-0.071, 26.2,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # individual offset

axs[1,0].hist(slopes_LIR, bins=100)
mu = slopes_LIR.mean()
sigma = slopes_LIR.std()
axs[1,0].text(-0.145, 31.5, 'C: ESPER_LIR', fontsize=11) # per cruise offset
axs[1,0].text(-0.145, 23.5,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # per cruise offset
#axs[1,0].text(-0.071, 34.2, 'C: ESPER_LIR', fontsize=11) # individual offset
#axs[1,0].text(-0.071, 26.2,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # individual offset

axs[0,1].hist(slopes_surf_NN, bins=100)
mu = slopes_surf_NN.mean()
sigma = slopes_surf_NN.std()
axs[0,1].text(-0.145, 31.5, 'B: ESPER_NN (< 25 m)', fontsize=11) # per cruise offset
axs[0,1].text(-0.145, 23.5,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # per cruise offset
#axs[0,1].text(-0.071, 34.2, 'B: ESPER_NN (< 25 m)', fontsize=11) # individual offset
#axs[0,1].text(-0.071, 26.2,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # individual offset

axs[1,1].hist(slopes_NN, bins=100)
mu = slopes_NN.mean()
sigma = slopes_NN.std()
axs[1,1].text(-0.145, 31.5, 'D: ESPER_NN', fontsize=11) # per cruise offset
axs[1,1].text(-0.145, 23.5,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # per cruise offset
#axs[1,1].text(-0.071, 34.2, 'D: ESPER_NN', fontsize=11) # individual offset
#axs[1,1].text(-0.071, 26.2,'$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=11) # individual offset

plt.xlabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol\;kg^{-1}\;yr^{-1}$)')
plt.ylabel('Number of Occurrences')

#%% print statistics
perc_pos = 100*(1 - len(slopes_surf_LIR[slopes_surf_LIR<0])/1000.0)
print('% of simulations with positive trend, surface ESPER LIR:', perc_pos, '%')

perc_pos = 100*(1 - len(slopes_surf_NN[slopes_surf_NN<0])/1000.0)
print('% of simulations with positive trend, surface ESPER NN:', perc_pos, '%')

perc_pos = 100*(1 - len(slopes_LIR[slopes_LIR<0])/1000.0)
print('% of simulations with positive trend, ESPER LIR:', perc_pos, '%')

perc_pos = 100*(1 - len(slopes_NN[slopes_NN<0])/1000.0)
print('% of simulations with positive trend, ESPER NN:', perc_pos, '%')

#%% just plot LIR surface and full depth

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,3), dpi=200, sharex=True, sharey=True)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

axs[0].hist(slopes_surf_LIR, bins=100)
axs[0].set_xlim([-0.15, 0.15]) # per cruise offset
axs[0].set_ylim([0, 50]) # per cruise offset
#axs[0].set_xlim([-0.075, 0.075]) # individual offset
mu = slopes_surf_LIR.mean()
sigma = slopes_surf_LIR.std()
axs[0].text(-0.14, 45, 'ESPER_LIR (< 25 m)', fontsize=14)
axs[0].text(-0.14, 35, '$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=12)

axs[1].hist(slopes_LIR, bins=100)
mu = slopes_LIR.mean()
sigma = slopes_LIR.std()
axs[1].text(-0.14, 45, 'ESPER_LIR (Full Depth)', fontsize=14)
axs[1].text(-0.14, 35, '$\mu={:.4f}$\n$\sigma={:.4f}$'.format(mu, sigma), fontsize=12)

plt.xlabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol\;kg^{-1}\;yr^{-1}$)')
plt.ylabel('Number of Occurrences')

# %% make box plot graph of transect slopes from mc simulation

# SURFACE LIR
# pull surface values
all_trimmed_mc_surf = all_trimmed_mc[all_trimmed_mc.G2depth < all_trimmed_mc.surface_depth]
# turn into dict with transects as keys
trimmed_mc_surf = p1.trim_go_ship(all_trimmed_mc_surf, go_ship_cruise_nums_2023)
all_slopes_surf = p1.transect_box_plot(trimmed_mc_surf, G2talk_mc, 'LIR')

# FULL-OCEAN LIR
# turn into dict with transects as keys
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)
all_slopes_full = p1.transect_box_plot(trimmed_mc, G2talk_mc, 'LIR')

# set up plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,5), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# make box plot for surface
axs[0].boxplot(all_slopes_surf, vert=True, labels=list(trimmed_mc.keys()))
axs[0].axhline(y=0, color='r', linestyle='--')
axs[0].set_ylim(-5, 5)
axs[0].text(1, 3.9, 'A: Surface (< 25 m)', fontsize=11)

# make box plot for full depth
axs[1].boxplot(all_slopes_full, vert=True, labels=list(trimmed_mc.keys()))
axs[1].axhline(y=0, color='r', linestyle='--')
axs[1].set_ylim(-5, 5)
axs[1].text(1, 3.9, 'B: Full Depth', fontsize=11)
axs[1].tick_params(axis='x', labelrotation=90)

ax.set_ylabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)')
#ax.set_ylabel('Slope of Measured $A_{T}$ over Time\n($µmol$ $kg^{-1}$ $yr^{-1}$)')

ax.yaxis.set_label_coords(-0.57,0.55)

#%% calculate error: error in TREND, not point
# u_esper = standard deviation in slope across all 16 equations
# u_sample= standard deviation in slopes predicted by mc analysis
# U = summation of u_esper and u_sample in quadrature

# doing this with robust regression

# calculate u_esper
esper_type = 'LIRtalk' # LIR, NN, or Mixed (change separately for u_sample below)
esper_sel = all_trimmed
#esper_sel = fall_mc
#esper_sel = esper_sel[esper_sel['Ensemble_Mean_TA_LIR_1-8'].notna()] # needed when doing ensemble mean that doesn't include eqn. 16
#esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
esper_sel = esper_sel[(esper_sel.G2depth > 200) & (esper_sel.G2depth < 2000)] # do surface values (< 25 m) only
slopes_rlm = np.zeros(16)
slopes_ols = np.zeros(16)
#slopes_rlm = np.zeros(8) # if doing equations 1-8 or 9-16
#slopes_ols = np.zeros(8) # if doing equations 1-8 or 9-16

for i in range(0,16):
#for i in range(8,16):

    # sort by time
    esper_sel = esper_sel.sort_values(by=['dectime'],ascending=True)

    # calculate the difference in TA betwen GLODAP and ESPERS, store for regression
    esper_sel = esper_sel.dropna(subset=['G2talk', esper_type+str(i+1)])
    del_alk = esper_sel.loc[:,'G2talk'] - esper_sel.loc[:,esper_type+str(i+1)]
    x = esper_sel['dectime'].to_numpy()
    y = del_alk.to_numpy()

    # fit model and print summary
    x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
    rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()

    ols_model = sm.OLS(y, x_model)
    ols_results = ols_model.fit()

    slopes_rlm[i] = rlm_results.params[1]
    slopes_ols[i] = ols_results.params[1]
    #slopes_rlm[i-8] = rlm_results.params[1] # if doing equations 9-16 only
    #slopes_ols[i-8] = ols_results.params[1] # if doing equations 9-16 only

u_esper = slopes_rlm.std() # change if RLM or OLS used for u_esper here

# calculate u_sample
u_sample = slopes_surf_LIR.std() # for SURFACE, LIR
#u_sample = slopes_LIR.std() # for FULL DEPTH, LIR
#u_sample = slopes_surf_NN.std() # for SURFACE, NN
#u_sample = slopes_NN.std() # for FULL DEPTH, NN

U = np.sqrt(u_esper**2 + u_sample**2)
print(round(U,3))

# %% plot global ensemble mean regression for each GO-SHIP transect

# plot surface values and do linear regression
slopes = np.zeros(len(trimmed.keys()))
pvalues = np.zeros(len(trimmed.keys()))
i = 0

for keys in trimmed:
    if len(trimmed[keys]) > 0: # if dictionary key is not empty 
        transect = trimmed[keys]
        surface = transect[transect.G2depth < transect.surface_depth]
        x = surface.dectime
        y = surface.G2talk - surface.Ensemble_Mean_TA
        
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
        slopes[i] = slope
        pvalues[i] = pvalue
        
        # make regression of surface values
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
        plt.scatter(surface.datetime,y,s=1)
        fig.text(0.6, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
        fig.text(0.6, 0.78, '$p-value={:.4e}$'.format(pvalue), fontsize=14)
        ax.plot(surface.datetime, intercept + slope * surface.dectime, color="r", lw=1);
        ax.set_title(str(keys) + ': Difference in Measured and ESPER-Predicted $A_{T}$ (< 25 m)')
        ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
        ax.set_ylim(-70,70)
        ax.set_xlim(all_trimmed.datetime.min(),all_trimmed.datetime.max())
    
        transect = trimmed[keys]
        x = transect.dectime
        y = transect.G2talk - transect.Ensemble_Mean_TA
    
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
    
        # make regression for full depth values
        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()
        plt.scatter(transect.datetime,y,s=1)
        fig.text(0.6, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
        fig.text(0.6, 0.78, '$p-value={:.4e}$'.format(pvalue), fontsize=14)
        ax.plot(transect.datetime, intercept + slope * transect.dectime, color="r", lw=1);
        ax.set_title(str(keys) + ': Difference in Measured and ESPER-Predicted $A_{T}$ (Full Depth)')
        ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
        ax.set_ylim(-70,70)
        ax.set_xlim(all_trimmed.datetime.min(),all_trimmed.datetime.max())
        i += 1
    else:
        slopes[i] = np.nan
        pvalues[i] = np.nan
        i += 1

#%% calculate average ESPERs coefficients

# read in coefficients extracted from MATLAB (already averaged across all 16 equations)
#coeffs = pd.read_csv(filepath + coeffs_file, names=['x', 'TA_S', 'TA_T', 'TA_N', 'TA_O', 'TA_Si'])
coeffs = pd.read_csv(filepath + 'ESPER_LIR_coeffs_eqn_16.csv', names=['x', 'TA_S', 'TA_T', 'TA_N', 'TA_O', 'TA_Si'])

# attach G2cruise, G2station, G2depth, and surface_depth columns so trimming will work
coeffs = pd.concat([coeffs, espers[['G2expocode', 'G2cruise', 'G2station',
                                    'G2region', 'G2cast', 'dectime',
                                    'datetime', 'G2latitude', 'G2longitude',
                                    'G2depth', 'surface_depth',
                                    'G2temperature', 'G2salinity', 'G2oxygen',
                                    'G2nitrate', 'G2silicate', 'G2phosphate',
                                    'G2talk']].copy()], axis=1)

# trim to pick out points on standard transect
coeffs_trimmed = p1.trim_go_ship(coeffs, go_ship_cruise_nums_2023)
coeffs_all_trimmed = pd.concat(coeffs_trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
coeffs_all_trimmed = coeffs_all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

# constrain to surface depth only
coeffs_all_trimmed = coeffs_all_trimmed[coeffs_all_trimmed.G2depth < coeffs_all_trimmed.surface_depth]

# average across all samples
coeffs_all_trimmed = coeffs_all_trimmed.drop(columns=['G2expocode', 'G2cruise',
                                                      'G2station', 'G2region',
                                                      'G2cast', 'dectime',
                                                      'datetime', 'G2latitude',
                                                      'G2longitude', 'G2depth',
                                                      'surface_depth',
                                                      'G2temperature',
                                                      'G2salinity', 'G2oxygen',
                                                      'G2nitrate',
                                                      'G2silicate',
                                                      'G2phosphate', 'G2talk'])
avg_coeffs = coeffs_all_trimmed.mean(axis=0)

# calculate ratio of TA to S in surface ocean (GLODAP data)
# surface values only
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth]

# divide G2talk by G2salinity
TA_S_GLODAP =  esper_sel.G2talk / esper_sel.G2salinity

# average across all samples
avg_TA_S_GLODAP= TA_S_GLODAP.mean(axis=0)

# per sample difference between glodap TA/S ratio and esper TA/S ratio
TA_S_diff = np.array(list(TA_S_GLODAP)) - np.array(list(coeffs_all_trimmed['TA_S']))
avg_TA_S_diff = TA_S_diff.mean(axis=0)

# plot differences bewteen glodap TA/S ratio and esper TA/S ratio on a map 
# set up map, atlantic-centered view
fig = plt.figure(figsize=(10,3.5), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

lat = esper_sel.G2latitude
lon = esper_sel.G2longitude

# deal with some longitudes needing to be transformed
# if lon = -180, make it = 180
lon[lon == -180] = 180

# if lon > 180, subtract 360
lon[lon > 180] -= 360

im = ax.scatter(lon,lat,c=TA_S_diff, cmap=cmo.balance, vmin=-40, vmax=40, transform=ccrs.PlateCarree(), marker='o', edgecolors='none', s=15)
fig.colorbar(im, label='Difference in TA/S ratios between\nGLODAP and ESPERs ($µmol kg^{-1} PSU^{-1}$)', pad=0.02)

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)

# %% compare results from each equation with equation 16
# only with LIR for now
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only

# preallocate array to store slopes & p-values
ols_slopes_LIR = np.zeros(15)
ols_pvalues_LIR = np.zeros(15)
rlm_slopes_LIR = np.zeros(15)
rlm_pvalues_LIR = np.zeros(15)

ols_slopes_NN = np.zeros(15)
ols_pvalues_NN = np.zeros(15)
rlm_slopes_NN = np.zeros(15)
rlm_pvalues_NN = np.zeros(15)

# calculate ∆ between each equation and equation 16
for i in range(1, 16):
    esper_LIR = 'LIRtalk' + str(i)
    esper_NN = 'NNtalk' + str(i)
    del_alk_LIR = esper_sel[esper_LIR] - esper_sel['LIRtalk16']
    del_alk_NN = esper_sel[esper_NN] - esper_sel['NNtalk16']

    x = esper_sel['dectime'].to_numpy()
    y_LIR = del_alk_LIR.to_numpy()
    y_NN = del_alk_NN.to_numpy()
    
    # get rid of NaN values
    not_nan_idx_LIR = np.where(~np.isnan(y_LIR))
    x_LIR = x[not_nan_idx_LIR]
    y_LIR = y_LIR[not_nan_idx_LIR]
    
    not_nan_idx_NN = np.where(~np.isnan(y_NN))
    x_NN = x[not_nan_idx_NN]
    y_NN = y_NN[not_nan_idx_NN]
    
    # calculate trend with ols/rlm for each ∆equation instead of ensemble mean
    # fit model and print summary
    x_model_LIR = sm.add_constant(x_LIR) # this is required in statsmodels to get an intercept
    rlm_model_LIR = sm.RLM(y_LIR, x_model_LIR, M=sm.robust.norms.HuberT())
    rlm_results_LIR = rlm_model_LIR.fit()
    
    x_model_NN = sm.add_constant(x_NN) # this is required in statsmodels to get an intercept
    rlm_model_NN = sm.RLM(y_NN, x_model_NN, M=sm.robust.norms.HuberT())
    rlm_results_NN = rlm_model_NN.fit()

    ols_model_LIR = sm.OLS(y_LIR, x_model_LIR)
    ols_results_LIR = ols_model_LIR.fit()
    
    ols_model_NN = sm.OLS(y_NN, x_model_NN)
    ols_results_NN = ols_model_NN.fit()
    
    # store slope and p-value for each equation
    ols_slopes_LIR[i-1] = ols_results_LIR.params[1]
    ols_pvalues_LIR[i-1] = ols_results_LIR.pvalues[1]
    rlm_slopes_LIR[i-1] = rlm_results_LIR.params[1]
    rlm_pvalues_LIR[i-1] = rlm_results_LIR.pvalues[1]
    
    ols_slopes_NN[i-1] = ols_results_NN.params[1]
    ols_pvalues_NN[i-1] = ols_results_NN.pvalues[1]
    rlm_slopes_NN[i-1] = rlm_results_NN.params[1]
    rlm_pvalues_NN[i-1] = rlm_results_NN.pvalues[1]
    
# plot slopes and pvalues (scatter plot?)
fig, axs = plt.subplots(2, 1, figsize=(7,4), dpi=200, sharex=True)

# break into significant slopes and nonsignificant slopes
ols_sig_LIR = np.copy(ols_slopes_LIR)
ols_sig_LIR[ols_pvalues_LIR >= 0.05] = np.nan

ols_notsig_LIR = np.copy(ols_slopes_LIR)
ols_notsig_LIR[ols_pvalues_LIR < 0.05] = np.nan

rlm_sig_LIR = np.copy(rlm_slopes_LIR)
rlm_sig_LIR[rlm_pvalues_LIR >= 0.05] = np.nan

rlm_notsig_LIR = np.copy(rlm_slopes_LIR)
rlm_notsig_LIR[rlm_pvalues_LIR < 0.05] = np.nan

ols_sig_NN = np.copy(ols_slopes_NN)
ols_sig_NN[ols_pvalues_NN >= 0.05] = np.nan

ols_notsig_NN = np.copy(ols_slopes_NN)
ols_notsig_NN[ols_pvalues_NN < 0.05] = np.nan

rlm_sig_NN = np.copy(rlm_slopes_NN)
rlm_sig_NN[rlm_pvalues_NN >= 0.05] = np.nan

rlm_notsig_NN = np.copy(rlm_slopes_NN)
rlm_notsig_NN[rlm_pvalues_NN < 0.05] = np.nan

axs[0].scatter(range(1,16), ols_sig_LIR, c='deepskyblue', marker='o', label='OLS (Significant)')
axs[0].scatter(range(1,16), ols_notsig_LIR, facecolors='none', marker='o', edgecolors='deepskyblue', label='OLS (Not Significant)')
axs[0].scatter(range(1,16), rlm_sig_LIR, c='orchid', marker='o', label='RLM (Significant)')
axs[0].scatter(range(1,16), rlm_notsig_LIR, facecolors='none', marker='o', edgecolors='orchid', label='RLM (Not Significant)')

axs[1].scatter(range(1,16), ols_sig_NN, c='deepskyblue', marker='o', label='OLS (Significant)')
axs[1].scatter(range(1,16), ols_notsig_NN, facecolors='none', marker='o', edgecolors='deepskyblue', label='OLS (Not Significant)')
axs[1].scatter(range(1,16), rlm_sig_NN, c='orchid', marker='o', label='RLM (Significant)')
axs[1].scatter(range(1,16), rlm_notsig_NN, facecolors='none', marker='o', edgecolors='orchid', label='RLM (Not Significant)')

axs[0].axhline(y=0.0, ls='--', c='k')
axs[1].axhline(y=0.0, ls='--', c='k')

axs[1].set_xlabel('Equation Number')
axs[1].set_ylabel('Difference between $A_\mathrm{T}$ Predicted\nby Each Eqn. and Eqn. 16 over Time\n($µmol\;kg^{-1}\;yr^{-1}$)')
axs[1].yaxis.set_label_coords(-0.1,1.1)

axs[0].text(0.45, -0.024, 'A: ESPER_LIR', fontsize=11)
axs[1].text(0.45, -0.024, 'B: ESPER_NN', fontsize=11)

axs[0].set_ylim([-0.027, 0.027])
axs[1].set_ylim([-0.027, 0.027])

axs[1].legend(bbox_to_anchor = (0.88, -0.35), ncol=2)

plt.xticks(range(1,16))

# %% compare ∆TA with salinity, temperature, nutrients, and E-P cycles

var_name = 'G2nitrate'
esper_type = 'Ensemble_Mean_TA_LIR'

# make figure
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9.5,6), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface ocean, full timeseries
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values only
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,0], 'Surface-only', 1, 100, 0)

# surface ocean, pre-2005
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values only
esper_sel = esper_sel[esper_sel.dectime <= 2005] # before 2005
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,1], 'Surface-only, 1991-2005', 1, 100, 0)

# surface ocean, post-2005
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values only
esper_sel = esper_sel[esper_sel.dectime > 2005] # after 2005
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,2], 'Surface-only, 2005-2021', 1, 100, 1)

# full ocean, full timeseries
esper_sel = all_trimmed
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,0], 'Full depth', 1, 100, 0)

# full ocean, pre-2005
esper_sel = all_trimmed[all_trimmed.dectime <= 2005] # before 2005
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,1], 'Full depth, 1991-2005', 1, 100, 0)

# full ocean, post-2005
esper_sel = all_trimmed[all_trimmed.dectime > 2005] # after 2005
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,2], 'Full depth, 2005-2021', 1, 100, 1)

#ax.set_xlabel('Salinity (PSU)')
#ax.set_xlabel('Temperature (ºC)')
ax.set_xlabel('Nitrate ($µmol\;kg^{-1}$)')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

# %% regional analysis
# North Atlantic: A02, A05, A16N, A20, A22, A25, AR07E, AR07W, A17 (above 0º latitude)
# South Atlantic: A10, A12, A135, A16S, A17 (below 0º and above -60º latitude)
# North Pacific: P01, P02, P03, P09, P10, P13, P14, P16N, P18 (above 0º latitude)
# South Pacific: SR03, P13, P14, P15, P16N, P16S, P18, P21 (below 0º and above -60º latitude)
# Indian Ocean: I01, I03, I05, I06S, I07, I08N, I08S, I09N, I09S, I10 (above -60º latitude)
# Southern Ocean: A12, A135, I06S, I07, I08S, I09S, P14, P15, P16S, P18, S04I, S04P, SR01, SR03, SR04 (below -60º latitude)
# Arctic Ocean: ARC01E, ARC01W, A29

# north atlantic
north_atlantic = {key: value for key, value in trimmed.items() if key in {'A02', 'A05', 'A16N', 'A20', 'A22', 'A25', 'AR07E', 'AR07W', 'A17'}}
north_atlantic = pd.concat(north_atlantic.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
north_atlantic = north_atlantic.drop_duplicates(ignore_index=True) # drop duplicates
north_atlantic = north_atlantic[north_atlantic['G2latitude'] > 0] # keep only above 0º latitude

# south atlantic
south_atlantic = {key: value for key, value in trimmed.items() if key in {'A10', 'A12', 'A135', 'A16S', 'A17'}}
south_atlantic = pd.concat(south_atlantic.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
south_atlantic = south_atlantic.drop_duplicates(ignore_index=True) # drop duplicates
south_atlantic = south_atlantic[south_atlantic['G2latitude'] > -60] # keep only above -60º latitude
south_atlantic = south_atlantic[south_atlantic['G2latitude'] <= 0] # keep only below 0º latitude

# north pacific
north_pacific = {key: value for key, value in trimmed.items() if key in {'P01', 'P02', 'P03', 'P09', 'P10', 'P13', 'P14', 'P16N', 'P18'}}
north_pacific = pd.concat(north_pacific.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
north_pacific = north_pacific.drop_duplicates(ignore_index=True) # drop duplicates
north_pacific = north_pacific[north_pacific['G2latitude'] > 0] # keep only above 0º latitude

# south pacific
south_pacific = {key: value for key, value in trimmed.items() if key in {'SR03', 'P13', 'P14', 'P15', 'P16N', 'P16S', 'P18', 'P21'}}
south_pacific = pd.concat(south_pacific.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
south_pacific = south_pacific.drop_duplicates(ignore_index=True) # drop duplicates
south_pacific = south_pacific[south_pacific['G2latitude'] > -60] # keep only above -60º latitude
south_pacific = south_pacific[south_pacific['G2latitude'] <= 0] # keep only below 0º latitude

# indian ocean
indian = {key: value for key, value in trimmed.items() if key in {'I01', 'I03', 'I05', 'I06', 'I07', 'I08N', 'I08S', 'I09N', 'I09S', 'I10'}}
indian = pd.concat(indian.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
indian = indian.drop_duplicates(ignore_index=True) # drop duplicates
indian = indian[indian['G2latitude'] > -60] # keep only above -60º latitude

# southern ocean
southern = {key: value for key, value in trimmed.items() if key in {'A12', 'A135', 'I06', 'I07', 'I08S', 'I09S', 'P14', 'P15', 'P16S', 'P18', 'S04I', 'S04P', 'SR01', 'SR03', 'SR04'}}
southern = pd.concat(southern.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
southern = southern.drop_duplicates(ignore_index=True) # drop duplicates
southern = southern[southern['G2latitude'] <= -60] # keep only below -60º latitude

# arctic ocean
arctic = {key: value for key, value in trimmed.items() if key in {'ARC01E', 'ARC01W', 'A29'}}
arctic = pd.concat(arctic.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
arctic = arctic.drop_duplicates(ignore_index=True) # drop duplicates

# %% plot regions on a map colored separately
# set up map
fig = plt.figure(figsize=(10,3.5), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree()) # atlantic-centered view
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
#ax = plt.axes(projection=ccrs.Orthographic(0,90)) # arctic-centered view (turn off "extent" variable)

extent = [-180, 180, -90, 90]
ax.set_extent(extent)

basins = [north_atlantic, south_atlantic, north_pacific, south_pacific, indian, southern, arctic]
basin_names = ['North Atlantic', 'South Atlantic', 'North Pacific', 'South Pacific', 'Indian', 'Southern', 'Arctic']

for basin, name in zip(basins, basin_names):
    # deal with some longitudes needing to be transformed
    # if lon = -180, make it = 180
    basin.G2longitude[basin.G2longitude == -180] = 180
    # if lon > 180, subtract 360
    basin.G2longitude[basin.G2longitude > 180] -= 360
    
    # round lat and lon to nearest 2º to make map prettier
    round_to = 4
    lat = basin.G2latitude
    lat = (lat / round_to).round().astype(int) * round_to
    lon = basin.G2longitude
    lon = (lon / round_to).round().astype(int) * round_to
    
    ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=15,label=name)

ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,alpha=0)
g1.bottom_labels = True
g1.left_labels = True
ax.add_feature(cfeature.LAND,color='k', zorder=12)
ax.legend(bbox_to_anchor = (1, -0.1), ncol=4)

# %% plot 2D histogram ∆TA trends for surface and deep for each region
# with robust regression (statsmodels rlm)

basin = north_pacific
#basin = pd.concat([north_pacific, south_pacific])

# make figure
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.5,2.7), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
basin_sel = basin[basin.G2depth < basin.surface_depth] # do surface values (< 25 m) only
#basin_sel = basin[basin.G2latitude > 45]
#basin_sel = basin_sel[basin_sel.G2depth < 1000] 
#basin_sel = basin_sel[basin_sel.dectime >= 2010]
p1.plot2dhist(basin_sel, esper_type, fig, axs[0], 'Surface (< 25 m)', 1)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
basin_sel = basin # full depth
#basin_sel = basin_sel[basin_sel.dectime >= 2010]
p1.plot2dhist(basin_sel, esper_type, fig, axs[1], 'Full Depth', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.25,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$\n($µmol\;kg^{-1}$)')
###ax.set_ylabel('Measured $A_{T}$($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% plot number of measurements in each region over time
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), dpi=200, sharex=True, sharey=True, layout='constrained')

basins = [north_atlantic, south_atlantic, north_pacific, south_pacific, indian, southern, arctic]
basin_names = ['North Atlantic', 'South Atlantic', 'North Pacific', 'South Pacific', 'Indian', 'Southern', 'Arctic']

for basin, name in zip(basins, basin_names):
    grouped = basin.groupby('datetime').count()
    grouped_year = grouped.groupby(pd.Grouper(freq='Y')).sum()
    ax.plot(grouped_year.G2talk,label=name)

ax.legend(bbox_to_anchor = (0.92, -0.1), ncol=4)
ax.set_ylabel('Number of $A_{T}$ Measurements')

#%% do each season over time too
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), dpi=200, sharex=True, sharey=True, layout='constrained')

winter = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] < 10))]
spring = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] < 10))]
summer = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] > 10))]
fall = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] > 10))]

seasons = [winter, spring, summer, fall]
season_names = ['Winter', 'Spring', 'Summer', 'Fall']

for season, name in zip(seasons, season_names):
    grouped = season.groupby('datetime').count()
    grouped_year = grouped.groupby(pd.Grouper(freq='Y')).sum()
    ax.plot(grouped_year.G2talk,label=name)

ax.legend(bbox_to_anchor = (0.8, -0.1), ncol=4)
ax.set_ylabel('Number of $A_{T}$ Measurements')

#%% look at alkalinity in each region over time
basins = [north_atlantic, south_atlantic, north_pacific, south_pacific, indian, southern, arctic]
basin_names = ['North Atlantic', 'South Atlantic', 'North Pacific', 'South Pacific', 'Indian', 'Southern', 'Arctic']

# calculate trend in AT over time for each region
#ax.scatter(north_atlantic.dectime,north_atlantic.G2talk)

for basin, name in zip(basins, basin_names):

    x = basin.dectime
    y = basin.G2talk
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), dpi=200, sharex=True, sharey=True, layout='constrained')
    ax.scatter(x,y)
    ax.set_ylabel('$A_{T}$ ($µmol\;kg^{-1}$)')
    ax.set_title(name)
    
    x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
    rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    ols_model = sm.OLS(y, x_model)
    ols_results = ols_model.fit()
    
    ax.plot(x_model.iloc[:,1], rlm_results.fittedvalues, ls='-', color='black', label='RLM')
    
    print(name)
    print('ols slope, p value: ' + str(ols_results.params.iloc[1]) + ', ' + str(ols_results.pvalues.iloc[1]))
    print('rlm slope, p value: ' + str(rlm_results.params.iloc[1]) + ', ' + str(rlm_results.pvalues.iloc[1]))
    print('')
# %% run (or upload) MC simulation FOR EACH REGION to create array of simulated G2talk values (by cruise offset)
#num_mc_runs = 1000
#basins = [north_atlantic, south_atlantic, north_pacific, south_pacific, indian, southern, arctic]
#basin_abbr = ['NA', 'SA', 'NP', 'SP', 'IO', 'SO', 'AO']
#for basin, abbr in zip(basins, basin_abbr):
#    print(abbr)
#    G2talk_mc = p1.create_mc_cruise_offset(basin, num_mc_runs)
#    # export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#    G2talk_mc = pd.DataFrame(G2talk_mc)
#    G2talk_mc.to_csv(filepath + 'G2talk_mc_simulated_' + abbr + '.csv' , index=False)

G2talk_mc_NA = pd.read_csv(filepath + 'G2talk_mc_simulated_NA.csv', na_values = -9999)
G2talk_std_NA = G2talk_mc_NA.std(axis=1)

G2talk_mc_SA = pd.read_csv(filepath + 'G2talk_mc_simulated_SA.csv', na_values = -9999)
G2talk_std_SA = G2talk_mc_SA.std(axis=1)

G2talk_mc_NP = pd.read_csv(filepath + 'G2talk_mc_simulated_NP.csv', na_values = -9999)
G2talk_std_NP = G2talk_mc_NP.std(axis=1)

G2talk_mc_SP = pd.read_csv(filepath + 'G2talk_mc_simulated_SP.csv', na_values = -9999)
G2talk_std_SP = G2talk_mc_SP.std(axis=1)

G2talk_mc_IO = pd.read_csv(filepath + 'G2talk_mc_simulated_IO.csv', na_values = -9999)
G2talk_std_IO = G2talk_mc_IO.std(axis=1)

G2talk_mc_SO = pd.read_csv(filepath + 'G2talk_mc_simulated_SO.csv', na_values = -9999)
G2talk_std_SO = G2talk_mc_SO.std(axis=1)

G2talk_mc_AO = pd.read_csv(filepath + 'G2talk_mc_simulated_AO.csv', na_values = -9999)
G2talk_std_AO = G2talk_mc_AO.std(axis=1)

G2talk_mc_regions = [G2talk_mc_NA, G2talk_mc_SA, G2talk_mc_NP, G2talk_mc_SP, G2talk_mc_IO, G2talk_mc_SO, G2talk_mc_AO]
G2talk_std_regions = [G2talk_std_NA, G2talk_std_SA, G2talk_std_NP, G2talk_std_SP, G2talk_std_IO, G2talk_std_SO, G2talk_std_AO]

# %% loop through monte carlo simulation-produced G2talk to do global ensemble mean regression
all_trimmed_basin = north_pacific
G2talk_mc_basin = G2talk_mc_NP

# plot surface values and do regular linear regression
all_trimmed_basin.reset_index(drop=True, inplace=True)
G2talk_mc_basin.reset_index(drop=True, inplace=True)
all_trimmed_mc_basin = pd.concat([all_trimmed_basin, G2talk_mc_basin], axis=1)
#all_trimmed_mc_basin_surf = all_trimmed_mc_basin[all_trimmed_mc_basin.G2depth < all_trimmed_mc_basin.surface_depth]
all_trimmed_mc_basin = all_trimmed_mc_basin[all_trimmed_mc_basin.G2latitude > 45]
all_trimmed_mc_basin_surf = all_trimmed_mc_basin[all_trimmed_mc_basin.G2depth < 1000]

x = all_trimmed_mc_basin.dectime
x_surf = all_trimmed_mc_basin_surf.dectime

# preallocate arrays for storing slope and p-values
slopes_surf_LIR = np.zeros(G2talk_mc_basin.shape[1])
pvalues_surf_LIR = np.zeros(G2talk_mc_basin.shape[1])

slopes_LIR = np.zeros(G2talk_mc_basin.shape[1])
pvalues_LIR = np.zeros(G2talk_mc_basin.shape[1])

slopes_surf_NN = np.zeros(G2talk_mc_basin.shape[1])
pvalues_surf_NN = np.zeros(G2talk_mc_basin.shape[1])

slopes_NN = np.zeros(G2talk_mc_basin.shape[1])
pvalues_NN = np.zeros(G2talk_mc_basin.shape[1])

for i in range(0,G2talk_mc_basin.shape[1]): 
    y_surf_LIR = all_trimmed_mc_basin_surf[str(i)] - all_trimmed_mc_basin_surf.Ensemble_Mean_TA_LIR
    y_LIR = all_trimmed_mc_basin[str(i)] - all_trimmed_mc_basin.Ensemble_Mean_TA_LIR
    y_surf_NN = all_trimmed_mc_basin_surf[str(i)] - all_trimmed_mc_basin_surf.Ensemble_Mean_TA_NN
    y_NN = all_trimmed_mc_basin[str(i)] - all_trimmed_mc_basin.Ensemble_Mean_TA_NN
    
    slope_surf_LIR, intercept_surf_LIR, _, pvalue_surf_LIR, _ = stats.linregress(x_surf, y_surf_LIR, alternative='two-sided')
    slope_LIR, intercept_LIR, _, pvalue_LIR, _ = stats.linregress(x, y_LIR, alternative='two-sided')
    slope_surf_NN, intercept_surf_NN, _, pvalue_surf_NN, _ = stats.linregress(x_surf, y_surf_NN, alternative='two-sided')
    slope_NN, intercept_NN, _, pvalue_NN, _ = stats.linregress(x, y_NN, alternative='two-sided')
    
    slopes_surf_LIR[i] = slope_surf_LIR
    pvalues_surf_LIR[i] = pvalue_surf_LIR
    slopes_LIR[i] = slope_LIR
    pvalues_LIR[i] = pvalue_LIR
    slopes_surf_NN[i] = slope_surf_NN
    pvalues_surf_NN[i] = pvalue_surf_NN
    slopes_NN[i] = slope_NN
    pvalues_NN[i] = pvalue_NN

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,5), dpi=200, sharex=True, sharey=True)
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

axs[0,0].hist(slopes_surf_LIR, bins=100)
axs[0,0].set_xlim([-0.15, 0.15]) # per cruise offset
#axs[0,0].set_xlim([-0.07, 0.07]) # individual offset
mu = slopes_surf_LIR.mean()
sigma = slopes_surf_LIR.std()
fig.text(0.14, 0.825, 'A', fontsize=14)
fig.text(0.236, 0.84, '$\mu={:.4f}, \sigma={:.4f}$'.format(mu, sigma), fontsize=12)

axs[1,0].hist(slopes_LIR, bins=100)
mu = slopes_LIR.mean()
sigma = slopes_LIR.std()
fig.text(0.14, 0.415, 'C', fontsize=14)
fig.text(0.236, 0.43, '$\mu={:.4f}, \sigma={:.4f}$'.format(mu, sigma), fontsize=12)

axs[0,1].hist(slopes_surf_NN, bins=100)
mu = slopes_surf_NN.mean()
sigma = slopes_surf_NN.std()
fig.text(0.56, 0.825, 'B', fontsize=14)
fig.text(0.661, 0.84, '$\mu={:.4f}, \sigma={:.4f}$'.format(mu, sigma), fontsize=12)

axs[1,1].hist(slopes_NN, bins=100)
mu = slopes_NN.mean()
sigma = slopes_NN.std()
fig.text(0.56, 0.415, 'D', fontsize=14)
fig.text(0.661, 0.43, '$\mu={:.4f}, \sigma={:.4f}$'.format(mu, sigma), fontsize=12)

plt.xlabel('Slope of Measured $A_{T}$ - ESPER-Estimated $A_{T}$ over Time ($µmol\;kg^{-1}\;yr^{-1}$)')
plt.ylabel('Number of Occurrences')

#%% calculate error: error in TREND, not point
# u_esper = standard deviation in slope across all 16 equations
# u_sample= standard deviation in slopes predicted by mc analysis
# U = summation of u_esper and u_sample in quadrature
# CALCULATES UNCERTANTIES FOR EACH BASIN

df_empty = pd.DataFrame()
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)

north_atlantic.reset_index(drop=True, inplace=True)
G2talk_mc_NA.reset_index(drop=True, inplace=True)

south_atlantic.reset_index(drop=True, inplace=True)
G2talk_mc_SA.reset_index(drop=True, inplace=True)

north_pacific.reset_index(drop=True, inplace=True)
G2talk_mc_NP.reset_index(drop=True, inplace=True)

south_pacific.reset_index(drop=True, inplace=True)
G2talk_mc_SP.reset_index(drop=True, inplace=True)

indian.reset_index(drop=True, inplace=True)
G2talk_mc_IO.reset_index(drop=True, inplace=True)

southern.reset_index(drop=True, inplace=True)
G2talk_mc_SO.reset_index(drop=True, inplace=True)

arctic.reset_index(drop=True, inplace=True)
G2talk_mc_AO.reset_index(drop=True, inplace=True)

basins = [all_trimmed_mc, north_atlantic, south_atlantic, north_pacific,
          south_pacific, indian, southern, arctic, trimmed_mc['A02'],
          trimmed_mc['A05'], trimmed_mc['A10'], trimmed_mc['A12'],
          trimmed_mc['A135'], trimmed_mc['A16N'], trimmed_mc['A16S'],
          trimmed_mc['A17'], trimmed_mc['A20'], trimmed_mc['A22'],
          trimmed_mc['A25'], trimmed_mc['A29'], trimmed_mc['AR07E'],
          trimmed_mc['AR07W'], trimmed_mc['ARC01E'], trimmed_mc['I03'],
          trimmed_mc['I05'], trimmed_mc['I06'], trimmed_mc['I07'],
          trimmed_mc['I08N'], trimmed_mc['I08S'], trimmed_mc['I09N'],
          trimmed_mc['I09S'], trimmed_mc['I10'], trimmed_mc['P01'],
          trimmed_mc['P02'], trimmed_mc['P03'], trimmed_mc['P06'],
          trimmed_mc['P09'], trimmed_mc['P10'], trimmed_mc['P13'],
          trimmed_mc['P14'], trimmed_mc['P15'], trimmed_mc['P16N'],
          trimmed_mc['P16S'], trimmed_mc['P17N'], trimmed_mc['P18'],
          trimmed_mc['P21'], trimmed_mc['S04I'], trimmed_mc['SR04'],
          trimmed_mc['S04P'], trimmed_mc['SR01'], trimmed_mc['SR03']]


mc_basins = [df_empty, G2talk_mc_NA, G2talk_mc_SA, G2talk_mc_NP, G2talk_mc_SP,
             G2talk_mc_IO, G2talk_mc_SO, G2talk_mc_AO, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty]

# test out subarctic north pacific
#north_pacific.reset_index(drop=True, inplace=True)
#G2talk_mc_NP.reset_index(drop=True, inplace=True)
#all_trimmed_mc_NP = pd.concat([all_trimmed_basin, G2talk_mc_basin], axis=1)
#subarctic_north_pacific_mc = all_trimmed_mc_NP[all_trimmed_mc_NP.G2latitude > 45]
#basins = [subarctic_north_pacific_mc.iloc[:,0:124]]
#mc_basins = [subarctic_north_pacific_mc.iloc[:,-1000:]]

basin_U_surf_LIR = np.zeros(len(basins))
basin_U_surf_NN = np.zeros(len(basins))
basin_U_LIR = np.zeros(len(basins))
basin_U_NN = np.zeros(len(basins))

u_esper_LIR = np.zeros(len(basins))
u_esper_NN = np.zeros(len(basins))
u_esper_surf_LIR = np.zeros(len(basins))
u_esper_surf_NN = np.zeros(len(basins))

u_sample_surf_LIR = np.zeros(len(basins))
u_sample_LIR = np.zeros(len(basins))
u_sample_surf_NN = np.zeros(len(basins))
u_sample_NN = np.zeros(len(basins))

for basin, mc_basin, j in zip(basins, mc_basins, range(0,len(basins))):
    
    # calculate u_esper
    esper = basin
    esper_surf = basin[basin.G2depth < basin.surface_depth] # do surface values (< 25 m) only
    #esper_surf = basin[basin.G2depth < 1000]
    
    slopes_rlm_LIR = np.zeros(16)
    slopes_rlm_NN = np.zeros(16)
    slopes_rlm_surf_LIR = np.zeros(16)
    slopes_rlm_surf_NN = np.zeros(16)
    
    slopes_ols_LIR = np.zeros(16)
    slopes_ols_NN = np.zeros(16)
    slopes_ols_surf_LIR = np.zeros(16)
    slopes_ols_surf_NN = np.zeros(16)
    
    for i in range(0,16):
    
        # sort by time
        esper = esper.sort_values(by=['dectime'],ascending=True)
        esper_surf = esper_surf.sort_values(by=['dectime'],ascending=True)
    
        # calculate the difference in TA betwen GLODAP and ESPERS, store for regression
        esper_LIR = esper.dropna(subset=['G2talk', 'LIRtalk' + str(i+1)])
        esper_NN = esper.dropna(subset=['G2talk', 'NNtalk' + str(i+1)])
        esper_surf_LIR = esper_surf.dropna(subset=['G2talk', 'LIRtalk' + str(i+1)])
        esper_surf_NN= esper_surf.dropna(subset=['G2talk', 'NNtalk' + str(i+1)])
        
        del_alk_LIR = esper_LIR.loc[:,'G2talk'] - esper_LIR.loc[:,'LIRtalk' + str(i+1)]
        del_alk_NN = esper_NN.loc[:,'G2talk'] - esper_NN.loc[:,'NNtalk' + str(i+1)]
        del_alk_surf_LIR = esper_surf_LIR.loc[:,'G2talk'] - esper_surf_LIR.loc[:,'LIRtalk' + str(i+1)]
        del_alk_surf_NN = esper_surf_NN.loc[:,'G2talk'] - esper_surf_NN.loc[:,'NNtalk' + str(i+1)]
        
        x_LIR = esper_LIR['dectime'].to_numpy()
        x_surf_LIR = esper_surf_LIR['dectime'].to_numpy()
        x_NN = esper_NN['dectime'].to_numpy()
        x_surf_NN = esper_surf_NN['dectime'].to_numpy()
        
        y_LIR = del_alk_LIR.to_numpy()
        y_NN = del_alk_NN.to_numpy()
        y_surf_LIR = del_alk_surf_LIR.to_numpy()
        y_surf_NN = del_alk_surf_NN.to_numpy()
    
        # fit model and print summary
        x_model_LIR = sm.add_constant(x_LIR) # this is required in statsmodels to get an intercept
        x_model_surf_LIR = sm.add_constant(x_surf_LIR) # this is required in statsmodels to get an intercept
        x_model_NN = sm.add_constant(x_NN) # this is required in statsmodels to get an intercept
        x_model_surf_NN = sm.add_constant(x_surf_NN) # this is required in statsmodels to get an intercept
        
        rlm_model_LIR = sm.RLM(y_LIR, x_model_LIR, M=sm.robust.norms.HuberT())
        rlm_results_LIR = rlm_model_LIR.fit()
        
        rlm_model_NN = sm.RLM(y_NN, x_model_NN, M=sm.robust.norms.HuberT())
        rlm_results_NN = rlm_model_NN.fit()
        
        rlm_model_surf_LIR = sm.RLM(y_surf_LIR, x_model_surf_LIR, M=sm.robust.norms.HuberT())
        rlm_results_surf_LIR = rlm_model_surf_LIR.fit()
        
        rlm_model_surf_NN = sm.RLM(y_surf_NN, x_model_surf_NN, M=sm.robust.norms.HuberT())
        rlm_results_surf_NN = rlm_model_surf_NN.fit()
    
        slopes_rlm_LIR[i] = rlm_results_LIR.params[1]
        slopes_rlm_NN[i] = rlm_results_NN.params[1]
        slopes_rlm_surf_LIR[i] = rlm_results_surf_LIR.params[1]
        slopes_rlm_surf_NN[i] = rlm_results_surf_NN.params[1]
        
        #ols_model_LIR = sm.OLS(y_LIR, x_model_LIR)
        #ols_results_LIR = ols_model_LIR.fit()
        
        #ols_model_NN = sm.OLS(y_NN, x_model_NN)
        #ols_results_NN = ols_model_NN.fit()
        
        #ols_model_surf_LIR = sm.OLS(y_surf_LIR, x_model_surf_LIR)
        #ols_results_surf_LIR = ols_model_surf_LIR.fit()
        
        #ols_model_surf_NN = sm.OLS(y_surf_NN, x_model_surf_NN)
        #ols_results_surf_NN = ols_model_surf_NN.fit()
        
        #slopes_ols_LIR[i] = ols_results_LIR.params[1]
        #slopes_ols_NN[i] = ols_results_NN.params[1]
        #slopes_ols_surf_LIR[i] = ols_results_surf_LIR.params[1]
        #slopes_ols_surf_NN[i] = ols_results_surf_NN.params[1]
    
    u_esper_LIR[j] = slopes_rlm_LIR.std()
    u_esper_NN[j] = slopes_rlm_NN.std()
    u_esper_surf_LIR[j] = slopes_rlm_surf_LIR.std()
    u_esper_surf_NN[j] = slopes_rlm_surf_NN.std()
    
    #u_esper_LIR[j] = slopes_ols_LIR.std()
    #u_esper_NN[j] = slopes_ols_NN.std()
    #u_esper_surf_LIR[j] = slopes_ols_surf_LIR.std()
    #u_esper_surf_NN[j] = slopes_ols_surf_NN.std()
    
    # calculate u_sample
    # loop through monte carlo simulation-produced G2talk to do global ensemble mean regression
    all_trimmed_basin = basin
    G2talk_mc_basin = mc_basin

    # plot surface values and do regular linear regression
    all_trimmed_basin.reset_index(drop=True, inplace=True)
    G2talk_mc_basin.reset_index(drop=True, inplace=True)
    all_trimmed_mc_basin = pd.concat([all_trimmed_basin, G2talk_mc_basin], axis=1)
    all_trimmed_mc_basin_surf = all_trimmed_mc_basin[all_trimmed_mc_basin.G2depth < all_trimmed_mc_basin.surface_depth]
    x = all_trimmed_mc_basin.dectime
    x_surf = all_trimmed_mc_basin_surf.dectime

    # preallocate arrays for storing slope and p-values
    slopes_surf_LIR = np.zeros(G2talk_mc.shape[1])
    pvalues_surf_LIR = np.zeros(G2talk_mc.shape[1])

    slopes_LIR = np.zeros(G2talk_mc.shape[1])
    pvalues_LIR = np.zeros(G2talk_mc.shape[1])

    slopes_surf_NN = np.zeros(G2talk_mc.shape[1])
    pvalues_surf_NN = np.zeros(G2talk_mc.shape[1])

    slopes_NN = np.zeros(G2talk_mc.shape[1])
    pvalues_NN = np.zeros(G2talk_mc.shape[1])

    for i in range(0,G2talk_mc.shape[1]): 
        y_surf_LIR = all_trimmed_mc_basin_surf[str(i)] - all_trimmed_mc_basin_surf.Ensemble_Mean_TA_LIR
        y_LIR = all_trimmed_mc_basin[str(i)] - all_trimmed_mc_basin.Ensemble_Mean_TA_LIR
        y_surf_NN = all_trimmed_mc_basin_surf[str(i)] - all_trimmed_mc_basin_surf.Ensemble_Mean_TA_NN
        y_NN = all_trimmed_mc_basin[str(i)] - all_trimmed_mc_basin.Ensemble_Mean_TA_NN
        
        slope_surf_LIR, intercept_surf_LIR, _, pvalue_surf_LIR, _ = stats.linregress(x_surf, y_surf_LIR, alternative='two-sided')
        slope_LIR, intercept_LIR, _, pvalue_LIR, _ = stats.linregress(x, y_LIR, alternative='two-sided')
        slope_surf_NN, intercept_surf_NN, _, pvalue_surf_NN, _ = stats.linregress(x_surf, y_surf_NN, alternative='two-sided')
        slope_NN, intercept_NN, _, pvalue_NN, _ = stats.linregress(x, y_NN, alternative='two-sided')
        
        slopes_surf_LIR[i] = slope_surf_LIR
        pvalues_surf_LIR[i] = pvalue_surf_LIR
        slopes_LIR[i] = slope_LIR
        pvalues_LIR[i] = pvalue_LIR
        slopes_surf_NN[i] = slope_surf_NN
        pvalues_surf_NN[i] = pvalue_surf_NN
        slopes_NN[i] = slope_NN
        pvalues_NN[i] = pvalue_NN
    
    u_sample_surf_LIR[j] = slopes_surf_LIR.std() # for SURFACE, LIR
    u_sample_LIR[j] = slopes_LIR.std() # for FULL DEPTH, LIR
    u_sample_surf_NN[j] = slopes_surf_NN.std() # for SURFACE, NN
    u_sample_NN[j] = slopes_NN.std() # for FULL DEPTH, NN
    
    basin_U_surf_LIR[j] = np.sqrt(u_esper_surf_LIR[j]**2 + u_sample_surf_LIR[j]**2)
    basin_U_surf_NN[j] = np.sqrt(u_esper_surf_NN[j]**2 + u_sample_surf_NN[j]**2)
    basin_U_LIR[j] = np.sqrt(u_esper_LIR[j]**2 + u_sample_LIR[j]**2)
    basin_U_NN[j] = np.sqrt(u_esper_NN[j]**2 + u_sample_NN[j]**2)
    
# %% calculate arrays of slopes as well

basin_trend_surf_LIR = np.zeros(len(basins))
basin_trend_surf_NN = np.zeros(len(basins))
basin_trend_LIR = np.zeros(len(basins))
basin_trend_NN = np.zeros(len(basins))

for basin, j in zip(basins, range(0,len(basins))):
    esper = basin
    esper_surf = basin[basin.G2depth < basin.surface_depth] # do surface values (< 25 m) only

    # sort by time
    esper = esper.sort_values(by=['dectime'],ascending=True)
    esper_surf = esper_surf.sort_values(by=['dectime'],ascending=True)

    # calculate the difference in TA betwen GLODAP and ESPERS, store for regression
    del_alk_surf_LIR = esper_surf.loc[:,'G2talk'] - esper_surf.loc[:,'Ensemble_Mean_TA_LIR']
    #del_alk_surf_LIR = esper_surf.loc[:,'G2talk'] - esper_surf.loc[:,'TA_Ensemble_-_16_LIR']
    x_surf_LIR = esper_surf['dectime'].to_numpy()
    y_surf_LIR = del_alk_surf_LIR.to_numpy()
    
    del_alk_surf_NN = esper_surf.loc[:,'G2talk'] - esper_surf.loc[:,'Ensemble_Mean_TA_NN']
    #del_alk_surf_NN = esper_surf.loc[:,'G2talk'] - esper_surf.loc[:,'TA_Ensemble_-_16_NN']
    x_surf_NN = esper_surf['dectime'].to_numpy()
    y_surf_NN = del_alk_surf_NN.to_numpy()
    
    del_alk_LIR = esper.loc[:,'G2talk'] - esper.loc[:,'Ensemble_Mean_TA_LIR']
    #del_alk_LIR = esper.loc[:,'G2talk'] - esper.loc[:,'TA_Ensemble_-_16_LIR']
    x_LIR = esper['dectime'].to_numpy()
    y_LIR = del_alk_LIR.to_numpy()
    
    del_alk_NN = esper.loc[:,'G2talk'] - esper.loc[:,'Ensemble_Mean_TA_NN']
    #del_alk_NN = esper.loc[:,'G2talk'] - esper.loc[:,'TA_Ensemble_-_16_NN']
    x_NN = esper['dectime'].to_numpy()
    y_NN = del_alk_NN.to_numpy()
         
    # fit model and print summary
    x_model_surf_LIR = sm.add_constant(x_surf_LIR) # this is required in statsmodels to get an intercept
    rlm_model_surf_LIR = sm.RLM(y_surf_LIR, x_model_surf_LIR, M=sm.robust.norms.HuberT())
    rlm_results_surf_LIR = rlm_model_surf_LIR.fit()
    
    x_model_surf_NN = sm.add_constant(x_surf_NN) # this is required in statsmodels to get an intercept
    rlm_model_surf_NN = sm.RLM(y_surf_NN, x_model_surf_NN, M=sm.robust.norms.HuberT())
    rlm_results_surf_NN = rlm_model_surf_NN.fit()
    
    x_model_LIR = sm.add_constant(x_LIR) # this is required in statsmodels to get an intercept
    rlm_model_LIR = sm.RLM(y_LIR, x_model_LIR, M=sm.robust.norms.HuberT())
    rlm_results_LIR = rlm_model_LIR.fit()
    
    x_model_NN = sm.add_constant(x_NN) # this is required in statsmodels to get an intercept
    rlm_model_NN = sm.RLM(y_NN, x_model_NN, M=sm.robust.norms.HuberT())
    rlm_results_NN = rlm_model_NN.fit()
    
    basin_trend_surf_LIR[j] = rlm_results_surf_LIR.params[1]
    basin_trend_surf_NN[j] = rlm_results_surf_NN.params[1]
    basin_trend_LIR[j] = rlm_results_LIR.params[1]
    basin_trend_NN[j] = rlm_results_NN.params[1]
    
    #x_model_surf_LIR = sm.add_constant(x_surf_LIR) # this is required in statsmodels to get an intercept
    #ols_model_surf_LIR = sm.OLS(y_surf_LIR, x_model_surf_LIR)
    #ols_results_surf_LIR = ols_model_surf_LIR.fit()
    
    #x_model_surf_NN = sm.add_constant(x_surf_NN) # this is required in statsmodels to get an intercept
    #ols_model_surf_NN = sm.OLS(y_surf_NN, x_model_surf_NN)
    #ols_results_surf_NN = ols_model_surf_NN.fit()
    
    #x_model_LIR = sm.add_constant(x_LIR) # this is required in statsmodels to get an intercept
    #ols_model_LIR = sm.OLS(y_LIR, x_model_LIR)
    #ols_results_LIR = ols_model_LIR.fit()
    
    #x_model_NN = sm.add_constant(x_NN) # this is required in statsmodels to get an intercept
    #ols_model_NN = sm.OLS(y_NN, x_model_NN)
    #ols_results_NN = ols_model_NN.fit()
    
    #basin_trend_surf_LIR[j] = ols_results_surf_LIR.params[1]
    #basin_trend_surf_NN[j] = ols_results_surf_NN.params[1]
    #basin_trend_LIR[j] = ols_results_LIR.params[1]
    #basin_trend_NN[j] = ols_results_NN.params[1]
    
    
#%% print trend ± uncertainty
for i in range(0,8):
    #print(round(basin_trend_surf_LIR[i],3), '±', round(basin_U_surf_LIR[i],3))
    #print(round(basin_trend_LIR[i],3), '±', round(basin_U_LIR[i],3))
    #print(round(basin_trend_surf_NN[i],3), '±', round(basin_U_surf_NN[i],3))
    print(round(basin_trend_NN[i],3), '±', round(basin_U_NN[i],3))

#%% plot trends with error bars - TRANSECTS ONLY
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4.5), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR[8:])))

axs[0].errorbar(x-0.15, basin_trend_surf_LIR[8:], yerr=basin_U_surf_LIR[8:], fmt="o", c='indigo', alpha = 0.5)
axs[0].errorbar(x+0.15, basin_trend_surf_NN[8:], yerr=basin_U_surf_NN[8:], fmt="o", c='dodgerblue', alpha = 0.5)

axs[1].errorbar(x-0.15, basin_trend_LIR[8:], yerr=basin_U_LIR[8:], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[1].errorbar(x+0.15, basin_trend_NN[8:], yerr=basin_U_NN[8:], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)

axs[0].axhline(y=0, color='k', linestyle='--')
axs[1].axhline(y=0, color='k', linestyle='--')

axs[0].set_ylim([-0.5, 0.5])   
axs[1].set_ylim([-0.5, 0.5]) 
#axs[0].set_ylim([-7.5, 7.5])   
#axs[1].set_ylim([-7.5, 7.5])    
axs[1].set_xlim(-0.5, len(basin_trend_LIR[8:]) - 0.5) 
 
basin_abbr = ['A02', 'A05', 'A10',
              'A12', 'A135', 'A16N', 'A16S', 'A17', 'A20', 'A22', 'A25', 'A29',
              'AR07E', 'AR07W', 'ARC01E', 'I03', 'I05', 'I06', 'I07', 'I08N', 'I08S',
              'I09N', 'I09S', 'I10', 'P01', 'P02', 'P03', 'P06', 'P09', 'P10', 'P13',
              'P14', 'P15', 'P16N', 'P16S', 'P17N', 'P18', 'P21', 'S04I',
              'SR04', 'S04P', 'SR01', 'SR03']
axs[1].set_xticks(x, basin_abbr)

ax.set_ylabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)')
#ax.set_ylabel('Temporal Trend in Measured $A_{T}$ - (Ensemble Mean\nESPER-Estimated $A_{T}$ - Eqn. 16 Estimate), ($µmol$ $kg^{-1}$ $yr^{-1}$)')
ax.yaxis.set_label_coords(-0.57,0.55)
axs[1].tick_params(axis='x', labelrotation=90)
axs[0].text(0, -0.45, 'A', fontsize=12)
axs[1].text(0, -0.45, 'B', fontsize=12)
axs[1].legend(bbox_to_anchor = (0.6, 0.21), ncol=2)

#axs[0].text(0, -2, 'A', fontsize=12)
#axs[1].text(0, -2, 'B', fontsize=12)
#axs[1].legend(bbox_to_anchor = (1, 0.21), ncol=2)
#axs[0].text(0, -7, 'Surface (< 25 m)', fontsize=12)
#axs[1].text(0, -7, 'Full Depth', fontsize=12)

#%% plot trends with error bars
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8,4.5), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR)))

axs[0].errorbar(x-0.15, basin_trend_surf_LIR, yerr=basin_U_surf_LIR, fmt="o", c='indigo', alpha = 0.5)
axs[0].errorbar(x+0.15, basin_trend_surf_NN, yerr=basin_U_surf_NN, fmt="o", c='dodgerblue', alpha = 0.5)

axs[1].errorbar(x-0.15, basin_trend_LIR, yerr=basin_U_LIR, fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[1].errorbar(x+0.15, basin_trend_NN, yerr=basin_U_NN, fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)

axs[0].axhline(y=0, color='k', linestyle='--')
axs[1].axhline(y=0, color='k', linestyle='--')

#axs[0].set_ylim([-0.5, 0.5])   
#axs[1].set_ylim([-0.5, 0.5]) 
#axs[0].set_ylim([-7.5, 7.5])   
#axs[1].set_ylim([-7.5, 7.5])    
axs[1].set_xlim(-0.5, len(basin_trend_LIR) - 0.5) 
 
basin_abbr = ['Global', 'NAO', 'SAO', 'NPO', 'SPO', 'IO', 'SO', 'AO', 'A02', 'A05', 'A10',
              'A12', 'A135', 'A16N', 'A16S', 'A17', 'A20', 'A22', 'A25', 'A29',
              'AR07E', 'AR07W', 'ARC01E', 'I03', 'I05', 'I06', 'I07', 'I08N', 'I08S',
              'I09N', 'I09S', 'I10', 'P01', 'P02', 'P03', 'P06', 'P09', 'P10', 'P13',
              'P14', 'P15', 'P16N', 'P16S', 'P17N', 'P18', 'P21', 'S04I',
              'SR04', 'S04P', 'SR01', 'SR03']
axs[1].set_xticks(x, basin_abbr)

ax.set_ylabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)')
#ax.set_ylabel('Temporal Trend in Measured $A_{T}$ - (Ensemble Mean\nESPER-Estimated $A_{T}$ - Eqn. 16 Estimate), ($µmol$ $kg^{-1}$ $yr^{-1}$)')
ax.yaxis.set_label_coords(-0.57,0.55)
axs[1].tick_params(axis='x', labelrotation=90)
#axs[0].text(0, -0.45, 'A', fontsize=12)
#axs[1].text(0, -0.45, 'B', fontsize=12)
#axs[1].legend(bbox_to_anchor = (0.6, 0.21), ncol=2)

axs[0].text(0, -2, 'A: Surface (< 25 m)', fontsize=12)
axs[1].text(0, -2, 'B: Full Depth', fontsize=12)
axs[1].legend(bbox_to_anchor = (1, 0.21), ncol=2)
#axs[0].text(0, -7, 'Surface (< 25 m)', fontsize=12)
#axs[1].text(0, -7, 'Full Depth', fontsize=12)

#%% plot trends with error bars: 3 panel (surface, depth, surface - depth)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7.5,3.5), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR[0:8])))

# surface
axs[0].errorbar(x-0.15, basin_trend_surf_LIR[0:8], yerr=basin_U_surf_LIR[0:8], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[0].errorbar(x+0.15, basin_trend_surf_NN[0:8], yerr=basin_U_surf_NN[0:8], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)
axs[0].axhline(y=0, color='k', linestyle='--')

# depth
axs[1].errorbar(x-0.15, basin_trend_LIR[0:8], yerr=basin_U_LIR[0:8], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[1].errorbar(x+0.15, basin_trend_NN[0:8], yerr=basin_U_NN[0:8], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)
axs[1].axhline(y=0, color='k', linestyle='--')

# surface - depth
trend_diff_LIR = basin_trend_surf_LIR - basin_trend_LIR
trend_diff_NN = basin_trend_surf_NN - basin_trend_NN
error_summed_LIR = np.sqrt(basin_U_surf_LIR**2 + basin_U_LIR**2)
error_summed_NN = np.sqrt(basin_U_surf_NN**2 + basin_U_NN**2)

axs[2].errorbar(x-0.15, trend_diff_LIR[0:8], yerr=error_summed_LIR[0:8], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[2].errorbar(x+0.15, trend_diff_NN[0:8], yerr=error_summed_NN[0:8], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)
axs[2].axhline(y=0, color='k', linestyle='--')

axs[0].set_ylim([-0.25, 0.25])   
axs[0].set_xlim(-0.5, len(basin_trend_LIR[0:8]) - 0.5) 
 
basin_abbr = ['Global', 'N. Atlantic', 'S. Atlantic', 'N. Pacific', 'S. Pacific', 'Indian', 'Southern', 'Arctic']
axs[0].set_xticks(x, basin_abbr)
axs[0].tick_params(axis='x', labelrotation=90)
axs[1].tick_params(axis='x', labelrotation=90)
axs[2].tick_params(axis='x', labelrotation=90)

ax.set_ylabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)',fontsize=11)
ax.yaxis.set_label_coords(-0.55,0.5)
axs[0].text(-0.35, -0.23, 'A: Surface (< 25 m)', fontsize=11)
axs[1].text(-0.35, -0.23, 'B: Full Depth', fontsize=11)
axs[2].text(-0.35, -0.23, 'C: Surface - Full Depth', fontsize=11)
axs[0].legend(bbox_to_anchor = (0.58, 0.32), ncol=1)


#%% plot trends with error bars - REGIONS ONLY (for presentation)
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR[0:8])))

axs[0].errorbar(x-0.15, basin_trend_surf_LIR[0:8], yerr=basin_U_surf_LIR[0:8], fmt="o", c='indigo', alpha = 0.5)
axs[0].errorbar(x+0.15, basin_trend_surf_NN[0:8], yerr=basin_U_surf_NN[0:8], fmt="o", c='dodgerblue', alpha = 0.5)

axs[1].errorbar(x-0.15, basin_trend_LIR[0:8], yerr=basin_U_LIR[0:8], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs[1].errorbar(x+0.15, basin_trend_NN[0:8], yerr=basin_U_NN[0:8], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)

axs[0].axhline(y=0, color='k', linestyle='--')
axs[1].axhline(y=0, color='k', linestyle='--')

axs[0].set_ylim([-0.25, 0.25])   
axs[1].set_ylim([-0.25, 0.25])   
axs[1].set_xlim(-0.5, len(basin_trend_LIR[0:8]) - 0.5) 
 
basin_abbr = ['Global', 'N. Atlantic', 'S. Atlantic', 'N. Pacific', 'S. Pacific', 'Indian', 'Southern', 'Arctic']
axs[1].set_xticks(x, basin_abbr)

ax.set_ylabel('Temporal Trend in $∆A_{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)',fontsize=10)
ax.yaxis.set_label_coords(-0.62,0.58)
axs[1].tick_params(axis='x', labelrotation=45)
axs[0].text(-0.35, -0.22, 'Surface (< 25 m)', fontsize=10)
axs[1].text(-0.35, -0.22, 'Full Depth', fontsize=10)
#axs[1].legend(bbox_to_anchor = (0.8, -0.75), ncol=2)
axs[1].legend(bbox_to_anchor = (1.0, 0.25), ncol=2)


#axs[0].text(0, -2, 'A', fontsize=12)
#axs[1].text(0, -2, 'B', fontsize=12)
#axs[1].legend(bbox_to_anchor = (1, 0.21), ncol=2)

#%% plot trends with error bars (surface - depth)
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,4.5), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR)))

trend_diff_LIR = basin_trend_surf_LIR - basin_trend_LIR
trend_diff_NN = basin_trend_surf_NN - basin_trend_NN
error_summed_LIR = np.sqrt(basin_U_surf_LIR**2 + basin_U_LIR**2)
error_summed_NN = np.sqrt(basin_U_surf_NN**2 + basin_U_NN**2)

axs.errorbar(x-0.15, trend_diff_LIR, yerr=error_summed_LIR, fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs.errorbar(x+0.15, trend_diff_NN, yerr=error_summed_NN, fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)

axs.axhline(y=0, color='k', linestyle='--')

#axs[0].set_ylim([-0.5, 0.5])   
#axs[1].set_ylim([-0.5, 0.5]) 
#axs[0].set_ylim([-7.5, 7.5])   
#axs[1].set_ylim([-7.5, 7.5])    
axs.set_xlim(-0.5, len(basin_trend_LIR) - 0.5) 
 
basin_abbr = ['Global', 'NAO', 'SAO', 'NPO', 'SPO', 'IO', 'SO', 'AO', 'A02', 'A05', 'A10',
              'A12', 'A135', 'A16N', 'A16S', 'A17', 'A20', 'A22', 'A25', 'A29',
              'AR07E', 'AR07W', 'ARC01E', 'I03', 'I05', 'I06', 'I07', 'I08N', 'I08S',
              'I09N', 'I09S', 'I10', 'P01', 'P02', 'P03', 'P06', 'P09', 'P10', 'P13',
              'P14', 'P15', 'P16N', 'P16S', 'P17N', 'P18', 'P21', 'S04I',
              'SR04', 'S04P', 'SR01', 'SR03']
axs.set_xticks(x, basin_abbr)

axs.set_ylabel('Temporal Trend in Measured $A_{T}$ - ESPER-Estimated $A_{T}$\n($µmol$ $kg^{-1}$ $yr^{-1}$)')
#ax.set_ylabel('Temporal Trend in Measured $A_{T}$ - (Ensemble Mean\nESPER-Estimated $A_{T}$ - Eqn. 16 Estimate), ($µmol$ $kg^{-1}$ $yr^{-1}$)')
#axs.yaxis.set_label_coords(-0.62,0.55)
axs.tick_params(axis='x', labelrotation=90)
#axs[0].text(0, -0.45, 'A', fontsize=12)
#axs[1].text(0, -0.45, 'B', fontsize=12)
#axs[1].legend(bbox_to_anchor = (0.6, 0.21), ncol=2)

axs.text(0, -2, 'A', fontsize=12)
axs.legend(bbox_to_anchor = (1, 0.21), ncol=2)
#axs[0].text(0, -7, 'Surface (< 25 m)', fontsize=12)
#axs[1].text(0, -7, 'Full Depth', fontsize=12)

#%% plot trends with error bars - surface - depth, REGIONS ONLY
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

x = np.array(range(0,len(basin_trend_LIR[0:8])))

trend_diff_LIR = basin_trend_surf_LIR - basin_trend_LIR
trend_diff_NN = basin_trend_surf_NN - basin_trend_NN
error_summed_LIR = np.sqrt(basin_U_surf_LIR**2 + basin_U_LIR**2)
error_summed_NN = np.sqrt(basin_U_surf_NN**2 + basin_U_NN**2)

axs.errorbar(x-0.15, trend_diff_LIR[0:8], yerr=error_summed_LIR[0:8], fmt="o", c='indigo', label='ESPER LIR', alpha = 0.5)
axs.errorbar(x+0.15, trend_diff_NN[0:8], yerr=error_summed_NN[0:8], fmt="o", c='dodgerblue', label='ESPER NN', alpha = 0.5)

axs.axhline(y=0, color='k', linestyle='--')

axs.set_ylim([-0.25, 0.25])   
axs.set_xlim(-0.5, len(basin_trend_LIR[0:8]) - 0.5) 
 
basin_abbr = ['Global', 'N. Atlantic', 'S. Atlantic', 'N. Pacific', 'S. Pacific', 'Indian', 'Southern', 'Arctic']
axs.set_xticks(x, basin_abbr)

axs.set_ylabel('Temporal Trend in Surface $∆A_{T}$ - Trend\nin Full Depth $∆A_{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)',fontsize=10)
axs.tick_params(axis='x', labelrotation=45)
#axs.text(-0.35, -0.18, 'Surface  - Full Depth' , fontsize=10)
#axs[1].legend(bbox_to_anchor = (0.8, -0.75), ncol=2)
axs.legend(bbox_to_anchor = (0.68, 0.12), ncol=2)


#axs[0].text(0, -2, 'A', fontsize=12)
#axs[1].text(0, -2, 'B', fontsize=12)
#axs[1].legend(bbox_to_anchor = (1, 0.21), ncol=2)

# %% make box plot graph of transect slopes from mc simulation

# create dictionary with regional data + mc as values, regions as keys
north_atlantic.reset_index(drop=True, inplace=True)
G2talk_mc_NA.reset_index(drop=True, inplace=True)
NA_mc = pd.concat([north_atlantic, G2talk_mc_NA], axis=1)
NA_mc_surf = NA_mc[NA_mc.G2depth < NA_mc.surface_depth]

south_atlantic.reset_index(drop=True, inplace=True)
G2talk_mc_SA.reset_index(drop=True, inplace=True)
SA_mc = pd.concat([south_atlantic, G2talk_mc_SA], axis=1)
SA_mc_surf = SA_mc[SA_mc.G2depth < SA_mc.surface_depth]

north_pacific.reset_index(drop=True, inplace=True)
G2talk_mc_NP.reset_index(drop=True, inplace=True)
NP_mc = pd.concat([north_pacific, G2talk_mc_NP], axis=1)
NP_mc_surf = NP_mc[NP_mc.G2depth < NP_mc.surface_depth]

south_pacific.reset_index(drop=True, inplace=True)
G2talk_mc_SP.reset_index(drop=True, inplace=True)
SP_mc = pd.concat([south_pacific, G2talk_mc_SP], axis=1)
SP_mc_surf = SP_mc[SP_mc.G2depth < SP_mc.surface_depth]

indian.reset_index(drop=True, inplace=True)
G2talk_mc_IO.reset_index(drop=True, inplace=True)
IO_mc = pd.concat([indian, G2talk_mc_IO], axis=1)
IO_mc_surf = IO_mc[IO_mc.G2depth < IO_mc.surface_depth]

southern.reset_index(drop=True, inplace=True)
G2talk_mc_SO.reset_index(drop=True, inplace=True)
SO_mc = pd.concat([southern, G2talk_mc_SO], axis=1)
SO_mc_surf = SO_mc[SO_mc.G2depth < SO_mc.surface_depth]

arctic.reset_index(drop=True, inplace=True)
G2talk_mc_AO.reset_index(drop=True, inplace=True)
AO_mc = pd.concat([arctic, G2talk_mc_AO], axis=1)
AO_mc_surf = AO_mc[AO_mc.G2depth < AO_mc.surface_depth]

trimmed_mc_basin = {'NAO' : NA_mc, 'SAO' : SA_mc, 'NPO' : NP_mc, 'SPO' : SP_mc,
                    'IO' : IO_mc, 'SO' : SO_mc, 'AO' : AO_mc}

trimmed_mc_basin_surf = {'NAO' : NA_mc_surf, 'SAO' : SA_mc_surf,
                         'NPO' : NP_mc_surf, 'SPO' : SP_mc_surf,
                         'IO' : IO_mc_surf, 'SO' : SO_mc_surf,
                         'AO' : AO_mc_surf}

# SURFACE LIR
all_slopes_surf_LIR = p1.transect_box_plot(trimmed_mc_basin_surf, G2talk_mc, 'LIR')
all_slopes_surf_NN = p1.transect_box_plot(trimmed_mc_basin_surf, G2talk_mc, 'NN')

# FULL-OCEAN LIR
all_slopes_full_LIR = p1.transect_box_plot(trimmed_mc_basin, G2talk_mc, 'LIR')
all_slopes_full_NN = p1.transect_box_plot(trimmed_mc_basin, G2talk_mc, 'NN')

# set up plot
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# make box plot for surface
axs[0,0].boxplot(all_slopes_surf_LIR, vert=True, labels=list(trimmed_mc_basin_surf.keys()))
axs[0,0].axhline(y=0, color='r', linestyle='--')
axs[0,0].set_ylim(-1.5, 1.5)
axs[0,0].text(0.67, -1.35, 'A: ESPER_LIR (< 25 m)', fontsize=11)

# make box plot for full depth
axs[1,0].boxplot(all_slopes_full_LIR, vert=True, labels=list(trimmed_mc_basin.keys()))
axs[1,0].axhline(y=0, color='r', linestyle='--')
axs[1,0].set_ylim(-1.5, 1.5)
axs[1,0].text(0.67, -1.35, 'C:  ESPER_LIR', fontsize=11)

axs[0,1].boxplot(all_slopes_surf_NN, vert=True, labels=list(trimmed_mc_basin_surf.keys()))
axs[0,1].axhline(y=0, color='r', linestyle='--')
axs[0,1].set_ylim(-1.5, 1.5)
axs[0,1].text(0.67, -1.35, 'B: ESPER_NN (< 25 m)', fontsize=11)

# make box plot for full depth
axs[1,1].boxplot(all_slopes_full_NN, vert=True, labels=list(trimmed_mc_basin.keys()))
axs[1,1].axhline(y=0, color='r', linestyle='--')
axs[1,1].set_ylim(-1.5, 1.5)
axs[1,1].text(0.67, -1.35, 'D: ESPER_NN', fontsize=11)

ax.set_ylabel('Temporal Trend in $∆A_\mathrm{T}$ ($µmol$ $kg^{-1}$ $yr^{-1}$)')

ax.yaxis.set_label_coords(-0.54,0.55)
axs[1,0].tick_params(axis='x', labelrotation=90)
axs[1,1].tick_params(axis='x', labelrotation=90)

#%% do calculations on OA amplification for conclusion
avg_surf_TA = 2305 # µmol/kg, carter et al., 2014
avg_TA = 2362 # carter et al., 2014

ppm_1994 = 358.96 # unc = 0.12, at Mauna Loa, https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt
ppm_2024 = 426.91 # unc = 0.12, at Mauna Loa, https://gml.noaa.gov/ccgg/trends/mlo.html

# DIC in 1994
results_1994 = pyco2.sys(par1=avg_surf_TA, par1_type=1, par2=ppm_1994, par2_type=9)
print('Surface DIC in 1994:', results_1994['dic'])

# DIC in 2024 (assuming fixed TA)
results_2024 = pyco2.sys(par1=avg_surf_TA, par1_type=1, par2=ppm_2024, par2_type=9)
print('Surface DIC in 2024 (no feedback):', results_2024['dic'])

# DIC in 2024 (assuming elevated TA from CO2-biotic calcification feedback)
TA_past_30 = 0.072 * 30 # umol/kg, using surface ocean ESPER LIR ∆TA trend
results_2024_bc = pyco2.sys(par1=(avg_surf_TA+TA_past_30), par1_type=1, par2=ppm_2024, par2_type=9)
print('Surface DIC in 2024 (yes feedback):', results_2024_bc['dic'])

# % change with bc feedback
percent_change = 100*(((results_2024_bc['dic'] - results_1994['dic']) - (results_2024['dic'] - results_1994['dic']))/(results_2024['dic'] - results_1994['dic']))
print('% change in surface DIC:', percent_change, '%')

# mass change
# mass of surface ocean is 9.2e18 k
# molar mass of carbon is 12.011 g/mol
mass_change = (results_2024_bc['dic'] - results_2024['dic'])*1e-6*9.2e18*12.011
print('Mass change in surface DIC:', mass_change, 'g')
print('Mass change in surface DIC:', mass_change/1e15, 'Pg')

#%% plot hot and bats data over time (load data)

# load in hot data
HOT = pd.read_csv(filepath + 'niskin_v2.csv', na_values='nd')
HOT['datetime'] = pd.to_datetime(HOT['ISO_DateTime_UTC'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce') # convert to datetime
HOT = HOT.dropna(subset=['ALKALIN']) # get rid of nans
HOT = HOT.dropna(subset=['SALINITY']) # get rid of nans
HOT = HOT.reset_index(drop=True) # reset index

# convert from datetime to decimal time
# allocate decimal year column
HOT.insert(0,'dectime', 0.0)

for i in range(len(HOT)):
    # convert datetime object to decimal time
    date = HOT['datetime'].iloc[i]
    year = date.year
    this_year_start = dt(year=year, month=1, day=1)
    next_year_start = dt(year=year+1, month=1, day=1)
    year_elapsed = date - this_year_start
    year_duration = next_year_start - this_year_start
    fraction = year_elapsed / year_duration
    decimal_time = date.year + fraction
    
    # save to glodap dataset
    HOT.loc[i,'dectime'] = decimal_time

HOT = HOT[['EXPOCODE','WHPID','STNNBR','CASTNO',
           'dectime','datetime','Latitude','Longitude','Depth_max',
           'CTDTMP','SALINITY','OXYGEN','NO2_NO3',
           'SILCAT','PHSPHT','ALKALIN','pH']]
        
#HOT.to_csv(filepath + 'HOT_for_ESPERs.csv', index=False)

#%% load in HOT DOGS data
headers = ['botid', 'date', 'time', 'press', 'temp', 'bsal', 'boxy', 'alk', 'phos', 'nit', 'sil']
def parse_mmddyy(date_str):
    try:
        return pd.to_datetime(date_str.strip(), format='%m%d%y')
    except ValueError:
        # Return NaT (Not a Time) if parsing fails
        return pd.NaT
HOT = pd.read_csv(filepath + 'hot_dogs_data.txt', na_values=['-00009','-9.0','-9.00'], converters={'date': parse_mmddyy}, skiprows=5, header=None, names=headers, index_col=False)
HOT.replace(-9, np.nan, inplace=True) # set nan values correctly
HOT = HOT.dropna(subset=['date'])

# convert from datetime to decimal time
# allocate decimal year column
HOT.insert(0,'dectime', 0.0)

for i in range(len(HOT)):
    # convert datetime object to decimal time
    date = HOT['date'].iloc[i]
    year = date.year
    this_year_start = dt(year=year, month=1, day=1)
    next_year_start = dt(year=year+1, month=1, day=1)
    year_elapsed = date - this_year_start
    year_duration = next_year_start - this_year_start
    fraction = year_elapsed / year_duration
    decimal_time = date.year + fraction
    
    # save to glodap dataset
    HOT.loc[i,'dectime'] = decimal_time

HOT = HOT[['botid', 'dectime', 'press', 'temp', 'bsal', 'boxy', 'alk', 'phos', 'nit', 'sil']]
        
#HOT.to_csv(filepath + 'HOT_DOGS_for_ESPERs.csv', index=False)

#%% load in BATS data
headers = ['Id', 'yyyymmdd', 'decy', 'time', 'latN', 'lonW', 'QF', 'Depth', 'Temp', 'CTD_S', 'Sal1', 'Sig-th', 'O2(1)', 'OxFixT', 'Anom1', 'CO2', 'Alk', 'NO31', 'NO21', 'PO41', 'Si1', 'POC', 'PON', 'TOC', 'TN', 'Bact', 'POP', 'TDP', 'SRP', 'BSi', 'LSi', 'Pro', 'Syn', 'Piceu', 'Naneu']
BATS = pd.read_csv(filepath + 'bats_bottle.txt', skiprows=59, header=None, names=headers, delimiter='\t')
BATS['datetime'] = pd.to_datetime(BATS['yyyymmdd'], format='%Y%m%d', errors='coerce') # convert to datetime
BATS.replace(-999, np.nan, inplace=True) # set nan values correctly
BATS = BATS.dropna(subset=['Alk']) # get rid of nans
BATS = BATS.dropna(subset=['Sal1']) # get rid of nans
BATS = BATS[BATS['QF'] == 2] # get rid of QF not equal to 2
BATS = BATS.reset_index(drop=True) # reset index

BATS = BATS[['Id','decy','datetime','latN','lonW','Depth','Temp','Sal1',
             'O2(1)','NO31','Si1','PO41','Alk']]
#BATS.to_csv(filepath + 'BATS_for_ESPERs.csv', index=False)

#%% plot hot and bats data over time (load in ESPERs, analyze)
espers_HOT = pd.read_csv(filepath + 'HOT_with_ESPER_TA.csv') # to do the normal ESPER
espers_HOT_DOGS = pd.read_csv(filepath + 'HOT_DOGS_with_ESPER_TA.csv') # to do the normal ESPER
espers_BATS = pd.read_csv(filepath + 'BATS_with_ESPER_TA.csv') # to do the normal ESPER

espers_HOT['surface_depth'] = 25
espers_HOT = p1.ensemble_mean(espers_HOT)
espers_HOT_DOGS['surface_depth'] = 25
espers_HOT_DOGS = p1.ensemble_mean(espers_HOT_DOGS)
espers_BATS['surface_depth'] = 25
espers_BATS = p1.ensemble_mean(espers_BATS)
#%% make figure (hot)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
#esper_type = 'LIRtalk1' # LIR, NN, or Mixed
esper_sel = espers_HOT
#esper_sel = espers_HOT.dropna(subset=['LIRtalk1'])
p1.plot2dhist_HOT(esper_sel, esper_type, fig, axs[0], 'ESPER_LIR (Full Depth)', 0)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
#esper_type = 'NNtalk1' # LIR, NN, or Mixed
esper_sel = espers_HOT
#esper_sel = espers_HOT.dropna(subset=['NNtalk1'])
p1.plot2dhist_HOT(esper_sel, esper_type, fig, axs[1], 'ESPER_NN (Full Depth)', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% make figure (hot dogs)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_HOT_DOGS
esper_sel = esper_sel[esper_sel.press < esper_sel.surface_depth]
esper_sel = esper_sel[esper_sel.dectime != 0]
esper_sel = esper_sel.dropna(subset=['alk'])
esper_sel = esper_sel.dropna(subset=['Ensemble_Mean_TA_LIR'])
p1.plot2dhist_HOT_DOGS(esper_sel, esper_type, fig, axs[0,0], 'ESPER_LIR (< 25 m)', 0)

# surface ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_HOT_DOGS
esper_sel = esper_sel[esper_sel.press < esper_sel.surface_depth]
esper_sel = esper_sel[esper_sel.dectime != 0]
esper_sel = esper_sel.dropna(subset=['alk'])
esper_sel = esper_sel.dropna(subset=['Ensemble_Mean_TA_NN'])
p1.plot2dhist_HOT_DOGS(esper_sel, esper_type, fig, axs[0,1], 'ESPER_NN (< 25 m)', 1)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_HOT_DOGS
esper_sel = esper_sel[esper_sel.dectime != 0]
esper_sel = esper_sel.dropna(subset=['alk'])
esper_sel = esper_sel.dropna(subset=['Ensemble_Mean_TA_LIR'])
p1.plot2dhist_HOT_DOGS(esper_sel, esper_type, fig, axs[1,0], 'ESPER_LIR (Full Depth)', 0)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_HOT_DOGS
esper_sel = esper_sel[esper_sel.dectime != 0]
esper_sel = esper_sel.dropna(subset=['alk'])
esper_sel = esper_sel.dropna(subset=['Ensemble_Mean_TA_NN'])
p1.plot2dhist_HOT_DOGS(esper_sel, esper_type, fig, axs[1, 1], 'ESPER_NN (Full Depth)', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% plot hot data against salinity
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_HOT
esper_sel = espers_HOT.dropna(subset=['SILCAT'])
p1.plot2dhist_S_HOT(esper_sel, esper_type, fig, axs[0], 'ESPER_LIR (Full Depth)', 0)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_HOT
esper_sel = espers_HOT.dropna(subset=['SILCAT'])
p1.plot2dhist_S_HOT(esper_sel, esper_type, fig, axs[1], 'ESPER_NN (Full Depth)', 1)

ax.set_xlabel('Silicate')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% make figure (different bats depth levels)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[esper_sel.Depth < esper_sel.surface_depth] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.decy < 2012]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[0,0], 'ESPER_LIR (< 25 m)', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[esper_sel.decy < 2012]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[1,0], 'ESPER_LIR (Full Depth)', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[esper_sel.Depth < esper_sel.surface_depth] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.decy < 2012]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[0,1], 'ESPER_NN (< 25 m)', 1)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[esper_sel.decy < 2012]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[1,1], 'ESPER_NN (Full Depth)', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('$∆A_\mathrm{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5,3), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[(esper_sel.Depth > 0) & (esper_sel.Depth < 500)]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[0], 'ESPER_LIR (500 - 1000 m)', 0)

# NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = espers_BATS
esper_sel = esper_sel[(esper_sel.Depth > 0) & (esper_sel.Depth < 500)]
p1.plot2dhist_BATS(esper_sel, esper_type, fig, axs[1], 'ESPER_NN (500 - 1000 m)', 1)

# scatter bats data depth vs. time
plt.figure(figsize=(6,6))
plt.scatter(espers_BATS.decy, espers_BATS.Depth, marker='o')
plt.gca().invert_yaxis()
plt.title('$∆A_\mathrm{T}$ measurement depth at BATS')
plt.xlabel('Year')
plt.ylabel('Depth (m)')



