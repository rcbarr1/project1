#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project1_main.py
Author: Reese Barrett
Date: 2023-10-31

Description: Main script for Project 1, calls functions written in project1.py
    for data analysis
    
To-Do:
    - write function to do corrections in North Pacific (add to glodap_qc)
    - translate call_ESPERs.m to python once ESPERs in Python are released
"""

import project1_JPN as p1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm
import cmocean
import cmocean.cm as cmo
import PyCO2SYS as pyco2


filepath = '/Users/Reese/Documents/Research Projects/project1/data/' # where GLODAP data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)
input_mc_cruise_file = 'G2talk_mc_simulated_JPN.csv' # MC (per cruise) simulated data
input_mc_individual_file = 'G2talk_mc_individual_simulated_JPN.csv' # MC (per cruise) simulated data
coeffs_file = 'ESPER_LIR_coeffs.csv' # ESPER LIR coefficients saved from MATLAB
monthly_clim_file = 'monthlyclim.mat'

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

# drop pre-2010 Japanese cruises here also so I don't have to re-run espers
# p1.go_ship_only and p1.trim_go_ship already updated in this version
JPN_drop = [461, 468, 502, 504, 272, 497, 495, 567, 602, 505, 459, 488]
mask = espers['G2cruise'].isin(JPN_drop)
espers = espers[~mask]

# %% set depth to use as boundary between surface and deep ocean

# static depth boundary
espers['surface_depth'] = 25

# to use dynamic mixed layer depth
#monthly_clim = loadmat(filepath + monthly_clim_file)
#MLD_da_max = monthly_clim['mld_da_max']
#MLD_da_mean = monthly_clim['mld_da_mean']
#latm = monthly_clim['latm']
#lonm = monthly_clim['lonm']

#max_MLDs = p1.find_MLD(espers.G2longitude, espers.G2latitude, MLD_da_max, latm, lonm, 0)
#mean_MLDs = p1.find_MLD(espers.G2longitude, espers.G2latitude, MLD_da_mean, latm, lonm, 1)

#espers['surface_depth'] = max_MLDs

# %% use KL divergence to determine which equations predict best (lower KL divergence = two datasets are closer)
kl_div = p1.kl_divergence(espers)
#kl_div.to_csv('kl_div.csv')

# %% calculate ensemble mean TA for each data point
espers = p1.ensemble_mean(espers)

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
ax1.hist(hist_data, bins=np.arange(np.nanmin(hist_data.flatten()), np.nanmax(hist_data.flatten()) + binwidth, binwidth), histtype ='step', stacked=True,fill=False, label=labels)
ax1.set_xlim([-5,5])
ax1.set_ylim([0,600000])
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
fig.text(0.14, 0.825, 'A', fontsize=11)

# plot ESPER NN results
hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['NNtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_NN']

ax2.hist(hist_data, bins=np.arange(np.nanmin(hist_data.flatten()), np.nanmax(hist_data.flatten()) + binwidth, binwidth), histtype ='step', stacked=True,fill=False, label=labels)
ax2.set_xlim([-5,5])
ax2.set_ylim([0,600000])
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

# label axis, set up legend
plt.ylabel('Number of $∆A_{T}$ Calculations')
plt.xlabel('ESPER-Estimated $A_{T}$ - Ensemble Mean $A_{T}$ ($µmol\;kg^{-1}$)')
fig.text(0.14, 0.41, 'B', fontsize=11)

handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], bbox_to_anchor = (1.05, 2.35), loc='upper left')

# %% USEFUL FOR VISUALIZING DATA LOCATIONS
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

im = ax.scatter(lon,lat,c=counts, cmap=cmo.dense, transform=ccrs.PlateCarree(), marker='o', edgecolors='none', s=15)
fig.colorbar(im, label='Number of Unique Years an\n$A_{T}$ Measurement Was Made', pad=0.02)

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

#%% plot difference between measured and espers on a map

# set up figure
fig = plt.figure(figsize=(10,4), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get espers data, exclude points with difference <5 or >-5 µmol/kg
surface = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth]
lat = surface.G2latitude
lon = surface.G2longitude
diff = surface.G2talk - surface.Ensemble_Mean_TA_LIR
to_plot = pd.DataFrame(data={'G2latitude' : lat, 'G2longitude' : lon, 'del_alk' : diff, 'abs_del_alk' : np.abs(diff)})
to_plot = to_plot[(to_plot.del_alk > -30) & (to_plot.del_alk < 30)]
to_plot = to_plot.sort_values('abs_del_alk',axis=0,ascending=True)

# create colormap
cmap = cmocean.tools.crop(cmo.balance, to_plot.del_alk.min(), to_plot.del_alk.max(), 0)

# plot data
pts = ax.scatter(to_plot.G2longitude,to_plot.G2latitude,transform=ccrs.PlateCarree(),s=30,c=to_plot.del_alk,cmap=cmap, alpha=0.15,edgecolors='none')
plt.colorbar(pts, ax=ax, label='Measured $A_{T}$ - ESPER-Estimated $A_{T}$ \n($µmol\;kg^{-1}$)')

#%% 2D histogram for global ensemble mean regression for all trimmed GO-SHIP
# with robust regression (statsmodels rlm)

winter = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] < 10))]
spring = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] > 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] < 10))]
summer = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed['G2latitude'] > 10))]
fall = all_trimmed.loc[((all_trimmed.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed['G2latitude'] < 10)) | ((all_trimmed.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed['G2latitude'] > 10))]

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed
#esper_sel = fall
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,0], 'A', 0)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[0,0], 'Surface (< 25 m), LIR', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
#esper_sel = fall
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,0], 'C', 0)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[1,0], 'Full Depth, LIR', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed
#esper_sel = fall
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,1], 'B', 1)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[0,1], 'Surface (< 25 m), NN', 1)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
#esper_sel = fall
#esper_sel = esper_sel[esper_sel.dectime >= 2000]
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,1], 'D', 1)
#p1.plot2dhist(esper_sel, esper_type, fig, axs[1,1], 'Full Depth, NN', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.17,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

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
esper_sel = all_trimmed # full depth
#esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist(esper_sel, esper_type, fig, axs[0], 'A', 1)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
#esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
#esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
p1.plot2dhist(esper_sel, esper_type, fig, axs[1], 'B', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.25,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$\n($µmol\;kg^{-1}$)')
###ax.set_ylabel('Measured $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

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
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[0,0], 'A', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[1,0], 'C', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < all_trimmed.surface_depth] # do surface values (< 25 m) only
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[0,1], 'B', 0)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
pts = p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[1,1], 'D', 0)

# adjust figure
ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.4,-0.1)
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)', labelpad=15)

# add single colorbar
fig.colorbar(pts, ax=axs.ravel().tolist(), label='Weight Assigned by RLM')


# %% loop through monte carlo simulation-produced G2talk to do global ensemble mean regression

# create seasons
winter_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed_mc['G2latitude'] > 10)) | ((all_trimmed_mc.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed_mc['G2latitude'] < 10))]
spring_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed_mc['G2latitude'] > 10)) | ((all_trimmed_mc.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed_mc['G2latitude'] < 10))]
summer_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([12, 1, 2])) & (all_trimmed_mc['G2latitude'] < 10)) | ((all_trimmed_mc.datetime.dt.month.isin([6, 7, 8])) & (all_trimmed_mc['G2latitude'] > 10))]
fall_mc = all_trimmed_mc.loc[((all_trimmed_mc.datetime.dt.month.isin([3, 4, 5])) & (all_trimmed_mc['G2latitude'] < 10)) | ((all_trimmed_mc.datetime.dt.month.isin([9, 10, 11])) & (all_trimmed_mc['G2latitude'] > 10))]

# plot surface values and do regular linear regression
mc_sel = all_trimmed_mc
#mc_sel = fall_mc
all_trimmed_mc_surf = mc_sel[mc_sel.G2depth < mc_sel.surface_depth]
x = all_trimmed_mc.dectime
x_surf = all_trimmed_mc_surf.dectime

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
    y_surf_LIR = all_trimmed_mc_surf[str(i)] - all_trimmed_mc_surf.Ensemble_Mean_TA_LIR
    y_LIR = all_trimmed_mc[str(i)] - all_trimmed_mc.Ensemble_Mean_TA_LIR
    y_surf_NN = all_trimmed_mc_surf[str(i)] - all_trimmed_mc_surf.Ensemble_Mean_TA_NN
    y_NN = all_trimmed_mc[str(i)] - all_trimmed_mc.Ensemble_Mean_TA_NN
    
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
#axs[0,0].set_xlim([-0.075, 0.075]) # individual offset
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

plt.xlabel('Slope of Measured $A_{T}$ - ESPER-Estimated $A_{T}$ over Time ($µmol\;kg^{-1}\;yr^{-1}$)')
plt.ylabel('Number of Occurrences')

# %% make box plot graph of transect slopes from mc simulation

# SURFACE LIR
# pull surface values
all_trimmed_mc = all_trimmed_mc[all_trimmed_mc.G2depth < all_trimmed_mc.surface_depth]
# turn into dict with transects as keys
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)
all_slopes_surf = p1.transect_box_plot(trimmed_mc, G2talk_mc, 'LIR')

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
axs[0].text(1.25, 3.8, 'A', fontsize=12)

# make box plot for full depth
axs[1].boxplot(all_slopes_full, vert=True, labels=list(trimmed_mc.keys()))
axs[1].axhline(y=0, color='r', linestyle='--')
axs[1].set_ylim(-5, 5)
axs[1].text(1.25, 3.8, 'B', fontsize=12)
axs[1].tick_params(axis='x', labelrotation=90)

ax.set_ylabel('Slope of Measured $A_{T}$ - ESPER-Estimated $A_{T}$ over Time\n($µmol$ $kg^{-1}$ $yr^{-1}$)')
#ax.set_ylabel('Slope of Measured $A_{T}$ over Time\n($µmol$ $kg^{-1}$ $yr^{-1}$)')

ax.yaxis.set_label_coords(-0.63,0.55)

#%% calculate error: error in TREND, not point
# u_esper = standard deviation in slope across all 16 equations
# u_sample= standard deviation in slopes predicted by mc analysis
# U = summation of u_esper and u_sample in quadrature

# doing this with robust regression

# calculate u_esper
esper_type = 'LIRtalk' # LIR, NN, or Mixed (change separately for u_sample below)
esper_sel = all_trimmed
#esper_sel = fall
esper_sel = esper_sel[esper_sel.G2depth < esper_sel.surface_depth] # do surface values (< 25 m) only
slopes_rlm = np.zeros(16)
slopes_ols = np.zeros(16)

for i in range(0,16):

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

u_esper = slopes_rlm.std() # change if RLM or OLS used for u_esper here

# calculate u_sample
u_sample = slopes_surf_LIR.std() # for SURFACE, LIR
#u_sample = slopes_LIR.std() # for FULL DEPTH, LIR
#u_sample = slopes_surf_NN.std() # for SURFACE, NN
#u_sample = slopes_NN.std() # for FULL DEPTH, NN

U = np.sqrt(u_esper**2 + u_sample**2)
print(U)

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
coeffs = pd.read_csv(filepath + coeffs_file, names=['x', 'TA_S', 'TA_T', 'TA_N', 'TA_O', 'TA_Si'])
#coeffs = pd.read_csv(filepath + 'ESPER_LIR_coeffs_eqn_10.csv', names=['x', 'TA_S', 'TA_T', 'TA_N', 'TA_O', 'TA_Si'])

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
#coeffs_all_trimmed = coeffs_all_trimmed[coeffs_all_trimmed.G2depth < coeffs_all_trimmed.surface_depth]

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
                                                      'G2phosphate', 'G2talk',
                                                      'x'])
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
axs[1].set_ylabel('Slope of Difference between $A_{T}$ Predicted\nby Each Eqn. and Eqn. 16 over Time\n($µmol\;kg^{-1}\;yr^{-1}$)')
axs[1].yaxis.set_label_coords(-0.1,1.1)

axs[0].text(0.4, -0.02, 'A', fontsize=12)
axs[1].text(0.4, -0.02, 'B', fontsize=12)

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
indian = {key: value for key, value in trimmed.items() if key in {'I01', 'I03', 'I05', 'I06S', 'I07', 'I08N', 'I08S', 'I09N', 'I09S', 'I10'}}
indian = pd.concat(indian.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
indian = indian.drop_duplicates(ignore_index=True) # drop duplicates
indian = indian[indian['G2latitude'] > -60] # keep only above -60º latitude

# southern ocean
southern = {key: value for key, value in trimmed.items() if key in {'A12', 'A135', 'I06S', 'I07', 'I08S', 'I09S', 'P14', 'P15', 'P16S', 'P18', 'S04I', 'S04P', 'SR01', 'SR03', 'SR04'}}
southern = pd.concat(southern.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
southern = southern.drop_duplicates(ignore_index=True) # drop duplicates
southern = southern[southern['G2latitude'] <= -60] # keep only below -60º latitude

# arctic ocean
arctic = {key: value for key, value in trimmed.items() if key in {'ARC01E', 'ARC01W', 'A29'}}
arctic = pd.concat(arctic.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
arctic = arctic.drop_duplicates(ignore_index=True) # drop duplicates

# %% break north pacific into three groups
# north pacific cruises: P01, P02, P03, P09, P10, P13, P14, P16N, P18 (above 0º latitude)

# group 1: american cruises
# 276, 277, 279, 280, 286, 296, 299, 301, 302, 304, 306, 307, 345, 1035, 1043,
# 1044, 1045
# any adjustments here? 302 = -12, 307 = +6

# group 2: japanese single or spec
# 272, 461, 468, 495, 497, 502, 504, 505, 567, 598, 602, 609, 1050, 1053, 1060,
# 1063, 1064, 1066, 1067, 1069, 1070, 1071, 1076, 1078, 1079, 1081, 1082, 1083,
# 1086, 1087, 1090, 1092, 1093, 1096, 1098, 1099, 1100, 1101, 2038, 2041, 2047,
# 2050, 2054, 2057, 2062, 2064, 2067, 2075, 2080, 2084, 2087, 2091, 2094, 2096,
# 2097, 2098, 2099, 2102, 2103, 4065, 4066, 4068, 4069, 4071, 4074, 4076, 4078,
# 4081, 4083, 4087, 4089, 5014, 5017
# any adjustments here? 461 = +6, 468 = +13, 567 = +20, 602 = +6, 1067 = +8, 
# 1069 = +6, 1071 = +4, 1078 = +3, 1079 = +4, 1090 = +6, 2091 = +4, 5017 = +3 

# group 3: japanese unknown, up to year 2008
# 406, 407, 408, 412, 439, 440, 459, 515, 517, 545, 546, 547, 548, 549, 550,
# 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565,
# 566, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 579, 580, 581, 582,
# 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
# 598, 599, 600, 601, 603, 604, 605, 606, 607, 608, 1056, 1057, 1058, 1080
# any adjustments here? 459 = +14

# keep only cruise numbers from group one
north_pacific_USA = north_pacific[
    north_pacific['G2cruise'].isin([276, 277, 279, 280, 286, 296, 299, 301,
                                    302, 304, 306, 307, 345, 1035, 1043, 1044,
                                    1045])]

# keep only cruise numbers from group two
north_pacific_JPNnew = north_pacific[
    north_pacific['G2cruise'].isin([272, 461, 468, 495, 497, 502, 504, 505,
                                    567, 598, 602, 609, 1050, 1053, 1060, 1063,
                                    1064, 1066, 1067, 1069, 1070, 1071, 1076,
                                    1078, 1079, 1081, 1082, 1083, 1086, 1087,
                                    1090, 1092, 1093, 1096, 1098, 1099, 1100,
                                    1101, 2038, 2041, 2047, 2050, 2054, 2057,
                                    2062, 2064, 2067, 2075, 2080, 2084, 2087,
                                    2091, 2094, 2096, 2097, 2098, 2099, 2102,
                                    2103, 4065, 4066, 4068, 4069, 4071, 4074,
                                    4076, 4078, 4081, 4083, 4087, 4089, 5014,
                                    5017])]

# keep only cruise numbers from group three
north_pacific_JPNold = north_pacific[
    north_pacific['G2cruise'].isin([406, 407, 408, 412, 439, 440, 459, 515,
                                    517, 545, 546, 547, 548, 549, 550, 551,
                                    552, 553, 554, 555, 556, 557, 558, 559,
                                    560, 561, 562, 563, 564, 565, 566, 568,
                                    569, 570, 571, 572, 573, 574, 575, 576,
                                    577, 579, 580, 581, 582, 583, 584, 585,
                                    586, 587, 588, 589, 590, 591, 592, 593,
                                    594, 595, 596, 597, 598, 599, 600, 601,
                                    603, 604, 605, 606, 607, 608, 1056, 1057,
                                    1058, 1080])]

# %% method change in indian ocean?
# 255, 252, 488, 251, 253, 355, 682, 254, 3034, 3041, 339, 4062, 249, 352, 1046, 250, 353, 3035, 72, 82, 256, 1054
# 255 (1995-96): closed-cell automated potentiometric titration systems
# 252 (1995): closed-cell automated potentiometric titration systems
# 488 (2003-4): nippon gran titration
# 251 (1995): closed-cell automated potentiometric titration systems
# 253 (1995): closed-cell automated potentiometric titration systems, bradshaw & brewer 1988
# 355 (2007): potentiometric, 2007 dickson guide
# 682 (2002): potentiometric
# 254 (1995): closed-cell automated potentiometric titration systems, bradshaw & brewer 1988
# 3034 (2018): HCl titration in close cell (potentiometric)
# 339 (1995): single-point titration (millero, 1993)
# 4062 (2019): spectrophotometry, yao & byrne, 1998
# 249 (1994): closed-cell automated potentiometric titration systems, millero 1993 & 1998
# 352 (2007): closed-cell
# 1046 (2016): two-stage open cell acidimetric, potentiometric
# 250 (1995): closed-cell automated potentiometric titration systems, bradshaw & brewer 1988
# 353 (2007): open-cell potentiometric
# 3035 (2016): two-stage open cell acidimetric, potentiometric
# 72 (2004-5): ??
# 82 (2000): potentiometric titration system similar to that described by brewer 1986
# 256 (1995): closed-cell automated potentiometric titration systems, millero 1993 & 1998
# 1054 (2015-16): spectrophotometry, nippon

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

basin = north_pacific_JPNnew
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

# %% run (or upload) MC simulation FOR EACH REGION to create array of simulated G2talk values (by cruise offset)
#num_mc_runs = 1000
#basins = [north_atlantic, south_atlantic, north_pacific, south_pacific, indian, southern, arctic]
#basin_abbr = ['NA', 'SA', 'NP', 'SP', 'IO', 'SO', 'AO']

#for basin, abbr in zip(basins, basin_abbr):
#    print(abbr)
#    G2talk_mc = p1.create_mc_cruise_offset(basin, num_mc_runs)
#    # export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#    G2talk_mc = pd.DataFrame(G2talk_mc)
#    G2talk_mc.to_csv(filepath + 'G2talk_mc_simulated_' + abbr + '_JPN.csv' , index=False)

G2talk_mc_NA = pd.read_csv(filepath + 'G2talk_mc_simulated_NA_JPN.csv', na_values = -9999)
G2talk_std_NA = G2talk_mc_NA.std(axis=1)

G2talk_mc_SA = pd.read_csv(filepath + 'G2talk_mc_simulated_SA_JPN.csv', na_values = -9999)
G2talk_std_SA = G2talk_mc_SA.std(axis=1)

G2talk_mc_NP = pd.read_csv(filepath + 'G2talk_mc_simulated_NP_JPN.csv', na_values = -9999)
G2talk_std_NP = G2talk_mc_NP.std(axis=1)

G2talk_mc_SP = pd.read_csv(filepath + 'G2talk_mc_simulated_SP_JPN.csv', na_values = -9999)
G2talk_std_SP = G2talk_mc_SP.std(axis=1)

G2talk_mc_IO = pd.read_csv(filepath + 'G2talk_mc_simulated_IO_JPN.csv', na_values = -9999)
G2talk_std_IO = G2talk_mc_IO.std(axis=1)

G2talk_mc_SO = pd.read_csv(filepath + 'G2talk_mc_simulated_SO_JPN.csv', na_values = -9999)
G2talk_std_SO = G2talk_mc_SO.std(axis=1)

G2talk_mc_AO = pd.read_csv(filepath + 'G2talk_mc_simulated_AO_JPN.csv', na_values = -9999)
G2talk_std_AO = G2talk_mc_AO.std(axis=1)

G2talk_mc_regions = [G2talk_mc_NA, G2talk_mc_SA, G2talk_mc_NP, G2talk_mc_SP, G2talk_mc_IO, G2talk_mc_SO, G2talk_mc_AO]
G2talk_std_regions = [G2talk_std_NA, G2talk_std_SA, G2talk_std_NP, G2talk_std_SP, G2talk_std_IO, G2talk_std_SO, G2talk_std_AO]

# %% loop through monte carlo simulation-produced G2talk to do global ensemble mean regression
all_trimmed_basin = north_atlantic
G2talk_mc_basin = G2talk_mc_NA

# plot surface values and do regular linear regression
all_trimmed_basin.reset_index(drop=True, inplace=True)
G2talk_mc_basin.reset_index(drop=True, inplace=True)
all_trimmed_mc_basin = pd.concat([all_trimmed_basin, G2talk_mc_basin], axis=1)
all_trimmed_mc_basin_surf = all_trimmed_mc_basin[all_trimmed_mc_basin.G2depth < all_trimmed_mc_basin.surface_depth]
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

basins = [all_trimmed_mc, north_atlantic, south_atlantic, north_pacific,
          south_pacific, indian, southern, arctic, trimmed_mc['A02'],
          trimmed_mc['A05'], trimmed_mc['A10'], trimmed_mc['A12'],
          trimmed_mc['A135'], trimmed_mc['A16N'], trimmed_mc['A16S'],
          trimmed_mc['A17'], trimmed_mc['A20'], trimmed_mc['A22'],
          trimmed_mc['A25'], trimmed_mc['A29'], trimmed_mc['AR07E'],
          trimmed_mc['AR07W'], trimmed_mc['ARC01E'], trimmed_mc['I05'],
          trimmed_mc['I06'], trimmed_mc['I07'], trimmed_mc['I08N'],
          trimmed_mc['I08S'], trimmed_mc['I09N'], trimmed_mc['I09S'],
          trimmed_mc['I10'], trimmed_mc['P01'], trimmed_mc['P03'],
          trimmed_mc['P06'], trimmed_mc['P09'], trimmed_mc['P10'],
          trimmed_mc['P13'], trimmed_mc['P14'], trimmed_mc['P15'],
          trimmed_mc['P16N'], trimmed_mc['P16S'], trimmed_mc['P17N'],
          trimmed_mc['P18'], trimmed_mc['P21'], trimmed_mc['S04I'],
          trimmed_mc['SR04'], trimmed_mc['S04P'], trimmed_mc['SR01'],
          trimmed_mc['SR03']]


mc_basins = [df_empty, G2talk_mc_NA, G2talk_mc_SA, G2talk_mc_NP, G2talk_mc_SP,
             G2talk_mc_IO, G2talk_mc_SO, G2talk_mc_AO, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty, df_empty, df_empty, df_empty,
             df_empty, df_empty, df_empty]

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
        esper_NN= esper.dropna(subset=['G2talk', 'NNtalk' + str(i+1)])
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
    x_surf_LIR = esper_surf['dectime'].to_numpy()
    y_surf_LIR = del_alk_surf_LIR.to_numpy()
    
    del_alk_surf_NN = esper_surf.loc[:,'G2talk'] - esper_surf.loc[:,'Ensemble_Mean_TA_NN']
    x_surf_NN = esper_surf['dectime'].to_numpy()
    y_surf_NN = del_alk_surf_NN.to_numpy()
    
    del_alk_LIR = esper.loc[:,'G2talk'] - esper.loc[:,'Ensemble_Mean_TA_LIR']
    x_LIR = esper['dectime'].to_numpy()
    y_LIR = del_alk_LIR.to_numpy()
    
    del_alk_NN = esper.loc[:,'G2talk'] - esper.loc[:,'Ensemble_Mean_TA_NN']
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

axs[0].set_ylim([-0.5, 0.5])   
axs[1].set_ylim([-0.5, 0.5])   
axs[1].set_xlim(-0.5, len(basin_trend_LIR) - 0.5) 
 
basin_abbr = ['Global', 'NAO', 'SAO', 'NPO', 'SPO', 'IO', 'SO', 'AO', 'A02', 'A05', 'A10',
              'A12', 'A135', 'A16N', 'A16S', 'A17', 'A20', 'A22', 'A25', 'A29',
              'AR07E', 'AR07W', 'ARC01E', 'I05', 'I06', 'I07', 'I08N', 'I08S',
              'I09N', 'I09S', 'I10', 'P01', 'P03', 'P06', 'P09', 'P10', 'P13',
              'P14', 'P15', 'P16N', 'P16S', 'P17N', 'P18', 'P21', 'S04I',
              'SR04', 'S04P', 'SR01', 'SR03']
axs[1].set_xticks(x, basin_abbr)

ax.set_ylabel('Temporal Trend in Measured $A_{T}$ - ESPER-Estimated $A_{T}$\n($µmol$ $kg^{-1}$ $yr^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.55)
axs[1].tick_params(axis='x', labelrotation=90)
axs[0].text(0, -0.45, '< 25 m', fontsize=12)
axs[1].text(0, -0.45, 'Full Depth', fontsize=12)
axs[1].legend(bbox_to_anchor = (0.6, 0.21), ncol=2)

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
axs[0,0].text(0.67, -1.35, 'A', fontsize=12)

# make box plot for full depth
axs[1,0].boxplot(all_slopes_full_LIR, vert=True, labels=list(trimmed_mc_basin.keys()))
axs[1,0].axhline(y=0, color='r', linestyle='--')
axs[1,0].set_ylim(-1.5, 1.5)
axs[1,0].text(0.67, -1.35, 'C', fontsize=12)

axs[0,1].boxplot(all_slopes_surf_NN, vert=True, labels=list(trimmed_mc_basin_surf.keys()))
axs[0,1].axhline(y=0, color='r', linestyle='--')
axs[0,1].set_ylim(-1.5, 1.5)
axs[0,1].text(0.67, -1.35, 'B', fontsize=12)

# make box plot for full depth
axs[1,1].boxplot(all_slopes_full_NN, vert=True, labels=list(trimmed_mc_basin.keys()))
axs[1,1].axhline(y=0, color='r', linestyle='--')
axs[1,1].set_ylim(-1.5, 1.5)
axs[1,1].text(0.67, -1.35, 'D', fontsize=12)

ax.set_ylabel('Slope of Measured $A_{T}$ - ESPER-Estimated $A_{T}$\nover Time ($µmol$ $kg^{-1}$ $yr^{-1}$)')

ax.yaxis.set_label_coords(-0.63,0.55)
axs[1,0].tick_params(axis='x', labelrotation=90)
axs[1,1].tick_params(axis='x', labelrotation=90)

#%% do calculations on OA amplification for conclusion
esper_sel = espers[espers.G2depth < espers.surface_depth]
avg_TA = esper_sel.G2talk.mean(skipna=True)
print(avg_TA)

ppm_1994 = 358.96 # unc = 0.12, at Mauna Loa, https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.txt
ppm_2024 = 426.91 # unc = 0.12, at Mauna Loa, https://gml.noaa.gov/ccgg/trends/mlo.html

# DIC in 1994
results_1994 = pyco2.sys(par1=avg_TA, par1_type=1, par2=ppm_1994, par2_type=9)
print(results_1994['dic'])
print(results_1994['pH'])

# DIC in 2024 (assuming fixed TA)
results_2024 = pyco2.sys(par1=avg_TA, par1_type=1, par2=ppm_2024, par2_type=9)
print(results_2024['dic'])
print(results_2024['pH'])

# DIC in 2024 (assuming elevated TA from CO2-biotic calcification feedback)
TA_past_30 = 0.080 * 30 # umol/kg, using surface ocean ESPER LIR ∆TA trend
results_2024_bc = pyco2.sys(par1=(avg_TA+TA_past_30), par1_type=1, par2=ppm_2024, par2_type=9)
print(results_2024_bc['dic'])
print(results_2024_bc['pH'])

# dic increase with bc feedback
print(results_2024_bc['dic']/results_2024['dic']*100)

# pH increase with bc feedback (MORE ALKALINE?)
print(results_2024_bc['pH']/results_2024['pH']*100)


























