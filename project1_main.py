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

# set-up

import project1 as p1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#from scipy import interpolate
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm
import cmocean
import cmocean.cm as cmo

filepath = '/Users/Reese/Documents/Research Projects/project1/data/' # where GLODAP data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)
input_mc_cruise_file = 'G2talk_mc_simulated.csv' # MC (per cruise) simulated data
input_mc_individual_file = 'G2talk_mc_individual_simulated.csv' # MC (per cruise) simulated data

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
 
# %%step 3: upload ESPERs outputs to here
espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv') # to do the normal ESPER
#espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA_GO-SHIP_LIR.csv') # to do the GO-SHIP trained ESPER
espers['datetime'] = pd.to_datetime(espers['datetime']) # recast datetime as datetime data type

# %% use KL divergence to determine which equations predict best (lower KL divergence = two datasets are closer)
kl_div = p1.kl_divergence(espers)
#kl_div.to_csv('kl_div.csv')

# %% calculate ensemble mean TA for each data point
espers = p1.ensemble_mean(espers)

# %% trim GO-SHIP + associated cruises to pick out data points on the standard transect
trimmed = p1.trim_go_ship(espers, go_ship_cruise_nums_2023)
all_trimmed = pd.concat(trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
all_trimmed = all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

#%% calculate average ESPERs coefficients

# read in coefficients extracted from MATLAB (already averaged across all 16 equations)
coeffs = pd.read_csv(filepath + 'ESPER_LIR_coeffs.csv', names=['x', 'TA_S', 'TA_T', 'TA_N', 'TA_O', 'TA_Si'])

# attach G2cruise and G2station columns so trimming will work
coeffs.insert(0, 'G2cruise', espers.G2cruise)
coeffs.insert(1, 'G2station', espers.G2station)
coeffs.insert(2, 'Ensemble_Mean_TA_LIR', espers.Ensemble_Mean_TA_LIR)

# trim to pick out points on standard transect
coeffs_trimmed = p1.trim_go_ship(coeffs, go_ship_cruise_nums_2023)
coeffs_all_trimmed = pd.concat(coeffs_trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
coeffs_all_trimmed = coeffs_all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

# average across all samples
coeffs_all_trimmed = coeffs_all_trimmed.drop(columns=['G2cruise', 'G2station', 'Ensemble_Mean_TA_LIR', 'x'])
avg_coeffs = coeffs_all_trimmed.mean(axis=0)

# %% run (or upload) MC simulation to create array of simulated G2talk values (by cruise offset)
#num_mc_runs = 1000
#G2talk_mc = p1.create_mc_cruise_offset(all_trimmed, num_mc_runs)
# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#G2talk_mc = pd.DataFrame(G2talk_mc)
#G2talk_mc.to_csv(filepath + input_mc_cruise_file, index=False)

G2talk_mc = pd.read_csv(filepath + input_mc_cruise_file, na_values = -9999)
G2talk_std = G2talk_mc.std(axis=1)

# %% run (or upload) MC simulation to create array of simulated G2talk values (individual point offset)
#num_mc_runs = 1000
#G2talk_mc = p1.create_mc_individual_offset(all_trimmed, num_mc_runs)
# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#G2talk_mc = pd.DataFrame(G2talk_mc)
#G2talk_mc.to_csv(filepath + input_mc_individual_file, index=False)

G2talk_mc = pd.read_csv(filepath + input_mc_individual_file, na_values = -9999)

#%% show layered histograms of distance from ensemble mean for each equation
# not sure if this is necessary or useful
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

# calculate average and std of ∆TA calculated with LIR
LIR_mean = np.nanmean(hist_data)
LIR_std = np.nanstd(hist_data)

# plot ESPER NN results
hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['NNtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_NN']

ax2.hist(hist_data, bins=np.arange(np.nanmin(hist_data.flatten()), np.nanmax(hist_data.flatten()) + binwidth, binwidth), histtype ='step', stacked=True,fill=False, label=labels)
ax2.set_xlim([-5,5])
ax2.set_ylim([0,600000])
ax2.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

# calculate average and std of ∆TA calculated with NN
NN_mean = np.nanmean(hist_data)
NN_std = np.nanstd(hist_data)

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
surface = all_trimmed[all_trimmed.G2depth < 25]
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

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6.5,4), dpi=200, sharex=True, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,0], 'A', 0)


# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,0], 'C', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
p1.plot2dhist(esper_sel, esper_type, fig, axs[0,1], 'B', 1)

# full ocean NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot2dhist(esper_sel, esper_type, fig, axs[1,1], 'D', 1)

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
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
p1.plot2dhist(esper_sel, esper_type, fig, axs[0], 'A', 2)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot2dhist(esper_sel, esper_type, fig, axs[1], 'B', 1)

ax.set_xlabel('Year')
ax.xaxis.set_label_coords(0.25,-0.65) # for 2d histogram
ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$\n($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.28)

#%% data points colored by weight assigned by robust regression (statsmodels rlm)
# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7,4.5), dpi=400, sharex=True, sharey=True)
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[0,0], 'A', 0)

# full ocean LIR
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
p1.plot_rlm_weights(esper_sel, esper_type, fig, axs[1,0], 'C', 0)

# surface NN
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
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

# plot surface values and do regular linear regression
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
all_trimmed_mc_surf = all_trimmed_mc[all_trimmed_mc.G2depth < 25]
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

#axs[0,0].hist(slopes_surf_LIR, bins=100)
axs[0,0].set_xlim([-0.15, 0.15]) # per cruise offset
axs[0,0].set_xlim([-0.07, 0.07]) # individual offset
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
plt.ylabel('Count')

# %% make box plot graph of transect slopes from mc simulation

# SURFACE LIR
# pull surface values
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
all_trimmed_mc = all_trimmed_mc[all_trimmed_mc.G2depth < 25]
# turn into dict with transects as keys
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)
all_slopes_surf = p1.transect_box_plot(trimmed_mc, G2talk_mc, 'LIR')

# FULL-OCEAN LIR
# pull all values
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
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

ax.yaxis.set_label_coords(-0.63,0.55)

#%% calculate error: error in TREND, not point
# u_esper = standard deviation in slope across all 16 equations
# u_sample= standard deviation in slopes predicted by mc analysis
# U = summation of u_esper and u_sample in quadrature

# doing this with robust regression

# calculate u_esper
esper_type = 'LIRtalk' # LIR, NN, or Mixed (change separately for u_sample below)
esper_sel = all_trimmed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
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

U = np.sqrt(u_esper**2 + u_sample**2)

# %% plot global ensemble mean regression for each GO-SHIP transect

# plot surface values and do linear regression
slopes = np.zeros(len(trimmed.keys()))
pvalues = np.zeros(len(trimmed.keys()))
i = 0

for keys in trimmed:
    if len(trimmed[keys]) > 0: # if dictionary key is not empty 
        transect = trimmed[keys]
        surface = transect[transect.G2depth < 25]
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

#%% plot ∆TA vs. predictor variables (robust regression shown)
# salinity and nitrate in full and deep ocean

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,4), dpi=200, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR vs. salinity
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
var_name = 'G2salinity'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,0], 'A', 2)

# full ocean LIR vs. salinity
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
var_name = 'G2salinity'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,0], 'C', 2)
axs[1,0].set_xlabel('Salinity (PSU)')

# surface NN vs. nitrate
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
var_name = 'G2nitrate'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,1], 'B', 1)

# full ocean NN vs. nitrate
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed # full depth
var_name = 'G2nitrate'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,1], 'D', 1)
axs[1,1].set_xlabel('Nitrate ($µmol\;kg^{-1}$)')

ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.55)

#%% plot ∆TA vs. predictor variables (robust regression shown)
# surface ocean only --> salinity top row, left before 2005 and right after 2005
# surface ocean only --> nitrate bottom row, left before 2005 and right after 2005

# make figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8,4), dpi=200, sharey=True, layout='constrained')
fig.add_subplot(111,frameon=False)
ax = fig.gca()
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# surface LIR vs. salinity (pre-2005)
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.dectime < 2005] # pre-2005 only
var_name = 'G2salinity'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,0], 'A', 2)

# surface LIR vs. salinity (post-2005)
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.dectime >= 2005] # post-2005 only
var_name = 'G2salinity'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,0], 'C', 2)
axs[1,0].set_xlabel('Salinity (PSU)')

# surface NN vs. nitrate (pre-2005)
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.dectime < 2005] # pre-2005 only
var_name = 'G2nitrate'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[0,1], 'B', 1)

# surface NN vs. nitrate (post-2005)
esper_type = 'Ensemble_Mean_TA_NN' # LIR, NN, or Mixed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
esper_sel = esper_sel[esper_sel.dectime >= 2005] # post-2005 only
var_name = 'G2nitrate'
p1.compare_TA_var(var_name, esper_sel, esper_type, fig, axs[1,1], 'D', 1)
axs[1,1].set_xlabel('Nitrate ($µmol\;kg^{-1}$)')

ax.set_ylabel('Measured $A_{T}$ - ESPER-Estimated $A_{T}$ ($µmol\;kg^{-1}$)')
ax.yaxis.set_label_coords(-0.62,0.55)



