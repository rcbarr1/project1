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

#%% calculate error for each point? this is a work in progress, not sure it makes sense logically
all_trimmed['error_LIR'] = np.sqrt(all_trimmed.Ensemble_Std_TA_LIR**2 + G2talk_std**2)
all_trimmed['error_NN'] = np.sqrt(all_trimmed.Ensemble_Std_TA_NN**2 + G2talk_std**2)
all_trimmed['error_Mixed'] = np.sqrt(all_trimmed.Ensemble_Std_TA_Mixed**2 + G2talk_std**2)

#%% show layered histograms of distance from ensemble mean for each equation
# not sure if this is necessary or useful
fig = plt.figure(figsize=(7,5), dpi=200)
ax = fig.gca()

hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['LIRtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_LIR']

n_bins = 400
labels = ['Eqn. 1', 'Eqn. 2', 'Eqn. 3', 'Eqn. 4', 'Eqn. 5', 'Eqn. 6', 'Eqn. 7',
          'Eqn. 8', 'Eqn. 9', 'Eqn. 10', 'Eqn. 11', 'Eqn. 12', 'Eqn. 13',
          'Eqn. 14', 'Eqn. 15', 'Eqn. 16']
ax.hist(hist_data, n_bins, histtype ='step', stacked=True,fill=False, label=labels)
ax.set_xlim([-5,5])
ax.set_ylim([0,600000])
ax.set_xlabel('ESPER LIR-Estimated TA - LIR Ensemble Mean TA ($µmol\;kg^{-1}$)')
ax.set_ylabel('Count')
#ax.legend()
fig.text(0.14, 0.825, 'A', fontsize=14)

fig = plt.figure(figsize=(7,5), dpi=200)
ax = fig.gca()
hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['NNtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_NN']

ax.hist(hist_data, n_bins, histtype ='step', stacked=True,fill=False, label=labels)
ax.set_xlim([-5,5])
ax.set_ylim([0,600000])
ax.set_xlabel('ESPER NN-Estimated TA - NN Ensemble Mean TA ($µmol\;kg^{-1}$)')
ax.set_ylabel('Count')
#ax.legend()
fig.text(0.14, 0.825, 'B', fontsize=14)

fig = plt.figure(figsize=(7,5), dpi=200)
ax = fig.gca()
hist_data = np.zeros((all_trimmed.shape[0], 16))
for i in range(0,16):
    hist_data[:,i] = all_trimmed['Mtalk' + str(i+1)] - all_trimmed['Ensemble_Mean_TA_Mixed']

ax.hist(hist_data, n_bins, histtype ='step', stacked=True,fill=False, label=labels)
ax.set_xlim([-5,5])
ax.set_ylim([0,600000])
ax.set_xlabel('ESPER Mixed-Estimated TA - Mixed Ensemble Mean TA ($µmol\;kg^{-1}$)')
ax.set_ylabel('Count')
ax.legend()
fig.text(0.14, 0.825, 'C', fontsize=14)

# %% start data visualization

# organize data by decimal time
espers = espers.sort_values(by=['dectime'],ascending=True)
# %% USEFUL FOR VISUALIZING DATA LOCATIONS
# set up map
# atlantic-centered view
#fig = plt.figure(figsize=(6.2,4.1))
fig = plt.figure(figsize=(5,4), dpi=200)
ax = plt.axes(projection=ccrs.PlateCarree())
# pacific-centered view
#fig = plt.figure(figsize=(6.3,4.1))
#ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
#ax = plt.axes(projection=ccrs.Orthographic(0,90)) # arctic-centered view (turn off "extent" variable)
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
#ax.set_title('North Atlantic Coverage of TA (GLODAPv2.2023)')
#extent = [5, 15, -52.5, -52]
#extent = [-30, 30, -80, 10]
extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get data from glodap
#lon = espers.G2longitude
#lat = espers.G2latitude
#plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',color='C0',s=1)

# or, plot all trimmed transects
for key in trimmed:
    df = trimmed[key]
    lon = df.G2longitude
    lat = df.G2latitude
    plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10,color='steelblue',alpha=0.5)

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
to_plot = to_plot.sort_values('abs_del_alk',axis=0,ascending=False)

# create colormap
cmap = cmocean.tools.crop(cmo.balance, to_plot.del_alk.min(), to_plot.del_alk.max(), 0)

# plot data
pts = ax.scatter(to_plot.G2longitude,to_plot.G2latitude,transform=ccrs.PlateCarree(),s=30,c=to_plot.del_alk,cmap=cmap, alpha=0.15,edgecolors='none')
plt.colorbar(pts, ax=ax, label='Measured TA - ESPER-Estimated TA \n($µmol\;kg^{-1}$)')

# %% plot global ensemble mean regression for all trimmed GO-SHIP 

# plot surface values and do regular linear regression
surface = all_trimmed[all_trimmed.G2depth < 25]
#surface = all_trimmed
x = surface.dectime
y = surface.G2talk - surface.Ensemble_Mean_TA_LIR

slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')

# make plot
fig = plt.figure(figsize=(7,5), dpi=200)
ax = plt.gca()
plt.scatter(surface.dectime,y, c='mediumaquamarine', s=10, alpha=0.8)
fig.text(0.52, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
fig.text(0.52, 0.78, '$p-value={:.4e}$'.format(pvalue), fontsize=14)
#fig.text(0.15, 0.83, 'A', fontsize=14)
x = np.linspace(surface.dectime.min(), surface.dectime.max(), num=100)
ax.plot(x, intercept + slope * x, color="firebrick", lw=2, ls='--');
#ax.set_title('Difference in Measured and ESPER LIR-Predicted TA along GO-SHIP Transects (< 25 m)')
#ax.set_title('Difference in Measured and ESPER LIR-Predicted TA along GO-SHIP Transects')
ax.set_ylabel('Measured TA - ESPER-Estimated TA \n($µmol\;kg^{-1}$)',fontsize=12)
ax.set_ylim(-70,70) # zoomed in
#ax.set_ylim(-375,225) # to see all data
ax.set_xlim(surface.dectime.min(),surface.dectime.max())

#%% 2D histogram for global ensemble mean regression for all trimmed GO-SHIP

# plot surface values and do regular linear regression
surface = all_trimmed[all_trimmed.G2depth < 25]
#surface = all_trimmed
x = surface.dectime
y = surface.G2talk - surface.Ensemble_Mean_TA_LIR

# plot histogram
fig = plt.figure(figsize=(7,5), dpi=200)
ax = plt.gca()
h = ax.hist2d(x, y, bins=150, norm='log', cmap=cmo.matter)

# add regression line
slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
ax.plot(x, intercept + slope * x, color="maroon", lw=1, ls='--');
fig.text(0.42, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=12)
fig.text(0.42, 0.78, '$p-value={:.4e}$'.format(pvalue), fontsize=12)

# add plot elements
ax.set_ylabel('Measured TA - ESPER-Estimated TA ($µmol\;kg^{-1}$)',fontsize=12)
ax.set_xlabel('Year',fontsize=12)
ax.set_ylim(-80,80) # zoomed in
plt.colorbar(h[3],label='Count')

# %% loop through monte carlo simulation-produced G2talk to do global ensemble mean regression

# plot surface values and do regular linear regression
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
all_trimmed_mc = all_trimmed_mc[all_trimmed_mc.G2depth < 25]
x = all_trimmed_mc.dectime

# preallocate arrays for storing slope and p-values
slopes = np.zeros(G2talk_mc.shape[1])
pvalues = np.zeros(G2talk_mc.shape[1])

for i in range(0,G2talk_mc.shape[1]): 
#for i in range(0,2):
    y = all_trimmed_mc[str(i)] - all_trimmed_mc.Ensemble_Mean_TA_LIR # this works for per cruise offsets, change ESPER method here
    #y = all_trimmed_mc[str(i)] - all_trimmed_mc.Ensemble_Mean_TA_LIR # this works for individual offsets, change ESPER method here

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
    
    slopes[i] = slope
    pvalues[i] = pvalue

# make histogram of slopes
fig = plt.figure(figsize=(6,4), dpi=200)
ax = plt.gca()
plt.hist(slopes, bins=100)
mu = slopes.mean()
sigma = slopes.std()
#ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions\n(1000 runs, normally-distributed error of 2 µmol/kg added to each cruise)')
ax.set_xlabel('Slope of Measured TA - ESPER-Estimated TA over Time ($µmol\;kg^{-1}$)')
ax.set_ylabel('Count')
fig.text(0.14, 0.825, 'A', fontsize=14)
fig.text(0.71, 0.825, '$\mu={:.4f}$'.format(mu), fontsize=14)
fig.text(0.71, 0.755, '$\sigma={:.4f}$'.format(sigma), fontsize=14)
ax.set_xlim([-0.15, 0.15]) # per cruise offset
ax.set_xlim([-0.07, 0.07]) # individual offset
ax.set_ylim([0, 37])

# scatter slopes and p values to see if any <0 are significant
#fig = plt.figure(figsize=(9,6))
#ax = plt.gca()
#plt.scatter(slopes,pvalues)
#plt.axhline(y = 0.05, color = 'r', linestyle = '--') 
#ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions & Associated p-Values\n(1000 runs, normally-distributed error of 2 µmol/kg added to each cruise)')
#ax.set_xlabel('Slope of Measured TA - ESPER-Estimated TA over Time ($µmol\;kg^{-1}$)')
#ax.set_ylabel('p-Value')
#ax.set_ylim([-0.05, 1])

# %% make box plot graph of transect slopes from mc simulation

# pull surface values
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
all_trimmed_mc = all_trimmed_mc[all_trimmed_mc.G2depth < 25]

# turn into dict with transects as keys
trimmed_mc = p1.trim_go_ship(all_trimmed_mc, go_ship_cruise_nums_2023)

# remove transects that have 0 or 1 repeats
# I01, P02_J, P03, P09, P10, P13, P17E, ARC01W, MED01
del trimmed_mc['I01']
del trimmed_mc['P02_J']
del trimmed_mc['P03']
del trimmed_mc['P09']
del trimmed_mc['P10']
del trimmed_mc['P13']
del trimmed_mc['P17E']
del trimmed_mc['ARC01W']
del trimmed_mc['MED01']

# get rid of empty dict entries
del_keys = []
for key in trimmed_mc:
    if trimmed_mc[key].empty:
        del_keys.append(key)

for key in del_keys:
    del trimmed_mc[key]

# preallocate np array to save slopes & p-values data in
all_slopes = [np.zeros(G2talk_mc.shape[1]) for i in range(0, len(trimmed_mc.keys()))] # number of transects by number of mc simulations
all_sig_slopes = [np.zeros(G2talk_mc.shape[1]) for i in range(0, len(trimmed_mc.keys()))] # number of transects by number of mc simulations (only to store statistically significant slopes)
all_pvalues = [np.zeros(G2talk_mc.shape[1]) for i in range(0, len(trimmed_mc.keys()))] # number of transects by number of mc simulations


# loop through transects
j = 0
for key in trimmed_mc:
    transect = trimmed_mc[key]
    x = transect.dectime
    
    transect_mc = transect.iloc[:,116:] # pull only mc simulated G2talk for this transect
    
    # calculate slope and p value for each mc run
    slopes = np.zeros(transect_mc.shape[1])
    pvalues = np.zeros(transect_mc.shape[1])
    sig_slopes = np.zeros(transect_mc.shape[1])
    
    # loop through mc simulations for this transect
    for i in range(0,transect_mc.shape[1]):
        y = transect_mc.iloc[:,i] - transect.Ensemble_Mean_TA_LIR # change ESPER method here
    
        slope, _, _, pvalue, _ = stats.linregress(x, y, alternative='two-sided')
        slopes[i] = slope
        pvalues[i] = pvalue
        if pvalue < 0.05:
            sig_slopes[i] = slope
        else:
            sig_slopes[i] = np.nan
    
    all_slopes[j] = slopes
    all_pvalues[j] = pvalues
    all_sig_slopes[j] = sig_slopes
    
    j += 1

# make box plot for slope
fig = plt.figure(figsize=(15,7), dpi = 200)
ax = plt.gca()
plt.boxplot(all_slopes, vert=True, labels=list(trimmed_mc.keys()))
plt.axhline(y=0, color='r', linestyle='--')
plt.xticks(rotation=90)
ax.set_ylabel('Slope of Measured TA - ESPER-Estimated TA over Time ($µmol\;kg^{-1}$)')
#ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions by Transect\n(1000 runs, normally-distributed error of 2 µmol/kg added to each cruise)')
ax.set_ylim(-5, 5)
fig.text(0.132, 0.84, 'A', fontsize=14)

# make box plot for p values
#fig = plt.figure(figsize=(15,7))
#ax = plt.gca()
#plt.boxplot(all_pvalues, vert=True, labels=list(trimmed_mc.keys()))
#plt.axhline(y=0.05, color='r', linestyle='--')
#plt.xticks(rotation=90)
#ax.set_ylabel('P-Value for Measured TA - ESPER-Estimated TA over Time ($µmol\;kg^{-1}$)')
#ax.set_title('Monte Carlo Simulation: P-Values of Linear Regressions by Transect\n(1000 runs, normally-distributed error of 2 µmol/kg added to each cruise)')
#ax.set_ylim(0, 1)

#%% make box plot for only slopes that have significant p-value
# filter to get rid of NaNs
all_sig_slopes_arr = np.array(all_sig_slopes)
mask = ~np.isnan(all_sig_slopes_arr)
filtered_data = [d[m] for d, m in zip(all_sig_slopes_arr, mask)]
# plot box plot
fig = plt.figure(figsize=(15,7))
ax = plt.gca()
plt.boxplot(filtered_data, vert=True, labels=list(trimmed_mc.keys()))
plt.axhline(y=0, color='r', linestyle='--')
plt.xticks(rotation=90)
ax.set_ylabel('Slope of Measured TA - ESPER-Estimated TA over Time ($µmol\;kg^{-1}$)')
ax.set_title('Monte Carlo Simulation: Significant Slopes of Linear Regressions by Transect\n(1000 runs, normally-distributed error of 2 µmol/kg added to each point)')
#ax.set_ylim(-2, 2)

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
        ax.set_title(str(keys) + ': Difference in Measured and ESPER-Predicted TA (< 25 m)')
        ax.set_ylabel('Measured TA - ESPER-Estimated TA ($µmol\;kg^{-1}$)')
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
        ax.set_title(str(keys) + ': Difference in Measured and ESPER-Predicted TA (Full Depth)')
        ax.set_ylabel('Measured TA - ESPER-Estimated TA ($µmol\;kg^{-1}$)')
        ax.set_ylim(-70,70)
        ax.set_xlim(all_trimmed.datetime.min(),all_trimmed.datetime.max())
        i += 1
    else:
        slopes[i] = np.nan
        pvalues[i] = np.nan
        i += 1

# %% do global mean ensemble robust regression (statsmodels rlm)
# SET ESPER ROUTINE HERE
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed

# subset if desired
esper_sel = all_trimmed
#esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
 
# sort by time
esper_sel = esper_sel.sort_values(by=['dectime'],ascending=True)

# calculate the difference in TA betwen GLODAP and ESPERS, store for regression
del_alk = esper_sel.loc[:,'G2talk'] - esper_sel.loc[:,esper_type]
x = esper_sel['dectime'].to_numpy()
y = del_alk.to_numpy()

# fit model and print summary
x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

ols_model = sm.OLS(y, x_model)
ols_results = ols_model.fit()

print(rlm_results.params)
print(rlm_results.bse)
print(
    rlm_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(rlm_results.params))]
    )
)

print(ols_results.params)
print(ols_results.bse)
print(
    ols_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(ols_results.params))]
    )
)

# make figure
fig = plt.figure(figsize=(9.3,5),dpi=400)
ax = fig.gca()
#ax.plot(x[:,1], y, 'o', label='data', alpha = 0.3, color='lightblue') # for scatterplot
h = ax.hist2d(x, y, bins=150, norm='log', cmap=cmo.matter) # for 2d histogram
ax.plot(x_model[:,1], rlm_results.fittedvalues, lw=1, ls='-', color='black', label='RLM')
ax.plot(x_model[:,1], ols_results.fittedvalues, lw=1, ls='-', color='gainsboro', label='OLS')
ax.set_ylim([-80, 80])
ax.set_xlabel('Year')
ax.set_ylabel('Measured TA - ESPER Estimated TA ($µmol\;kg^{-1}$)')
#legend = ax.legend(loc='lower left')
plt.colorbar(h[3],label='Count')

# print equations & p values for each regression type
fig.text(0.265, 0.83, 'OLS: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(ols_results.params[1],ols_results.params[0],ols_results.pvalues[1]), fontsize=12)
fig.text(0.265, 0.78, 'RLM: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(rlm_results.params[1],rlm_results.params[0],rlm_results.pvalues[1]), fontsize=12)
fig.text(0.14, 0.83, 'B', fontsize=12)

#%% plot ∆TA vs. salinity
# SET ESPER ROUTINE HERE
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed

# subset if desired
esper_sel = all_trimmed
esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only
esper_sel = all_trimmed[all_trimmed.G2salinity > 32] # do surface values (< 25 m) only
 
# sort by time
esper_sel = esper_sel.sort_values(by=['G2salinity'],ascending=True)


# calculate the difference in TA betwen GLODAP and ESPERS, store for regression
del_alk = esper_sel.loc[:,'G2talk'] - esper_sel.loc[:,esper_type]
x = esper_sel['G2salinity'].to_numpy()
y = del_alk.to_numpy()

# fit model and print summary
x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

ols_model = sm.OLS(y, x_model)
ols_results = ols_model.fit()

print(rlm_results.params)
print(rlm_results.bse)
print(
    rlm_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(rlm_results.params))]
    )
)

print(ols_results.params)
print(ols_results.bse)
print(
    ols_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(ols_results.params))]
    )
)

# make figure
fig = plt.figure(figsize=(9.3,5),dpi=400)
ax = fig.gca()
#ax.plot(x[:,1], y, 'o', label='data', alpha = 0.3, color='lightblue') # for scatterplot
h = ax.hist2d(x, y, bins=150, norm='log', cmap=cmo.matter) # for 2d histogram
ax.plot(x_model[:,1], rlm_results.fittedvalues, lw=1, ls='-', color='black', label='RLM')
ax.plot(x_model[:,1], ols_results.fittedvalues, lw=1, ls='-', color='gainsboro', label='OLS')
ax.set_ylim([-80, 80])
ax.set_xlabel('Salinity (PSU)')
ax.set_ylabel('Measured TA - ESPER Estimated TA ($µmol\;kg^{-1}$)')
legend = ax.legend(loc='lower left')
plt.colorbar(h[3],label='Count')

# print equations & p values for each regression type
fig.text(0.265, 0.83, 'OLS: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(ols_results.params[1],ols_results.params[0],ols_results.pvalues[1]), fontsize=12)
fig.text(0.265, 0.78, 'RLM: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(rlm_results.params[1],rlm_results.params[0],rlm_results.pvalues[1]), fontsize=12)
#fig.text(0.14, 0.83, 'B', fontsize=12)

#%% plot ∆TA vs. nitrate
# SET ESPER ROUTINE HERE
esper_type = 'Ensemble_Mean_TA_LIR' # LIR, NN, or Mixed

# subset if desired
esper_sel = all_trimmed
#esper_sel = all_trimmed[all_trimmed.G2depth < 25] # do surface values (< 25 m) only

# drop NaNs in nitrate
esper_sel = esper_sel.dropna(subset=['G2nitrate'])
 
# sort by time
esper_sel = esper_sel.sort_values(by=['G2nitrate'], ascending=True)

# calculate the difference in TA betwen GLODAP and ESPERS, store for regression
del_alk = esper_sel.loc[:,'G2talk'] - esper_sel.loc[:,esper_type]
x = esper_sel['G2nitrate'].to_numpy()
y = del_alk.to_numpy()

# fit model and print summary
x_model = sm.add_constant(x) # this is required in statsmodels to get an intercept
rlm_model = sm.RLM(y, x_model, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()

ols_model = sm.OLS(y, x_model)
ols_results = ols_model.fit()

print(rlm_results.params)
print(rlm_results.bse)
print(
    rlm_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(rlm_results.params))]
    )
)

print(ols_results.params)
print(ols_results.bse)
print(
    ols_results.summary(
        yname="y", xname=["var_%d" % i for i in range(len(ols_results.params))]
    )
)

# make figure
fig = plt.figure(figsize=(9.3,5),dpi=400)
ax = fig.gca()
#ax.plot(x[:,1], y, 'o', label='data', alpha = 0.3, color='lightblue') # for scatterplot
h = ax.hist2d(x, y, bins=150, norm='log', cmap=cmo.matter) # for 2d histogram
ax.plot(x_model[:,1], rlm_results.fittedvalues, lw=1, ls='-', color='black', label='RLM')
ax.plot(x_model[:,1], ols_results.fittedvalues, lw=1, ls='-', color='gainsboro', label='OLS')
ax.set_ylim([-80, 80])
ax.set_xlabel('Nitrate ($µmol\;kg^{-1}$)')
ax.set_ylabel('Measured TA - ESPER Estimated TA ($µmol\;kg^{-1}$)')
#legend = ax.legend(loc='lower left')
plt.colorbar(h[3],label='Count')

# print equations & p values for each regression type
fig.text(0.265, 0.83, 'OLS: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(ols_results.params[1],ols_results.params[0],ols_results.pvalues[1]), fontsize=12)
fig.text(0.265, 0.78, 'RLM: $y={:.4f}x {:+.4f}$, p-value$={:.3e}$'.format(rlm_results.params[1],rlm_results.params[0],rlm_results.pvalues[1]), fontsize=12)
#fig.text(0.14, 0.83, 'B', fontsize=12)






