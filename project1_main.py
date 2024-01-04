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
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#import cmocean

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

# %%step 3: upload ESPERs outputs to here
espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv')
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

# %% run (or upload) MC simulation to create array of simulated G2talk values (individual point offset)
#num_mc_runs = 1000
#G2talk_mc = p1.create_mc_individual_offset(all_trimmed, num_mc_runs)
# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
#G2talk_mc = pd.DataFrame(G2talk_mc)
#G2talk_mc.to_csv(filepath + input_mc_individual_file, index=False)

G2talk_mc = pd.read_csv(filepath + input_mc_individual_file, na_values = -9999)

# %% start data visualization

# organize data by decimal time
espers = espers.sort_values(by=['dectime'],ascending=True)
# %% USEFUL FOR VISUALIZING DATA LOCATIONS
# set up map
# atlantic-centered view
fig = plt.figure(figsize=(6.2,4.1))
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
    plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=1,color='steelblue')

# plot one cruise colored
df = trimmed['SR04']
lon = df.G2longitude
lat = df.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=1,color='crimson')

# %% plot global ensemble mean regression for all trimmed GO-SHIP 

# plot surface values and do regular linear regression
surface = all_trimmed[all_trimmed.G2depth < 25]
#surface = all_trimmed
x = surface.dectime
y = surface.G2talk - surface.Ensemble_Mean_TA

slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')

# make plot
fig = plt.figure(figsize=(9,6))
ax = plt.gca()
plt.scatter(surface.datetime,y,s=1)
fig.text(0.6, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
fig.text(0.6, 0.78, '$p-value={:.4e}$'.format(pvalue), fontsize=14)
ax.plot(surface.datetime, intercept + slope * surface.dectime, color="r", lw=1);
ax.set_title('Difference in Measured and ESPER-Predicted TA along GO-SHIP Transects (< 25 m)')
ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
ax.set_ylim(-70,70)
ax.set_xlim(all_trimmed.datetime.min(),all_trimmed.datetime.max())

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
    y = all_trimmed_mc[str(i)] - all_trimmed_mc.Ensemble_Mean_TA # this works for per cruise offsets
    #y = all_trimmed_mc[i] - all_trimmed_mc.Ensemble_Mean_TA # this works for individual offsets

    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
    
    slopes[i] = slope
    pvalues[i] = pvalue

# make histogram of slopes
fig = plt.figure(figsize=(9,6))
ax = plt.gca()
plt.hist(slopes, bins=100)
ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions\n(1000 runs, normally-distributed error of 2 µmol/kg added to each point)')
ax.set_xlabel('Slope of Measured TA - ESPER-Estimated TA over Time ($mmol\;kg^{-1}$)')
ax.set_ylabel('Count')

# scatter slopes and p values to see if any <0 are significant
fig = plt.figure(figsize=(9,6))
ax = plt.gca()
plt.scatter(slopes,pvalues)
plt.axhline(y = 0.05, color = 'r', linestyle = '--') 
ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions & Associated p-Values\n(1000 runs, normally-distributed error of 2 µmol/kg added to each point)')
ax.set_xlabel('Slope of Measured TA - ESPER-Estimated TA over Time ($mmol\;kg^{-1}$)')
ax.set_ylabel('p-Value')

# %% make box plot graph of transect slopes from mc simulation

# pull surface values
all_trimmed_mc = pd.concat([all_trimmed, G2talk_mc], axis=1)
#all_trimmed_mc = all_trimmed_mc[all_trimmed_mc.G2depth < 25]

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

# %% preallocate np array to save slopes & p-values data in
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
        y = transect_mc.iloc[:,i] - transect.Ensemble_Mean_TA
    
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
#fig = plt.figure(figsize=(15,7))
#ax = plt.gca()
#plt.boxplot(all_slopes, vert=True, labels=list(trimmed_mc.keys()))
#plt.axhline(y=0, color='r', linestyle='--')
#plt.xticks(rotation=90)
#ax.set_ylabel('Slope of Measured TA - ESPER-Estimated TA over Time ($mmol\;kg^{-1}$)')
#ax.set_title('Monte Carlo Simulation: Slopes of Linear Regressions by Transect\n(1000 runs, normally-distributed error of 2 µmol/kg added to each cruise)')
#ax.set_ylim(-2, 2)

# make box plot for p values
#fig = plt.figure(figsize=(15,7))
#ax = plt.gca()
#plt.boxplot(all_pvalues, vert=True, labels=list(trimmed_mc.keys()))
#plt.axhline(y=0.05, color='r', linestyle='--')
#plt.xticks(rotation=90)
#ax.set_ylabel('P-Value for Measured TA - ESPER-Estimated TA over Time ($mmol\;kg^{-1}$)')
#ax.set_title('Monte Carlo Simulation: P-Values of Linear Regressions by Transect\n(1000 runs, normally-distributed error of 2 µmol/kg added to each point)')
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
ax.set_ylabel(' Slope of Measured TA - ESPER-Estimated TA over Time ($mmol\;kg^{-1}$)')
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
        ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
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
        ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
        ax.set_ylim(-70,70)
        ax.set_xlim(all_trimmed.datetime.min(),all_trimmed.datetime.max())
        i += 1
    else:
        slopes[i] = np.nan
        pvalues[i] = np.nan
        i += 1
    
# %% do robust regression to take care of outliers

# SET ESPER ROUTINE HERE
esper_type = 'LIR' # LIR, NN, or M
equation_num = 1 # 1 through 16

# subset if desired
esper_sel = espers
#esper_sel = trimmed['P06']
# try arctic only
#esper_sel = espers[(espers.G2latitude < 55)]
#esper_sel = esper_sel[esper_sel.G2depth < 25] # do surface values (< 25 m) only 

# extract data
if 'del_alk' in esper_sel.columns:
    esper_sel.drop(columns=['del_alk'])
esper_sel['del_alk'] = esper_sel.G2talk - esper_sel[esper_type + 'talk' + str(equation_num)]
esper_sel = esper_sel[['dectime','datetime','del_alk','G2expocode']].dropna(axis=0)

# apply robust regression
x = esper_sel['dectime'].to_numpy().reshape(-1, 1) 
y = esper_sel['del_alk'].to_numpy().reshape(-1, 1) 

ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=round(esper_sel.shape[0]/2),
                         loss='absolute_error', random_state=42,
                         residual_threshold=10)

ransac.fit(x,y)

# get inlier mask and create outlier mask
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# create scatter plot for inlier dataset
fig = plt.figure(figsize=(7.5,5))
plt.scatter(x[inlier_mask], y[inlier_mask], c='steelblue', marker='o', label='Inliers', alpha=0.3, s=10)

# create scatter plot for outlier dataset
plt.scatter(x[outlier_mask], y[outlier_mask], c='lightgreen', marker='o', label='Outliers', alpha=0.3, s=10)

# draw best fit line
line_x = np.arange(x.min(), x.max(), 1)
line_y_ransac = ransac.predict(line_x[:, np.newaxis])
plt.plot(line_x, line_y_ransac, color='black', lw=2)

# output slope, intercept, and r2 (assuming time 0 is the first measurement )
slope = (line_y_ransac[-1][0] - line_y_ransac[0][0]) / (line_x[-1] - line_x[0])
intercept = line_y_ransac[0][0]
y_pred = ransac.predict(x)
r2 = r2_score(y,y_pred)

# calculate p value with two sample t test
# check if variances are equal - they are most definitely not
#print(np.var(y))
#print(np.var(y_pred))
result = stats.ttest_ind(a=y,b=y_pred,equal_var=False)
pvalue = result.pvalue


# formatting
ax = fig.gca()
if esper_type == 'M':
    esper_type = 'Mixed'
#ax.set_title('Difference in Measured and ESPER-Predicted (' + esper_type + ' Eqn. ' + str(equation_num) + ') TA \n with RANSAC Regression (GLODAPv2.2023 < 25 m, < 55º Latitude)')
ax.set_title('Difference in Measured and ESPER-Predicted (' + esper_type + ' Eqn. ' + str(equation_num) + ') TA \n with RANSAC Regression (GLODAPv2.2023 Transect P06 < 25 m)')
#ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
ax.set_ylim((-50,50))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)

# add box showing slope and r2
fig.text(0.14, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=12)
fig.text(0.14, 0.78, '$p-value={:.3e}$'.format(pvalue[0]), fontsize=12)

# do histogram to see where data is
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
esper_sel.hist(column='del_alk', bins = 200, ax=ax)
ax.set_title('GLODAPv2.2023 Transect P06 ' + esper_type + ' Eqn. ' + str(equation_num))
#ax.set_title('GLODAPv2.2023 < 25 m, < 55º Latitude, ' + esper_type + ' Eqn. ' + str(equation_num))
ax.set_xlabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
#ax.set_xlim((-50,50))

# calculate percent of data that is an inlier
percent_inlier = len(x[inlier_mask])/len(x) * 100

# %% find slope and p-value for RANSAC regression for all 3 methods, 16 equations

ransac_slope = np.zeros([16,3])
ransac_pvalue = np.zeros([16,3])

for j in range(0,3):
    if j == 0:
        esper_type = 'LIRtalk'
    elif j == 1:
        esper_type = 'NNtalk'
    else:
        esper_type = 'Mtalk'
        
    for i in range(1,17):
        
        # subset if desired
        esper_sel = espers
        # try arctic only
        #esper_sel = esper_sel[(esper_sel.G2latitude < 55)]
        esper_sel = esper_sel[esper_sel.G2depth < 25] # do surface values (< 25 m) only 
        esper_sel['del_alk'] = esper_sel.G2talk - esper_sel[esper_type + str(i)]
        esper_sel = esper_sel[['dectime','datetime','del_alk','G2expocode']].dropna(axis=0)
    
        # apply robust regression
        x = esper_sel['dectime'].to_numpy().reshape(-1, 1) 
        y = esper_sel['del_alk'].to_numpy().reshape(-1, 1) 
    
        ransac = RANSACRegressor(estimator=LinearRegression(), min_samples=round(esper_sel.shape[0]/2),
                             loss='absolute_error', random_state=42)#, residual_threshold=10)
        ransac.fit(x,y)
        
        # output slope, intercept, and p-value (assuming time 0 is the first measurement )
        line_x = np.arange(x.min(), x.max(), 1)
        line_y_ransac = ransac.predict(line_x[:, np.newaxis])
        slope = (line_y_ransac[-1][0] - line_y_ransac[0][0]) / (line_x[-1] - line_x[0])
        intercept = line_y_ransac[0][0]
        y_pred = ransac.predict(x)
        result = stats.ttest_ind(a=y,b=y_pred,equal_var=False)
        pvalue = result.pvalue
        
        # assign to table
        ransac_slope[i-1,j] = slope
        ransac_pvalue[i-1,j] = pvalue



