#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project1_main.py
Author: Reese Barrett
Date: 2023-10-31

Description: Main script for Project 1, calls functions written in project1.py
    for data analysis
    
To-Do:
    - write go_ship_only function to subset for GO-SHIP code
    - write function to do corrections in North Pacific (add to glodap_qc)
    - translate call_ESPERs.m to python once ESPERs in Python are released
"""

# set-up

import project1 as p1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import interpolate
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

filepath = '/Users/Reese/Documents/project1/data/' # where GLODAP data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)

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

# %% trim GO-SHIP + associated cruises to pick out data points on the standard transect
# glodap = p1.trim_go_ship(glodap)

# %% start data visualization

# organize data by decimal time
espers = espers.sort_values(by=['dectime'],ascending=True)
# %% USEFUL FOR VISUALIZING DATA LOCATIONS
# set up map
fig = plt.figure(figsize=(15,10))
#ax = plt.axes(projection=ccrs.PlateCarree()) # atlantic-centered view
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
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
lon = espers.G2longitude
lat = espers.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=1)
# %% plot change in TA over time
fig = plt.figure(figsize=(15,10))
axs = plt.axes()

# currently trying to group by year, average, and then plot
# this obviously doesn't work because it's ignoring latitude and longitude, but
# attempting to start
espers.groupby(espers['datetime'].dt.year)['G2talk'].mean().plot(kind='line',ax=axs)
axs.set_title('Annual Averages of Total Alkalinity measured by GLODAPv2.2023')
axs.set_xlabel('Year')
axs.set_ylabel('Total Alkalinity ($mmol\;kg^{-1}$)')

# practice linear regression with data graphed above
x = list(espers.groupby(espers['datetime'].dt.year)['G2talk'].groups.keys())
y = espers.groupby(espers['datetime'].dt.year)['G2talk'].mean().tolist()
slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')
xseq = np.linspace(x[0], x[-1], num=100)
axs.plot(xseq, intercept + slope * xseq, color="r", lw=1);

# add textbox
ax.set_ylim(2295,2370)
fig.text(0.7, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
fig.text(0.7, 0.8, '$r^2={:.4f}$'.format(rvalue), fontsize=14)

# %% plot TA with time
fig = plt.figure(figsize=(7,5))
ax = plt.axes()

plt.scatter(espers.datetime,espers.G2talk,marker='o',facecolors='none',edgecolors='steelblue')

ax.set_title('GLODAPv2.2023 Total Alkalinity Measurements')
ax.set_xlabel('Year')
ax.set_ylabel('Total Alkalinity ($mmol\;kg^{-1}$)')

# %% map of location of TA measurements colored by year

# set up map
fig = plt.figure(figsize=(15,6.5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
ax.set_title('Spatial and Temporal Coverage of Total Alkalinity (GLODAPv2.2023)')
#extent = [-155,-140,-70,70]
#ax.set_extent(extent)

# get data from glodap
lon = espers.G2longitude
lat = espers.G2latitude
time = espers.datetime.dt.year
plot = ax.scatter(lon,lat,c=time,cmap=plt.cm.plasma,transform=ccrs.PlateCarree(),marker='o',edgecolors='none')

# set up colorbar
c = plt.colorbar(plot,ax=ax)
c.set_label('Year')

# %% map of location of TA measurements colored by TA

# set up map
fig = plt.figure(figsize=(15,6.5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
ax.set_title('Surface (< 5 m Depth) Total Alkalinity Measurements (GLODAPv2.2023)')

# get data from glodap
lon = espers.G2longitude[espers.G2depth < 5]
lat = espers.G2latitude[espers.G2depth < 5]
alk = espers.G2talk[espers.G2depth < 5]
plot = ax.scatter(lon,lat,c=alk,cmap=cmocean.cm.matter,transform=ccrs.PlateCarree(),marker='o',edgecolors='none')

# set up colorbar
c = plt.colorbar(plot,ax=ax)
c.set_label('Total Alkalinity ($mmol\;kg^{-1}$)')

# %% plot TA and ESPERs with time
fig = plt.figure(figsize=(7,5))
ax = plt.axes()

# plot LIR results
espers.groupby(espers['datetime'].dt.year)['LIRtalk1'].mean().plot(kind='line',ax=ax,color='teal',alpha=0.6,linewidth=0.3,label='ESPER LIR')
for i in range(2,17):
    espers.groupby(espers['datetime'].dt.year)['LIRtalk'+str(i)].mean().plot(kind='line',ax=ax,color='teal',alpha=0.6,linewidth=0.3,label='_nolegend_')

# plot NN results
espers.groupby(espers['datetime'].dt.year)['NNtalk1'].mean().plot(kind='line',ax=ax,color='mediumorchid',alpha=0.6,linewidth=0.3,label='ESPER NN')
for i in range(2,17):
    espers.groupby(espers['datetime'].dt.year)['NNtalk'+str(i)].mean().plot(kind='line',ax=ax,color='mediumorchid',alpha=0.6,linewidth=0.3,label='_nolegend_')

# plot mixed results
espers.groupby(espers['datetime'].dt.year)['Mtalk1'].mean().plot(kind='line',ax=ax,color='gold',alpha=0.6,linewidth=0.3,label='ESPER Mixed')
for i in range(2,17):
    espers.groupby(espers['datetime'].dt.year)['Mtalk'+str(i)].mean().plot(kind='line',ax=ax,color='gold',alpha=0.6,linewidth=0.3,label='_nolegend_')

# plot actual GLODAP measurements
espers.groupby(espers['datetime'].dt.year)['G2talk'].mean().plot(kind='line',ax=ax,color='k',linewidth=1,label='GLODAP')

# formatting
ax.set_title('Annual Averages of Total Alkalinity')
ax.set_xlabel('Year')
ax.set_ylabel('Total Alkalinity ($mmol\;kg^{-1}$)')
ax.set_ylim(2295,2370)
leg = ax.legend(loc='upper right')
for line in leg.get_lines():
    line.set_linewidth(2)
    
# %% plot TA measured vs LIR equations with time
fig = plt.figure(figsize=(15,10))
ax = plt.axes()

#espers.groupby(espers['datetime'].dt.year)['LIRtalk1'].mean().plot(kind='line',ax=ax,color='teal',alpha=0.6,linewidth=0.3,label='ESPER LIR')
for i in range(1,17):
    label = 'ESPER LIR Eqn. ' + str(i)
    espers.groupby(espers['datetime'].dt.year)['LIRtalk'+str(i)].mean().plot(kind='line',ax=ax,linewidth=0.3,label=label)
    
espers.groupby(espers['datetime'].dt.year)['G2talk'].mean().plot(kind='line',ax=ax,color='k',linewidth=1,label='GLODAP')

# formatting
ax.set_title('Annual Averages of Total Alkalinity')
ax.set_xlabel('Year')
ax.set_ylabel('Total Alkalinity ($mmol\;kg^{-1}$)')
ax.set_ylim(2295,2370)
leg = ax.legend(loc='upper right',ncol=2)
for line in leg.get_lines():
    line.set_linewidth(2)
    
# %% look along a sample transect

# subset data by transect G2cruise = 1036
transect = espers[(espers.G2cruise == 1036) & (espers.G2longitude < -149) & (espers.G2longitude > -152)]

# show transect
fig = plt.figure(figsize=(5,3))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
extent = [-180,180,-90,90]
ax.set_extent(extent)
ax.set_title('Cruise 1036 Transect')
lon = transect.G2longitude
lat = transect.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o')

# make TA section plot
transect1 = transect[transect.datetime.dt.year == 2014]
fig = plt.figure(figsize=(10,7))
ax = plt.gca()
ax.invert_yaxis()

# pull GLODAP data
lat = transect1.G2latitude
depth = transect1.G2depth
alk = transect1.G2talk
plot = ax.scatter(lat,depth,c=alk,cmap=cmocean.cm.matter)

# formatting
ax.set_title('TA Measurements Along 2014 Cruise 1036 Transect')
ax.set_xlabel('Latitude (ºE)')
ax.set_ylabel('Depth (m)')
ax.set_xlim((transect1.G2latitude.min(),transect1.G2latitude.max()))
ax.set_ylim((transect1.G2depth.min(),transect1.G2depth.max()))
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Total Alkalinity ($mmol\;kg^{-1}$)')

# interpolated TA section plot
x_coord = np.linspace(transect1.G2latitude.min(),transect1.G2latitude.max(),500)
y_coord = np.linspace(transect1.G2depth.min(),transect1.G2depth.max(),500)
x_grid, y_grid = np.meshgrid(x_coord,y_coord)
z_gridded = interpolate.griddata((transect1.G2latitude,transect1.G2depth),transect1.G2talk,(x_grid,y_grid),method='linear',rescale=True)

fig = plt.figure(figsize=(10,7))
ax = plt.gca()
ax.invert_yaxis()
plt.pcolormesh(x_grid,y_grid,z_gridded,cmap=cmocean.cm.matter)
ax.set_title('TA Measurements: Cruise 1036, 2014')
ax.set_xlabel('Latitude (ºE)')
ax.set_ylabel('Depth (m)')
ax.set_xlim((transect1.G2latitude.min(),transect1.G2latitude.max()))
ax.set_ylim((transect1.G2depth.min(),transect1.G2depth.max()))
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Total Alkalinity ($mmol\;kg^{-1}$)')

# %% along transect - make section plot showing simple difference between GLODAP TA and ESPER LIR TA
fig = plt.figure(figsize=(10,7))
ax = plt.gca()
del_alk = transect1.G2talk - transect1.NNtalk13 # calculate simple difference
newcmap = cmocean.tools.crop(cmocean.cm.balance, del_alk.min(), del_alk.max(), 0) # pivot cmap around 0
plot = ax.scatter(lat,depth,c=del_alk,cmap=newcmap)

# formatting
ax.set_title('(Measured TA - ESPER-Predicted TA) Along 2014 Cruise 1036 Transect')
ax.set_xlabel('Latitude (ºE)')
ax.set_ylabel('Depth (m)')
ax.set_xlim((transect1.G2latitude.min(),transect1.G2latitude.max()))
ax.set_ylim((transect1.G2depth.min(),transect1.G2depth.max()))
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Difference in Total Alkalinity ($mmol\;kg^{-1}$)')

# interpolated TA plot showing simple difference between GLODAP TA and ESPER LIR TA
x_coord = np.linspace(transect1.G2latitude.min(),transect1.G2latitude.max(),500)
y_coord = np.linspace(transect1.G2depth.min(),transect1.G2depth.max(),500)
x_grid, y_grid = np.meshgrid(x_coord,y_coord)
z_gridded = interpolate.griddata((transect1.G2latitude,transect1.G2depth),del_alk,(x_grid,y_grid),method='linear',rescale=True)

fig = plt.figure(figsize=(10,7))
ax = plt.gca()
ax.invert_yaxis()
plt.pcolormesh(x_grid,y_grid,z_gridded,cmap=newcmap)
ax.set_title('(Measured TA - ESPER-Predicted TA): Cruise 1036, 2014')
ax.set_xlabel('Latitude (ºE)')
ax.set_ylabel('Depth (m)')
ax.set_xlim((transect1.G2latitude.min(),transect1.G2latitude.max()))
ax.set_ylim((transect1.G2depth.min(),transect1.G2depth.max()))
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Difference in Total Alkalinity ($mmol\;kg^{-1}$)')


# %% hovmoller plot for a specific station
# Sample: Station at ~ 155ºE, 9.5ºN has 258 data points
station = espers[(espers.G2longitude < 156) & (espers.G2longitude > 154) & (espers.G2latitude < 9.8) & (espers.G2latitude > 9.2)]

# draw map highlighting station
fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
ax.set_title('Spatial Coverage of Total Alkalinity (GLODAPv2.2023)')

# get data from glodap
lon = espers.G2longitude
lat = espers.G2latitude
plot = ax.scatter(lon,lat,s=0.2,transform=ccrs.PlateCarree(),marker='o',)
ax.scatter(155,9.5,s=40,transform=ccrs.PlateCarree(),color='r')

# extract data
time = station.datetime
depth = station.G2depth
del_alk = station.G2talk - station.LIRtalk1

# create colormap centered at 0
newcmap = cmocean.tools.crop(cmocean.cm.balance, del_alk.min(), del_alk.max(), 0) # pivot cmap around 0

# plot
fig = plt.figure(figsize=(10,7))
ax = plt.gca()
ax.invert_yaxis()
plot = ax.scatter(time,depth,c=del_alk,cmap=newcmap)
c = plt.colorbar(plot,ax=ax)
c.set_label('Difference in Total Alkalinity ($mmol\;kg^{-1}$)')
ax.set_ylabel('Depth (m)')
ax.set_title('Difference in Measured TA and ESPER-Predicted at 155ºE, 9.5ºN')

# %% hovmoller plot for an ocean basin
# Sample: North Atlantic
basin = espers[(espers.G2longitude < 12) & (espers.G2longitude > -98) & (espers.G2latitude < 55) & (espers.G2latitude > 0)]

# extract data (CHANGE WHICH EQUATION/ROUTINE CALLED HERE)
if 'del_alk' in basin.columns:
    basin.drop(columns=['del_alk'])
basin['del_alk'] = basin.G2talk - basin.Mtalk1

# get rid of outliers (talk to Brendan about what is appropriate for this?)
#basin = basin[(basin.del_alk < 50) & (basin.del_alk > -50)]

# create colormap centered at 0
newcmap = cmocean.tools.crop(cmocean.cm.balance, basin.del_alk.min(), basin.del_alk.max(), 0) # pivot cmap around 0

# make hovmoller plot
fig = plt.figure(figsize=(10,7))
ax = plt.gca()
plot = ax.scatter(basin.datetime,basin.G2depth,c=basin.del_alk,cmap=newcmap)
ax.set_xlim((basin.datetime.min(),basin.datetime.max()))
ax.set_ylim((basin.G2depth.min(),basin.G2depth.max()))
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Difference in Total Alkalinity ($mmol\;kg^{-1}$)')
ax.set_ylabel('Depth (m)')
ax.set_title('Difference in Measured and ESPER-Predicted TA in North Atlantic')

# interpolated hovmoller plot
x_coord = np.linspace(basin.dectime.min(),basin.dectime.max(),500)
y_coord = np.linspace(basin.G2depth.min(),basin.G2depth.max(),500)
x_grid, y_grid = np.meshgrid(x_coord,y_coord)
z_gridded = interpolate.griddata((basin.dectime,basin.G2depth),basin.del_alk,(x_grid,y_grid),method='linear',rescale=True)

fig = plt.figure(figsize=(10,7))
ax = plt.gca()
plt.pcolormesh(x_grid,y_grid,z_gridded,cmap=newcmap)
ax.set_xlim((basin.dectime.min(),basin.dectime.max()))
ax.set_ylim((basin.G2depth.min(),basin.G2depth.max()))
ax.locator_params(axis='x', nbins=10)
ax.invert_yaxis()
c = plt.colorbar(plot,ax=ax)
c.set_label('Difference in Total Alkalinity ($mmol\;kg^{-1}$)')
ax.set_ylabel('Depth (m)')
ax.set_title('Difference in Measured and ESPER-Predicted TA in North Atlantic')

# plot surface values and do linear regression
surface_basin = basin[basin.G2depth < 25]
surface_basin = surface_basin[['dectime','datetime','del_alk']].dropna(axis=0)
x = surface_basin.dectime
y = surface_basin.del_alk

slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y, alternative='two-sided')

# make plot
fig = plt.figure(figsize=(7,5))
ax = plt.gca()
plt.scatter(surface_basin.datetime,y,s=1)
fig.text(0.544, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=14)
fig.text(0.7, 0.78, '$r^2={:.4f}$'.format(rvalue), fontsize=14)
ax.plot(surface_basin.datetime, intercept + slope * surface_basin.dectime, color="r", lw=1);
ax.set_title('Difference in Measured and ESPER-Predicted TA in North Atlantic (< 25 m)')
ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
ax.set_ylim(-70,70)

# %% do robust regression to take care of outliers

# SET ESPER ROUTINE HERE
esper_type = 'NN' # LIR, NN, or M
equation_num = 7 # 1 through 16

# subset if desired
esper_sel = espers
# try arctic only
esper_sel = espers[(espers.G2latitude < 55)]
esper_sel = esper_sel[esper_sel.G2depth < 25] # do surface values (< 25 m) only 

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
ax.set_title('Difference in Measured and ESPER-Predicted (' + esper_type + ' Eqn. ' + str(equation_num) + ') TA \n with RANSAC Regression (GLODAPv2.2023 < 25 m, < 55º Latitude)')
ax.set_ylabel('Measured TA - ESPER-Estimated TA ($mmol\;kg^{-1}$)')
ax.set_ylim((-50,50))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)

# add box showing slope and r2
fig.text(0.14, 0.83, '$y={:.4f}x+{:.4f}$'.format(slope,intercept), fontsize=12)
fig.text(0.14, 0.78, '$p-value={:.3e}$'.format(pvalue[0]), fontsize=12)

# do histogram to see where data is
fig = plt.figure(figsize=(7,5))
ax = fig.gca()
esper_sel.hist(column='del_alk', bins = 200, ax=ax)
ax.set_title('GLODAPv2.2023 < 25 m, < 55º Latitude, ' + esper_type + ' Eqn. ' + str(equation_num))
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



