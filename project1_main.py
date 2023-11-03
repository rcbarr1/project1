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
filepath = '/Users/Reese/Documents/project1/data/' # where GLODAP data is stored
input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename

# %% import GLODAP data file
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

# %% filter data for only GO-SHIP cruises
#glodap = go_ship_only(glodap)

# %% do quality control
glodap = p1.glodap_qc(glodap)

# %% convert time to decimal time for use in ESPERs
glodap = p1.glodap_to_decimal_time(glodap)

# %% combine year, date, month columns into datetime

# %% call ESPERs
# this is done in MATLAB for now, will update when code is translated
# need to get Python ESPERs package and translate my MATLAB code calling ESPERs into Python
# for now: 1. save processed_glodap, 2. process in MATLAB (call_ESPERs.m), 3. upload again to here
# step 1: save processed_glodap

# select relevant columns
glodap = glodap[['G2expocode','G2cruise','G2station','G2region','G2cast',
                 'dectime','datetime','G2latitude','G2longitude','G2depth',
                 'G2temperature','G2salinity','G2oxygen','G2nitrate',
                 'G2silicate','G2phosphate','G2talk','G2phtsinsitutp']]
        
glodap.to_csv(filepath + 'GLODAPv2.2022_for_ESPERs.csv', index=False)
# %%step 3: upload ESPERs outputs to here
glodap = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv')

# %% start data visualization

# organize data by decimal time
glodap = glodap.sort_values(by=['dectime'],ascending=True)

# %% plot change in TA over time
fig = plt.figure(figsize=(7,5))
axs = plt.axes()

# currently trying to group by date, average, and then plot
# this obviously doesn't work because it's ignoring latitude and longitude, but
# attempting to start
glodap.groupby(glodap['datetime'].dt.day)['G2talk'].mean().plot(kind='line',ax=axs)
#plt.plot(glodap.groupby(glodap['datetime'].dt.day).mean(),glodap.G2talk)
axs.set_title('Total Alkalinity measured by GLODAPv2.2022')
axs.set_ylabel('Total Alkalinity ($mmol\;kg^{-1}$)')


# %% plot results from different ESPER methods (TA vs. time)
# fig, ax = plt.subplots(nrows= , ncols= , figsize = (5,5))












