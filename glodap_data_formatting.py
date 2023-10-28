#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: glodap_data_formatting.py
Author: Reese Barrett
Date: 2023-10-26

Description:
    - Reads all GLODAPv2.2022 .csv files into pandas dataframe
    - Selects columns of interest for use in ESPERs and in comparison with
    ESPERs outputs
    - Converts time to decimal time
    - Deals with quality control flags
    - Filters data to keep only GO-SHIP cruises
    
To-Do:
    - filter rows with S > 50 or < 5
    - subset for GO-SHIP cruises
    - finalize formatting of output file (drop cruise columns once GO-SHIP complete)
"""
# %% set-up

import pandas as pd
import numpy as np
from datetime import datetime as dt

filepath = './data/'
input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv'
output_file_name = 'GLODAPv2.2022_for_ESPERs.csv'

# %% read in data
glodap = pd.read_csv(filepath + input_GLODAP_file)

# %% clean up NaN data
glodap = glodap.replace(-9999,np.nan)

# %% deal with quality control flags
# we want to keep data only marked as 2 or 9?
# or maybe we start by only keeping rows that have TA flagged as 2

glodap = glodap[glodap['G2talkf'] == 2] # get rid of rows with no TA measurement
glodap = glodap[glodap['G2year'] != np.nan] # get rid of rows with no year
glodap = glodap[glodap['G2month'] != np.nan] # get rid of rows with no month
glodap = glodap[glodap['G2day'] != np.nan] # get rid of rows with no day
glodap['G2hour'] = glodap['G2hour'].replace(np.nan,0) # replace NaN hour with 0
glodap['G2minute'] = glodap['G2minute'].replace(np.nan,0) # replace NaN minute with 0
glodap = glodap[(glodap['G2salinity'] >= 5) & (glodap['G2salinity'] <= 50)] # filter data with salinity > 50 or < 5 (ESPERs do not support this)
glodap = glodap.reset_index(drop=True) # reset index

# %% select relevant columns and rename

# select columns
glodap = glodap[['G2expocode','G2cruise','G2station','G2region',
                 'G2cast','G2year','G2month','G2day','G2hour',
                 'G2minute','G2latitude','G2longitude','G2depth',
                 'G2theta','G2salinity','G2oxygen','G2nitrate',
                 'G2silicate','G2phosphate','G2talk',
                 'G2phtsinsitutp']]
    

# rename columns

glodap = glodap.rename(columns={'G2expocode':'Expo Code',
                                'G2cruise':'Cruise',
                                'G2station':'Station',
                                'G2region':'Region',
                                'G2cast':'Cast',
                                'G2year':'Year',
                                'G2month':'Month',
                                'G2day':'Day',
                                'G2hour':'Hour',
                                'G2minute':'Minute',
                                'G2latitude':'Latitude',
                                'G2longitude':'Longitude',
                                'G2depth':'Depth',
                                'G2theta':'Theta',
                                'G2salinity':'Salinity',
                                'G2oxygen':'Oxygen',
                                'G2nitrate':'Nitrate',
                                'G2silicate':'Silicate',
                                'G2phosphate':'Phosphate',
                                'G2talk':'TA',
                                'G2phtsinsitutp':'pH'})

# %% convert datetime to decimal time

# allocate decimal year column
glodap.insert(0,'Decimal Time', 0.0)

for i in range(len(glodap)):

    # convert glodap time to datetime object
    date = dt(int(glodap.loc[i,'Year']), int(glodap.loc[i,'Month']),
           int(glodap.loc[i,'Day']), int(glodap.loc[i,'Hour']),
           int(glodap.loc[i,'Minute']),0)
    
    # convert datetime object to decimal time
    year = date.year
    this_year_start = dt(year=year, month=1, day=1)
    next_year_start = dt(year=year+1, month=1, day=1)
    year_elapsed = date - this_year_start
    year_duration = next_year_start - this_year_start
    fraction = year_elapsed / year_duration
    decimal_time = date.year + fraction
    
    # save to glodap dataset
    glodap.loc[i,'Decimal Time'] = decimal_time

# %% filter data for only GO-SHIP cruises

# %% reformat and output GLODAP data as one file for use in ESPERS

# delete datetime information
glodap = glodap.drop(['Year','Month','Day','Hour','Minute'], axis=1)

# delete cruise information
# add this after 

glodap.to_csv(filepath + output_file_name, index=False)