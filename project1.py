#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project1.py
Author: Reese Barrett
Date: 2023-10-31

Description: Module holding functions developed for Project 1
- glodap_to_decimal_time
- glodap_qc_reformat
- go_ship_only
    
"""

# %% set-up
import numpy as np
from datetime import datetime as dt
import pandas as pd

def glodap_reformat_time(glodap):
    """
    Adds decimal time as "dectime" column and datetime as "datetime" column to
    GLODAP dataset calculated from G2year, G2month, G2day, G2hour, and
    G2minute. Filters out data with NaN year, month, or day information.
    
    Keyword arguments:
        glodap = pandas dataframe containing glodap dataset with original 
                headers
        
    Returns:
        glodap_out = same dataframe with additional column containing decimal
                    time
    """
    # get rid of rows with no year, month, or day
    glodap = glodap[glodap['G2year'] != np.nan] 
    glodap = glodap[glodap['G2month'] != np.nan]
    glodap = glodap[glodap['G2day'] != np.nan]
    
    # replace NaN hour or minute with 0
    glodap['G2hour'] = glodap['G2hour'].replace(np.nan,0)
    glodap['G2minute'] = glodap['G2minute'].replace(np.nan,0)
    
    # reset index
    glodap = glodap.reset_index(drop=True)
    
    # allocate decimal year column
    glodap.insert(0,'dectime', 0.0)
    #glodap.insert(1,'datetime', datetime.date(1,1,1))
    
    for i in range(len(glodap)):
    
        # convert glodap time to datetime object
        date = dt(int(glodap.loc[i,'G2year']), int(glodap.loc[i,'G2month']),
               int(glodap.loc[i,'G2day']), int(glodap.loc[i,'G2hour']),
               int(glodap.loc[i,'G2minute']),0)
        
        # convert datetime object to decimal time
        year = date.year
        this_year_start = dt(year=year, month=1, day=1)
        next_year_start = dt(year=year+1, month=1, day=1)
        year_elapsed = date - this_year_start
        year_duration = next_year_start - this_year_start
        fraction = year_elapsed / year_duration
        decimal_time = date.year + fraction
        
        # save to glodap dataset
        glodap.loc[i,'dectime'] = decimal_time
        #glodap.loc[i,'datetime'] = date
        glodap_out = glodap
        
    # create dataframe with correct headers to convert to datetime
    time = glodap[['G2year','G2month','G2day']]
    time = time.rename(columns={'G2year':'year','G2month':'month','G2day':'day'})
    dtime = pd.to_datetime(time)
    glodap.insert(0,'datetime', dtime)
    
    glodap_out = glodap    
                
    return glodap_out
      
    
def glodap_qc(glodap):
    """
    1. Selects only measured (not calculated) alkalinity
    2. Subsets data to only use open ocean salinity
    3. Eventually --> North Pacific corrections
    
    Keyword arguments:
        glodap = pandas dataframe containing glodap dataset with original
                headers plus G2dectime
        
    Returns:
        glodap_out = same dataframe with additional column containing decimal
                    time
    """
    
    # deal with quality control flags
    # only keep rows that have TA flagged as 2
    glodap = glodap[glodap['G2talkf'] == 2]
    
    #filter data with salinity to only look at open ocean
    glodap = glodap[(glodap['G2salinity'] >= 29) & (glodap['G2salinity'] <= 37)]
    
    # reset index
    glodap_out = glodap.reset_index(drop=True)
    
    return glodap_out
        
        
        