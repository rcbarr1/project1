#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: project1_.py
Author: Reese Barrett
Date: 2023-10-31

Description: Main script for Project 1, calls functions written in project1.py
    for data analysis
    
To-Do:
    - write go_ship_only function to subset for GO-SHIP code
    - write function to do corrections in North Pacific
    - translate call_ESPERs.m to python once ESPERs in Python are released
"""

# set-up

import project1 as p1
import pandas as pd
filepath = '/Users/Reese/Documents/project1/data/' # where GLODAP data is stored
input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename

# %% import GLODAP data file
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

# %% filter data for only GO-SHIP cruises
#glodap = go_ship_only(glodap)

# %% do North Pacific correction

# %% convert time to decimal time for use in ESPERs
glodap = p1.glodap_to_decimal_time(glodap)

# %% format GLODAP data for ESPERs
glodap = p1.glodap_qc_reformat(glodap)

# %% call ESPERs
# this is done in MATLAB for now, will update when code is translated
# need to get Python ESPERs package and translate my MATLAB code calling ESPERs into Python
# for now: 1. save processed_glodap, 2. process in MATLAB (call_ESPERs.m), 3. upload again to here
# step 1:
glodap.to_csv(filepath + 'GLODAPv2.2022_for_ESPERs.csv', index=False)
# %%step 3:
glodap = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv')

# %%start data visualization!

