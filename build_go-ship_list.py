#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: build_go-ship_list.py
Author: Reese Barrett
Date: 2023-10-31

Description:
    - Based on CAREER_0_BuildCruiseList.m (Brendan Carter)
    - GO-SHIP reference sections from https://www.go-ship.org/RefSecs/goship_ref_secs.html
    - Reads all GLODAPv2.2022 .csv files into pandas dataframe
    - Selects all GO-SHIP cruises from GLODAP data
    - Converts GLODAP data format to decimal date
    - Outputs:
        1. a dict that stores each GO-SHIP cruise, the relevant ocean basin, if
        the data is good to use, the cruise numbers of the data in GLODAP, and 
        the average decimal date of each single cruise
        2. a dataframe with all GO-SHIP cruises selected from GLODAP, converted
        to decimal time, with column labeling GO-SHIP transect 
    - Exports as ???
    
To-Do:
    - write this as a function to call in glodap_data_formatting.py
    - write dict
    - move decimal date conversion here
    - format and name output, update description
   
"""

# %% set-up

import pandas as pd
#import numpy as np
#from datetime import datetime as dt

filepath = '/Users/Reese/Documents/project1/data/'
input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv'
output_file_name = 'GLODAPv2.2022_for_ESPERs.csv'

# %% read in data
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

# %% search for Pacific Ocean cruises
# P01
n = 1


# P02
n += 1

# P03E
n += 1

# P03W
n += 1

# P04E
n += 1

# P04W
n += 1

# P06
n += 1

# P09
n += 1

# P10
n += 1

# P13
n += 1

# P14N
n += 1

# P14S
n += 1

# P15S
n += 1

# P16N
n += 1

# P16S
n += 1

# P17x/P17E
n += 1

# P18
n += 1

# 40N
n += 1

# SO4P
n += 1

# search for Arctic Ocean cruises
# ARC01
n += 1

# ARC02
n += 1

# search for Atlantic Ocean cruises
# Davis
n += 1

# AR07E
n += 1

# AR07W
n += 1

# AR28
n += 1

# SR01
n += 1

# SR04
n += 1

# A02
n += 1

# A05
n += 1

# A10
n += 1

# A12
n += 1

# A13.5
n += 1

# A16N
n += 1

# A16S
n += 1

# A17
n += 1

# A20
n += 1

# A22
n += 1

# A23
n += 1

# A25
n += 1

# A29
n += 1

# search for Indian Ocean cruises
# I01W
n += 1

# I01E
n += 1

# I03
n += 1

# I05
n += 1

# I06S
n += 1

# I07N
n += 1

# I07S
n += 1

# I08N
n += 1

# I08S
n += 1

# I09N
n += 1 

# 109S
n += 1

# I10
n += 1

# SR03
n += 1

# S04I
n += 1

# search for Mediterranean cruise
# MED01
n += 1






