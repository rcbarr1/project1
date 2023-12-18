#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: monte_carlo_dev.py
Author: Reese Barrett
Date: 2023-12-15

Description: Used to develop the Monte Carlo simulation for this analysis. Will
be incorporated into a function called in project1_main.py upon completion.
"""

import pandas as pd
import numpy as np
import project1 as p1

# %% upload ESPERs outputs to here
filepath = '/Users/Reese/Documents/Research Projects/project1/data/' # where GLODAP data is stored
# import GLODAP data files
glodap = pd.read_csv(filepath + 'GLODAPv2.2023_Merged_Master_File.csv', na_values = -9999)

_ , go_ship_cruise_nums_2023 = p1.go_ship_only(glodap) # get list of go_ship cruise numbers

# %% upload & process espers data

# upload ESPERs outputs to here
espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv')
espers['datetime'] = pd.to_datetime(espers['datetime']) # recast datetime as datetime data type

# calculate ensemble mean TA for each data point
espers = p1.ensemble_mean(espers)

# trim cruises to get rid of extraneous points not along transects
trimmed = p1.trim_go_ship(espers, go_ship_cruise_nums_2023)
all_trimmed = pd.concat(trimmed.values(), ignore_index=True) # flatten from dict of dataframes into one large dataframe
all_trimmed = all_trimmed.drop_duplicates(ignore_index=True) # drop duplicates

# %% apply offset to transects
cruise_nums = list(all_trimmed.G2cruise.unique())
num_mc_runs = 1000 # number of monte carlo simulations created

G2talk_mc = np.empty((len(all_trimmed.G2talk),num_mc_runs))

for j in range(0,num_mc_runs): # loop to repeat x times for MC analysis
    go_ship_offset = all_trimmed.copy()
    
    # loop through all cruises
    for i in cruise_nums:
        # get random number between from normal distribution with mean at 0 and
        # standard deviation at 2, representing ± 2 µmol/kg alkalinity
        offset = np.random.normal(loc = 0.0, scale = 2)
        offset = np.round(offset, decimals=1)
        
        # add offset to all rows tagged with cruise number i
        go_ship_offset.loc[go_ship_offset.G2cruise == i, ['G2talk']] += offset
    
    # select relevant column and store in numpy array
    G2talk_mc[:,j] = go_ship_offset['G2talk'].to_numpy()

# export go_ship dataframe as basis to run through espers, will switch out G2talk columns for simulated ones saved below
all_trimmed.to_csv(filepath + 'go_ship_trimmed_for_ESPERs.csv', index=False) # monte carlo run j

# export dataframe of simulated G2talk columns as .csv to put back with go_ship dataframe and run through espers        
G2talk_mc_df = pd.DataFrame(G2talk_mc)
G2talk_mc_df.to_csv(filepath + 'G2talk_mc_simulated.csv', index=False)












