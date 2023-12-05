#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: go_ship_filter.py
Author: Reese Barrett
Date: 2023-11-28

Description: Used to search for GO-SHIP cruises within GLODAP dataset.
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

filepath = '/Users/Reese/Documents/project1/data/' # where GLODAP data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)

# %% import GLODAP data file
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

# %% filter data for only GO-SHIP cruises
# I skipped WCOA (327 & 2025) and OVIDE (392,393,394,25,395,1032,1047,2011), included in the script Brendan sent but I don't think they're GO-SHIP cruises
go_ship_nums = {'A02':[37,43,24,49,2027],'A05':[225,341,695,699,1030],'A10':[34,487,347,2105,1008],'A12':[6,11,13,385,14,15,18,19,20,1004],'A135':[239,346,1004],'A16':[242,338,342,343,1041,1042],'A17':[297],'A20':[260,264,330],'A22':[261,265,329],
                'AR07E':[672,666,667,1108,1103],'AR07W':[159,160,161,158,162,167,163,166,165,164,1028,1026,1027,1029,1025],'ARC01E':[708,1040],'ARC01W':[197,1040],'Davis':[201,2014,2015,2016,2017,2018,2009,2019,2021,2022,2023],
                'I01':[255],'I03':[252,253,488],'I05':[251,253,355],'I06':[374,354,3033],'I07':[253,254,3034,3041],'I08N':[251,339],'I08S_I09N':[249,250,352,353,1046,3035],'I09S':[249,72],'I10':[256,1054],'MED01':[52,64],
                'P01':[461,116,502,504,1053],'P02':[459,272,1035],'P03':[2098],'P04':[319,2099],'P06':[243,486,273,3029,3030],'P09':[515,571,604,412,595,581,554,600,599,596,552,2002,559,549,556,555,558,547,608,1058,2087,1079,1067,1090,1071,2099,2080,2067,2075,2062,1093,2047,2041,2057,1101],
                'P10':[302,495,563,2087,2050,1099],'P13':[296,517,1081,1058,1063,1066,1069,1071,2432,2094,2047,2103,1076,2038,2041,2054,2064],'P14':[301,280,504,505],'P15':[280,84,1020],'P16':[285,286,245,277,350,206,1036,1043,1044],'P17':[1055],'P17N':[300,477],'P18':[279,345,1045],'P21':[270,507],
                'S04I':[67,288,1050,1051],'S04A':[8,11,13,15,19,20],'S04P':[717,295,3031],'SR03':[67,1021,68,75,2008]}

go_ship_nums_2023 = {'A02' : [24, 37, 43, 1006, 2027],
                     'A05' : [225, 341, 695, 699, 1030, 1109],
                     'A10' : [34, 347, 487, 676, 1008, 2105],
                     'A12' : [6, 11, 13, 14, 15, 18, 19, 20, 28, 233, 385, 1004],
                     'A135' : [239, 346, 1004],
                     #'A16' : [63, 322, 336, 338, 342, 343, 366, 694, 700, 1041, 1042],
                     'A16N' : [63, 322, 338, 342, 366, 694, 700, 1041],
                     'A16S' : [336, 343, 1042],
                     'A17' : [229, 230, 235, 297, 2013],
                     'A20' : [260, 264, 330, 5005],
                     'A22' : [261, 265, 329, 5006],
                     'A23' : [701],
                     'A25' : [31, 38, 45, 693, 2011, 5007],
                     'A29' : [634, 635, 1003, 1102, 1104],
                     'AR07E' : [31, 38, 45, 660, 666, 667, 672],
                     'AR07W' : [26, 38, 44, 151, 153, 155, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 698, 1025, 1026, 1027, 1028, 1029, 2011, 4005],
                     'ARC01E' : [708, 1040],
                     'ARC01W' : [1040],
                     'MED01' : [64]}

#                     'SR04' : [4, 5, 8, 11, 13, 15, 19, 20],

                

# test specific transect
#nums = go_ship_nums['ARC01E']
nums = go_ship_nums_2023['MED01']
#nums = [1104]
#go_ship = glodap
#go_ship = glodap[glodap["G2cruise"].isin(nums)]

# see data minus a transect (check if any are missing)
go_ship = glodap[~glodap["G2cruise"].isin(nums)]

# see all transects
#flat_nums = [element for sublist in (list(go_ship_nums.values())) for element in sublist]
#go_ship = glodap[glodap["G2cruise"].isin(flat_nums)]

# see data minus all transects (check if any are missing)
#flat_nums = [element for sublist in (list(go_ship_nums.values())) for element in sublist]
#go_ship = glodap[~glodap["G2cruise"].isin(flat_nums)]

#  USEFUL FOR VISUALIZING DATA LOCATIONS
# set up map
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
#ax.set_title('North Atlantic Coverage of TA (GLODAPv2.2023)')
#extent = [-57, -47, 52.5, 61]
#extent = [-10, 50, 20, 45]
#extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get data from glodap
#lon = espers.G2longitude
#lat = espers.G2latitude
lon = go_ship.G2longitude
lat = go_ship.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10)
# turn on to see what cruises are within the "extent" set above
matches = go_ship[(go_ship.G2longitude > extent[0]) & (go_ship.G2longitude < extent[1]) & (go_ship.G2latitude > extent[2]) & (go_ship.G2latitude < extent[3])]
print(matches[['G2expocode','G2cruise']])

# %% map all cruises seen in a given latitude/longitude box (set by "extent" above) to see if any match a GO-SHIP track
all_nums = matches.G2cruise.unique()
for i in range(len(all_nums)):
    nums = all_nums[i]
    
    go_ship = glodap[glodap["G2cruise"] == nums]

    # set up map
    fig = plt.figure(figsize=(15,10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution='110m',color='k')
    g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
    g1.top_labels = False
    g1.right_labels = False
    ax.add_feature(cfeature.LAND,color='k')
    ax.set_title('Cruise ' + str(nums))
    ax.set_extent(extent)

    lon = go_ship.G2longitude
    lat = go_ship.G2latitude
    plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='o',edgecolors='none',s=10)
    
    print(nums)
    print(go_ship.G2expocode.iloc[0])

# %% find cruises with a given expocode

expo = glodap[glodap.G2expocode == '45CE20100209']
print(expo.G2cruise.iloc[0])
