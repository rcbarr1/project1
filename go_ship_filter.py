#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: go_ship_filter.py
Author: Reese Barrett
Date: 2023-11-28

Description: Used to search for GO-SHIP cruises within GLODAP dataset. This
code became part of the "go_ship_only" function that is now in project1.py.
Contains WOCE, CLIVAR, GO-SHIP, GEOSECS, TTO, SOCCOM, and OVIDE data.
    
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
                     'MED01' : [64],
                     'I01' : [255],
                     'I03' : [252, 488],
                     'I05' : [251, 253, 355, 677, 682],
                     'I06' : [354, 373, 374, 3033],
                     'I07' : [254, 3034, 3041],
                     'I08N' : [251, 339, 4062],
                     'I08S' : [71, 249, 352, 1046],
                     'I09N' : [250, 353, 3035],
                     'I09S' : [72, 77, 249],
                     'I10' : [80, 82, 256, 1054],
                     'P01' : [299, 461, 468, 502, 504, 1053, 5014],
                     'P02' : [272, 459, 1035],
                     'P03' : [298, 497, 5017],
                     'P04' : [319],
                     'P06' : [243, 486, 273, 3029, 3030],
                     'P09' : [412, 515, 546, 547, 549, 550, 552, 554, 555, 556, 558, 559, 561, 562, 564, 565, 566, 568, 570, 571, 573, 576, 581, 583, 592, 595, 596, 599, 600, 603, 604, 607, 608, 609, 1056, 1057, 1058, 1067, 1071, 1079, 1080, 1082, 1083, 1087, 1090, 1093, 1100, 1101, 2041, 2047, 2057, 2062, 2067, 2075, 2080, 2087, 2099, 4066, 4068, 4069, 4071, 4078, 4089],
                     'P10' : [302, 495, 553, 557, 560, 563, 1087, 1090, 1093, 1098, 1099, 2050, 2057, 2062, 2075, 2087],
                     'P13' : [296, 360, 437, 439, 440, 517, 545, 548, 551, 553, 557, 560, 563, 567, 569, 572, 574, 575, 577, 579, 580, 582, 584, 585, 586, 587, 588, 589, 590, 591, 593, 594, 597, 598, 601, 602, 605, 606, 1058, 1060, 1063, 1064, 1066, 1069, 1071, 1076, 1078, 1079, 1081, 1092, 2038, 2041, 2047, 2054, 2064, 2084, 2091, 2094, 2096, 2097, 2102, 2103, 4063, 4069, 4074, 4076, 4081, 4083, 4087],
                     'P14' : [280, 301, 504, 505, 1050],
                     'P15' : [83, 84, 280, 335, 1020],
                     # 'P16' : [245, 276, 277, 285, 286, 304, 306, 307, 320, 350, 1036, 1043, 1044],
                     'P16N' : [276, 277, 286, 304, 306, 307, 1043, 1044],
                     'P16S' : [245, 285, 350, 1036],
                     'P17E' : [245, 246, 1055],
                     'P17N' : [300, 477],
                     'P18' : [279, 345, 1045],
                     'P21' : [270, 507, 1038],
                     'S04I' : [67, 73, 288, 1050, 1051],
                     'SR04' : [4, 5, 8, 11, 13, 15, 19, 20],
                     'S04P' : [295, 717, 3031],
                     'SR01' : [15, 19, 28, 332, 333, 675, 1111, 1113, 1114, 1115, 3043],
                     'SR03' : [65, 67, 68, 69, 70, 75, 76, 1021, 1022, 2008]}
                   

                

# test specific transect
#nums = go_ship_nums['ARC01E']
nums = go_ship_nums_2023['P03']
#nums = [233]
go_ship = glodap[glodap["G2cruise"].isin(nums)]

# see data minus a transect (check if any are missing)
#go_ship = glodap[~glodap["G2cruise"].isin(nums)]

#  USEFUL FOR VISUALIZING DATA LOCATIONS
# set up map
fig = plt.figure(figsize=(12,7))
#ax = plt.axes(projection=ccrs.PlateCarree()) # atlantic-centered view
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
ax.set_title('P03')
#extent = [136, 137.5, 10, 20]
extent = [110, -100, 0, 60]
#extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get data from glodap
#lon = espers.G2longitude
#lat = espers.G2latitude
lon = go_ship.G2longitude
lat = go_ship.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='x',edgecolors='none',s=20)
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
