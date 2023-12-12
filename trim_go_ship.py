#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: trim_go_shp.py
Author: Reese Barrett
Date: 2023-12-08

Description: Used to group cruises by transect and trim extraneous points. This
code became part of the "trim_go_ship" function that is now in project1.py.
Contains WOCE, CLIVAR, GO-SHIP, TTO, SOCCOM, and OVIDE data.
    
"""

import pandas as pd
import matplotlib.pyplot as plt
import project1 as p1
import cartopy.crs as ccrs
import cartopy.feature as cfeature

filepath = '/Users/Reese/Documents/project1/data/' # where GLODAP data is stored
#input_GLODAP_file = 'GLODAPv2.2022_Merged_Master_File.csv' # GLODAP data filename (2022)
input_GLODAP_file = 'GLODAPv2.2023_Merged_Master_File.csv' # GLODAP data filename (2023)

# %% import GLODAP data file
glodap = pd.read_csv(filepath + input_GLODAP_file, na_values = -9999)

# %% get cruise numbers associated with each transect
_, go_ship_cruise_nums_2023 = p1.go_ship_only(glodap)

# %% upload ESPERs outputs to here
espers = pd.read_csv(filepath + 'GLODAP_with_ESPER_TA.csv')
espers['datetime'] = pd.to_datetime(espers['datetime']) # recast datetime as datetime data type

# %% for each cruise transect, pull all associated cruises

# determine which points to keep for each transect
# XX.YY = cruise.station to trim
# save this information in a dataframe for each transect (i.e. "A10" has all of the relevant rows from "espers")

A02 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A02'])]
A02 = A02[~((A02.G2cruise == 24) & (A02.G2station == 51))] # trim 24.51

A05 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A05'])] # no trimming needed

A10 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A10'])]
for i in range(1,22):
    A10 = A10[~((A10.G2cruise == 676) & (A10.G2station == i))] # trim 676.[1:21]
   
A12 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A12'])]
for i in range(317,347):
    A12 = A12[~((A12.G2cruise == 18) & (A12.G2station == i))] # trim 18.[317:346]
A12 = A12[~((A12.G2cruise == 19) & (A12.G2station == 112))] # trim 19.112
A12 = A12[~((A12.G2cruise == 19) & (A12.G2station == 154))] # trim 19.154
A12 = A12[~((A12.G2cruise == 19) & (A12.G2station == 155))] # trim 19.155
for i in range(184,259):
    A12 = A12[~((A12.G2cruise == 19) & (A12.G2station == i))] # trim 19.[154:258]
for i in range(69,124):
    A12 = A12[~((A12.G2cruise == 20) & (A12.G2station == i))] # trim 19.[69:123]
for i in range(1,48):
    A12 = A12[~((A12.G2cruise == 233) & (A12.G2station == i))] # trim 233.[1:47]
for i in range(40,74):
    A12 = A12[~((A12.G2cruise == 1004) & (A12.G2station == i))] # trim 1004.[40:73]
    
A135 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A135'])]
for i in range(1,10):
    A135 = A135[~((A135.G2cruise == 1004) & (A135.G2station == i))] # trim 1004.[1:9]

A16N = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A16N'])]
for i in range(1,81):
    A16N = A16N[~((A16N.G2cruise == 63) & (A16N.G2station == i))] # trim 63.[1:80]
for i in range(125,130):
    A16N = A16N[~((A16N.G2cruise == 322) & (A16N.G2station == i))] # trim 322.[125:129]
A16N = A16N[~((A16N.G2cruise == 338) & (A16N.G2station == 1))] # trim 338.1
A16N = A16N[~((A16N.G2cruise == 338) & (A16N.G2station == 32))] # trim 338.32
for i in range(1,52):
    A16N = A16N[~((A16N.G2cruise == 366) & (A16N.G2station == i))] # trim 366.[1:51]
for i in range(13415,13467):
    A16N = A16N[~((A16N.G2cruise == 694) & (A16N.G2station == i))] # trim 694.[13415:13466]
    
A16S = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A16S'])]
for i in range(1,14):
    A16S = A16S[~((A16S.G2cruise == 343) & (A16S.G2station == i))] # trim 343.[1:13]
for i in range(101,114):
    A16S = A16S[~((A16S.G2cruise == 1042) & (A16S.G2station == i))] # trim 1042.[101:113]

A17 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A17'])]
A17 = A17[~((A17.G2cruise == 229) & (A17.G2station == 26))] # trim 229.26
A17 = A17[~((A17.G2cruise == 229) & (A17.G2station == 27))] # trim 229.27
for i in range(63,66):
    A17 = A17[~((A17.G2cruise == 235) & (A17.G2station == i))] # trim 235.[63:65]
for i in range(19,49):
    A17 = A17[~((A17.G2cruise == 236) & (A17.G2station == i))] # 236.[19:48]
for i in range(67,95):
    A17 = A17[~((A17.G2cruise == 236) & (A17.G2station == i))] # 236.[67:94]
for i in range(100,122):
    A17 = A17[~((A17.G2cruise == 236) & (A17.G2station == i))] # 236.[100:121]
for i in range(42,59):
    A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[42:58]
for i in range(118,142):
    A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[118:141]
for i in range(172,236):
    A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[172:235]
for i in range(77,82):
    A17 = A17[~((A17.G2cruise == 2013) & (A17.G2station == i))] # trim 2013.[77:81]

A20 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A20'])]
for i in range(1,30):
    A20 = A20[~((A20.G2cruise == 236) & (A20.G2station == i))] # 236.[1:29]
A20 = A20[~((A20.G2cruise == 236) & (A20.G2station == 35))] # 236.35
for i in range(38,119):
    A20 = A20[~((A20.G2cruise == 236) & (A20.G2station == i))] # 236.[38:118]
for i in range(120,122):
    A20 = A20[~((A20.G2cruise == 236) & (A20.G2station == i))] # 236.[120:121]

A22 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A22'])]
A22 = A22[~((A22.G2cruise == 261) & (A22.G2station == 1))]  # trim 261.1
for i in range(54,56):
    A22 = A22[~((A22.G2cruise == 265) & (A22.G2station == i))]  # trim 265.[54:55]

A25 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A25'])]
for i in range(101,118):
    A25 = A25[~((A25.G2cruise == 2011) & (A25.G2station == i))] # trim 2011.[101:117]
for i in range(121,137):
    A25 = A25[~((A25.G2cruise == 2011) & (A25.G2station == i))] # trim 2011.[121:136]
    
A29 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A29'])] # no trimming needed
for i in range(206,228):
    A29 = A29[~((A29.G2cruise == 635) & (A29.G2station == i))]  # 635.[206:227]
for i in range(265,402):
    A29 = A29[~((A29.G2cruise == 635) & (A29.G2station == i))]  # 635.[265:401]
for i in range(1,9):
    A29 = A29[~((A29.G2cruise == 1003) & (A29.G2station == i))]  # 1003.[1:8]
for i in range(562,567):
    A29 = A29[~((A29.G2cruise == 1104) & (A29.G2station == i))]  # 1104.[562:566]

AR07E = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['AR07E'])] # no trimming needed

AR07W = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['AR07W'])]
for i in range(658,672):
    AR07W = AR07W[~((AR07W.G2cruise == 26) & (AR07W.G2station == i))] # 26.[658:671]
for i in range(712,755):
    AR07W = AR07W[~((AR07W.G2cruise == 26) & (AR07W.G2station == i))] # 26.[712:754]
for i in range(506,552):
    AR07W = AR07W[~((AR07W.G2cruise == 38) & (AR07W.G2station == i))] # 38.[506:551]
for i in range(352,368):
    AR07W = AR07W[~((AR07W.G2cruise == 44) & (AR07W.G2station == i))] # 44.[352:367]
for i in range(381,451):
    AR07W = AR07W[~((AR07W.G2cruise == 44) & (AR07W.G2station == i))] # 44.[381:450]
for i in range(1,34):
    AR07W = AR07W[~((AR07W.G2cruise == 151) & (AR07W.G2station == i))] # 151.[1:33]
AR07W = AR07W[~((AR07W.G2cruise == 155) & (AR07W.G2station == 66))]  # 155.66
for i in range(44,86):
    AR07W = AR07W[~((AR07W.G2cruise == 158) & (AR07W.G2station == i))] # 158.[44:85]
for i in range(24,44):
    AR07W = AR07W[~((AR07W.G2cruise == 106) & (AR07W.G2station == i))] # 160.[24:43]
for i in range(78,131):
    AR07W = AR07W[~((AR07W.G2cruise == 160) & (AR07W.G2station == i))] # 160.[78:130]
for i in range(241,257):
    AR07W = AR07W[~((AR07W.G2cruise == 164) & (AR07W.G2station == i))] # 164.[241:256]
for i in range(48,95):
    AR07W = AR07W[~((AR07W.G2cruise == 165) & (AR07W.G2station == i))] # 165.[48:94]
for i in range(262,282):
    AR07W = AR07W[~((AR07W.G2cruise == 165) & (AR07W.G2station == i))] # 165.[262:281]
for i in range(434,465):
    AR07W = AR07W[~((AR07W.G2cruise == 166) & (AR07W.G2station == i))] # 166.[434:464]
for i in range(287,290):
    AR07W = AR07W[~((AR07W.G2cruise == 167) & (AR07W.G2station == i))] # 167.[287:289]
AR07W = AR07W[~((AR07W.G2cruise == 171) & (AR07W.G2station == 3))] # 171.3
for i in range(225,260):
    AR07W = AR07W[~((AR07W.G2cruise == 171) & (AR07W.G2station == i))] # 171.[225:259]
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 216))] # 174.216
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 219))] # 174.219
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 222))] # 174.222
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 227))] # 174.227
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 234))] # 174.234
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 245))] # 174.245
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 252))] # 174.252
AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == 262))] # 174.262
for i in range(271,302):
    AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == i))] # 174.[271:301]
for i in range(344,382):
    AR07W = AR07W[~((AR07W.G2cruise == 174) & (AR07W.G2station == i))]# 174.[344:381]
for i in range(39,75):
    AR07W = AR07W[~((AR07W.G2cruise == 698) & (AR07W.G2station == i))] # 698.[39:74]
AR07W = AR07W[~((AR07W.G2cruise == 1025) & (AR07W.G2station == 15))] # 1025.15
for i in range(181,251):
    AR07W = AR07W[~((AR07W.G2cruise == 1025) & (AR07W.G2station == i))] # 1025.[181:250]
for i in range(196,212):
    AR07W = AR07W[~((AR07W.G2cruise == 1027) & (AR07W.G2station == i))] # 1027.[196:211]
for i in range(1,137):
    AR07W = AR07W[~((AR07W.G2cruise == 2011) & (AR07W.G2station == i))] # 2011.[1:136]
AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == 2))] # 4005.2
for i in range(10,30):
    AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == i))] # 4005.[10:29]
AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == 161))] # 4005.161
AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == 166))] # 4005.166
for i in range(170,189):
    AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == i))] # 4005.[170:188]
AR07W = AR07W[~((AR07W.G2cruise == 4005) & (AR07W.G2station == 191))] # 4005.191

ARC01E = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['ARC01E'])]

ARC01W = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['ARC01W'])]

MED01 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['MED01'])] # no trimming needed

I01 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I01'])]
for i in range(859,893):
    I01 = I01[~((I01.G2cruise == 255) & (I01.G2station == i))] # 255.[859:892]
    
I03 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I03'])]
for i in range(586,608):
    I03 = I03[~((I03.G2cruise == 488) & (I03.G2station == i))] # 488.[586:607]

I05 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I05'])]
for i in range(283,395):
    I05 = I05[~((I05.G2cruise == 251) & (I05.G2station == i))] # 251.[283:394]
for i in range(574,611):
    I05 = I05[~((I05.G2cruise == 253) & (I05.G2station == i))] # 253.[574:610]
for i in range(670,709):
    I05 = I05[~((I05.G2cruise == 253) & (I05.G2station == i))] # 253.[670:708]
for i in range(1,14):
    I05 = I05[~((I05.G2cruise == 682) & (I05.G2station == i))] # 682.[1:13]

I06 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I06'])]
for i in range(77,100):
    I06 = I06[~((I06.G2cruise == 354) & (I06.G2station == i))] # 354.[77:99]
for i in range(95,97):
    I06 = I06[~((I06.G2cruise == 374) & (I06.G2station == i))] # 374.[95:96]
for i in range(1,8):
    I06 = I06[~((I06.G2cruise == 3033) & (I06.G2station == i))] # 3033.[1:7]
    
I07 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I07'])]
I07 = I07[~((I07.G2cruise == 254) & (I07.G2station == 708))] # 254.708
for i in range(811,856):
    I07 = I07[~((I07.G2cruise == 254) & (I07.G2station == i))] # 254.[811:855]
for i in range(112,125):
    I07 = I07[~((I07.G2cruise == 3034) & (I07.G2station == i))] # 3034.[112:124]

I08N = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I08N'])]
for i in range(355,441):
    I08N = I08N[~((I08N.G2cruise == 251) & (I08N.G2station == i))] # 251.[355:440]
for i in range(1,33):
    I08N = I08N[~((I08N.G2cruise == 339) & (I08N.G2station == i))] # 339.[1:32]

I08S = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I08S'])]
for i in range(85,146):
    I08S = I08S[~((I08S.G2cruise == 249) & (I08S.G2station == i))] # 249.[85:145]
for i in range(1,9):
    I08S = I08S[~((I08S.G2cruise == 352) & (I08S.G2station == i))] # 352.[1:8]

I09N = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I09N'])]
for i in range(148,150):
    I09N = I09N[~((I09N.G2cruise == 250) & (I09N.G2station == i))] # 250.[148:149]
for i in range(269,278):
    I09N = I09N[~((I09N.G2cruise == 250) & (I09N.G2station == i))] # 250.[269:277]
for i in range(175,181):
    I09N = I09N[~((I09N.G2cruise == 353) & (I09N.G2station == i))] # 353.[175:180]
for i in range(170,178):
    I09N = I09N[~((I09N.G2cruise == 3035) & (I09N.G2station == i))] # 3035.[170:177]

I09S = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I09S'])]
for i in range(72,114):
    I09S = I09S[~((I09S.G2cruise == 72) & (I09S.G2station == i))] # 72.[72:113]
for i in range(1,85):
    I09S = I09S[~((I09S.G2cruise == 249) & (I09S.G2station == i))] # 249.[1:84]

I10 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['I10'])]
for i in range(45,144):
    I10 = I10[~((I10.G2cruise == 82) & (I10.G2station == i))] # 82.[45:143]

P01 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P01'])]
for i in range(1,5):
    P01 = P01[~((P01.G2cruise == 461) & (P01.G2station == i))] # 461.[1:4]
P01 = P01[~((P01.G2cruise == 461) & (P01.G2station == 135))] # 461.135
for i in range(2,129):
    P01 = P01[~((P01.G2cruise == 504) & (P01.G2station == i))] # 504.[37:128]
P01 = P01[~((P01.G2cruise == 504) & (P01.G2station == 1002))] # 504.1002
P01 = P01[~((P01.G2cruise == 504) & (P01.G2station == 1004))] # 504.1004
P01 = P01[~((P01.G2cruise == 1053) & (P01.G2station == 151))] # 1053.151

P02 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P02'])] # no trimming needed

P03 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P03'])] # no trimming needed

P04 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P04'])] # no trimming needed

P06 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P06'])] # no trimming needed

P09 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P09'])]
for i in range(71,80):
    P09 = P09[~((P09.G2cruise == 609) & (P09.G2station == i))] # 609.[71:79]
for i in range(5486,5549):
    P09 = P09[~((P09.G2cruise == 1100) & (P09.G2station == i))] # 1100.[5486:5548]
for i in range(4974,5006):
    P09 = P09[~((P09.G2cruise == 2075) & (P09.G2station == i))] # 2075.[4974:5005]

P10 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P10'])]
for i in range(80,91):
    P10 = P10[~((P10.G2cruise == 302) & (P10.G2station == i))] # 302.[80:90]

P13 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P13'])]
for i in range(1,4):
    P13 = P13[~((P13.G2cruise == 431) & (P13.G2station == i))] # 431.[1:3]
for i in range(5,47):
    P13 = P13[~((P13.G2cruise == 431) & (P13.G2station == i))] # 431.[5:36]
for i in range(62,65):
    P13 = P13[~((P13.G2cruise == 431) & (P13.G2station == i))] # 431.[62:64]
for i in range(252,255):
    P13 = P13[~((P13.G2cruise == 431) & (P13.G2station == i))] # 431.[252:254]
for i in range(1,3):
    P13 = P13[~((P13.G2cruise == 439) & (P13.G2station == i))] # 439.[1:2]
P13 = P13[~((P13.G2cruise == 439) & (P13.G2station == 16))] # 439.16
P13 = P13[~((P13.G2cruise == 439) & (P13.G2station == 24))] # 439.24
P13 = P13[~((P13.G2cruise == 439) & (P13.G2station == 71))] # 439.71
P13 = P13[~((P13.G2cruise == 440) & (P13.G2station == 0))] # 440.0
for i in range(1,43):
    P13 = P13[~((P13.G2cruise == 517) & (P13.G2station == i))] # 517.[1:42]
P13 = P13[~((P13.G2cruise == 598) & (P13.G2station == 2428))] # 598.2428
P13 = P13[~((P13.G2cruise == 598) & (P13.G2station == 2432))] # 598.2432

P14 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P14'])]
for i in range(201,214):
    P14 = P14[~((P14.G2cruise == 268) & (P14.G2station == i))] # 268.[201:213]
for i in range(220,240):
    P14 = P14[~((P14.G2cruise == 268) & (P14.G2station == i))] # 268.[220:239]
for i in range(253,348):
    P14 = P14[~((P14.G2cruise == 268) & (P14.G2station == i))] # 268.[253:347]
for i in range(1,5):
    P14 = P14[~((P14.G2cruise == 280) & (P14.G2station == i))] # 280.[1:4]
for i in range(22,183):
    P14 = P14[~((P14.G2cruise == 280) & (P14.G2station == i))] # 280.[22:182]
for i in range(128,160):
    P14 = P14[~((P14.G2cruise == 504) & (P14.G2station == i))] # 504.[128:159]
P14 = P14[~((P14.G2cruise == 504) & (P14.G2station == 1113))] # 504.1113
for i in range(50,88):
    P14 = P14[~((P14.G2cruise == 1050) & (P14.G2station == i))] # 1050.[50:87]
for i in range(404,411):
    P14 = P14[~((P14.G2cruise == 1050) & (P14.G2station == i))] # 1050.[404:410]
for i in range(501,504):
    P14 = P14[~((P14.G2cruise == 1050) & (P14.G2station == i))] # 1050.[501:503]

P15 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P15'])]
for i in range(74,84):
    P15 = P15[~((P15.G2cruise == 83) & (P15.G2station == i))] # 83.[74:83]
P15 = P15[~((P15.G2cruise == 83) & (P15.G2station == 129))] # 83.129
for i in range(119,129):
    P15 = P15[~((P15.G2cruise == 84) & (P15.G2station == i))] # 84.[119:128]
for i in range(11,33):
    P15 = P15[~((P15.G2cruise == 280) & (P15.G2station == i))] # 280.[11:32]

P16N = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P16N'])]
for i in range(24,29):
    P16N = P16N[~((P16N.G2cruise == 304) & (P16N.G2station == i))] # 304.[24:28]
for i in range(1,16):
    P16N = P16N[~((P16N.G2cruise == 307) & (P16N.G2station == i))] # 307.[1:15]
P16N = P16N[~((P16N.G2cruise == 307) & (P16N.G2station == 999))] # 307.999
for i in range(191,208):
    P16N = P16N[~((P16N.G2cruise == 1044) & (P16N.G2station == i))] # 1044.[191:207]

P16S = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P16S'])]
for i in range(1,5):
    P16S = P16S[~((P16S.G2cruise == 1036) & (P16S.G2station == i))] # 1036.[1:4]

P17E = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P17E'])] # no trimming needed

P17N = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P17N'])]
for i in range(10,26):
    P17N = P17N[~((P17N.G2cruise == 300) & (P17N.G2station == i))] # 300.[10:25]
for i in range(122,139):
    P17N = P17N[~((P17N.G2cruise == 300) & (P17N.G2station == i))] # 300.[122:138]

P18 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P18'])]
for i in range(173,175):
    P18 = P18[~((P18.G2cruise == 345) & (P18.G2station == i))] # 345.[173:174]
P18 = P18[~((P18.G2cruise == 345) & (P18.G2station == 996))] # 345.996
for i in range(117,126):
    P18 = P18[~((P18.G2cruise == 1045) & (P18.G2station == i))] # 1045.[117:125]
for i in range(209,213):
    P18 = P18[~((P18.G2cruise == 1045) & (P18.G2station == i))] # 1045.[209:212]

P21 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P21'])] # no trimming needed

S04I = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['S04I'])]
for i in range(6,14):
   S04I = S04I[~((S04I.G2cruise == 67) & (S04I.G2station == i))] # 67.[6:13]
for i in range(54,65):
   S04I = S04I[~((S04I.G2cruise == 67) & (S04I.G2station == i))] # 67.[54:64]
for i in range(70,108):
   S04I = S04I[~((S04I.G2cruise == 67) & (S04I.G2station == i))] # 67.[70:107]
for i in range(1,9):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[1:8]
for i in range(35,45):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[35:44]
for i in range(52,72):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[52:71]
for i in range(75,86):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[75:85]
for i in range(90,103):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[90:102]
for i in range(108,121):
   S04I = S04I[~((S04I.G2cruise == 73) & (S04I.G2station == i))] # 73.[108:120]
for i in range(35,41):
   S04I = S04I[~((S04I.G2cruise == 288) & (S04I.G2station == i))] # 288.[35:40]
for i in range(65,80):
   S04I = S04I[~((S04I.G2cruise == 288) & (S04I.G2station == i))] # 288.[65:79]
for i in range(9,20):
   S04I = S04I[~((S04I.G2cruise == 1050) & (S04I.G2station == i))] # 1050.[9:19]
for i in range(502,504):
   S04I = S04I[~((S04I.G2cruise == 1050) & (S04I.G2station == i))] # 1050.[502:503]
for i in range(135,141):
   S04I = S04I[~((S04I.G2cruise == 1051) & (S04I.G2station == i))] # 1051.[135:140]
for i in range(162,165):
   S04I = S04I[~((S04I.G2cruise == 1051) & (S04I.G2station == i))] # 1051.[162:164]

SR04 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['SR04'])]
for i in range(99,185):
   SR04 = SR04[~((SR04.G2cruise == 19) & (SR04.G2station == i))] # 19.[99:184]
for i in range(21,67):
   SR04 = SR04[~((SR04.G2cruise == 20) & (SR04.G2station == i))] # 20.[21:66]
   
S04P = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['S04P'])]
for i in range(47,95):
   S04P = S04P[~((S04P.G2cruise == 295) & (S04P.G2station == i))] # 295.[47:94]
S04P = S04P[~((S04P.G2cruise == 295) & (S04P.G2station == 118))] # 295.118
for i in range(10,17):
   S04P = S04P[~((S04P.G2cruise == 3031) & (S04P.G2station == i))] # 3031.[10:16]
for i in range(31,53):
   S04P = S04P[~((S04P.G2cruise == 3031) & (S04P.G2station == i))] # 3031.[31:52]
for i in range(98,101):
   S04P = S04P[~((S04P.G2cruise == 3031) & (S04P.G2station == i))] # 3031.[98:100]

SR01 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['SR01'])]
for i in range(99,224):
   SR01 = SR01[~((SR01.G2cruise == 19) & (SR01.G2station == i))] # 19.[99:223]

SR03 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['SR03'])]
for i in range(1,58):
   SR03 = SR03[~((SR03.G2cruise == 67) & (SR03.G2station == i))] # 67.[1:57]
for i in range(4,14):
   SR03 = SR03[~((SR03.G2cruise == 68) & (SR03.G2station == i))] # 68.[4:14]
for i in range(1,50):
   SR03 = SR03[~((SR03.G2cruise == 69) & (SR03.G2station == i))] # 69.[1:49]
SR03 = SR03[~((SR03.G2cruise == 69) & (SR03.G2station == 67))] # 69.67
SR03 = SR03[~((SR03.G2cruise == 70) & (SR03.G2station == 17))] # 70.17
for i in range(89,122):
   SR03 = SR03[~((SR03.G2cruise == 70) & (SR03.G2station == i))] # 70.[89:121]
SR03 = SR03[~((SR03.G2cruise == 75) & (SR03.G2station == 2))] # 75.2
for i in range(61,107):
   SR03 = SR03[~((SR03.G2cruise == 2008) & (SR03.G2station == i))] # 2008.[61:106]

# %% test with P06, no trimming, to figure out what type of figures would be useful and how to store trimmed output
nums = go_ship_cruise_nums_2023['P16S']
#nums = [1050]
#transect = espers[espers["G2cruise"].isin(nums)]
#transect = glodap[glodap["G2cruise"].isin(nums)]
#transect = transect[(transect.G2talkf == 2)]
transect = P21

# set up map
fig = plt.figure(figsize=(12,7))
#ax = plt.axes(projection=ccrs.PlateCarree()) # atlantic-centered view 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180)) # paciifc-centered view
ax.coastlines(resolution='110m',color='k')
g1 = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,alpha=0)
g1.top_labels = False
g1.right_labels = False
ax.add_feature(cfeature.LAND,color='k')
#ax.set_title('Transect P06')
#extent = [-102.5, -90, -75, -20]
extent = [120, -60, -45, -0]
#extent = [-180, 180, -90, 90]
ax.set_extent(extent)

# get data from glodap
lon = transect.G2longitude
lat = transect.G2latitude
plot = ax.scatter(lon,lat,transform=ccrs.PlateCarree(),marker='x',s=20)
# %% return station number that is inside of lat/lon bounds
bad_area = extent
bad_points = transect[(transect.G2longitude > bad_area[0]) & (transect.G2longitude < bad_area[1]) & (transect.G2latitude > bad_area[2]) & (transect.G2latitude < bad_area[3])]

bad_points = bad_points.groupby(['G2cruise','G2station']).size().reset_index().rename(columns={0:'count'})
print(bad_points)
