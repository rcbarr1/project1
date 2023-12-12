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
from scipy import special


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
      

def go_ship_only(glodap):
    """
    Subsets GLODAPv2.2023 for data that is along a WOCE, CLIVAR, GO-SHIP,
    SOCCOM, TTO, GEOSECS, or OVIDE transect. Additionally, outputs a dictionary
    with all transects as keys where associated values are the GLODAPv2.2023
    G2cruise numbers of all cruises that contain data along all or part of that
    transect.
    
    Keyword arguments:
        glodap = pandas dataframe containing glodap dataset with original 
                 headers
        
    Returns:
        go_ship = subset of "glodap" dataframe that only contains entries from
                 WOCE, CLIVAR, GO-SHIP, SOCCOM, TTO, GEOSECS, or OVIDE cruises
                 
        go_ship_cruise_nums_2023 = dictionary where keys are transects and
                 values are G2cruise numbers, as defined by GLODAPv2.2023, that
                 contain data along all or part of the associated transect
    """
    go_ship_cruise_nums_2023 = {'A02' : [24, 37, 43, 1006, 2027],
                         'A05' : [225, 341, 695, 699, 1030, 1109],
                         'A10' : [34, 347, 487, 676, 1008, 2105],
                         'A12' : [6, 11, 13, 14, 15, 18, 19, 20, 28, 233, 385, 1004],
                         'A135' : [239, 346, 1004],
                         #'A16' : [63, 322, 336, 338, 342, 343, 366, 694, 700, 1041, 1042],
                         'A16N' : [63, 322, 338, 342, 366, 694, 700, 1041],
                         'A16S' : [336, 343, 1042],
                         'A17' : [229, 230, 235, 236, 297, 2013],
                         'A20' : [236, 260, 264, 330, 5005],
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
                         'P02' : [272, 459, 518, 1035],
                         'P03' : [298, 497, 5017],
                         'P04' : [319],
                         'P06' : [243, 486, 273, 3029, 3030],
                         'P09' : [412, 515, 546, 547, 549, 550, 552, 554, 555, 556, 558, 559, 561, 562, 564, 565, 566, 568, 570, 571, 573, 576, 581, 583, 592, 595, 596, 599, 600, 603, 604, 607, 608, 609, 1056, 1057, 1058, 1067, 1071, 1079, 1080, 1082, 1083, 1087, 1090, 1093, 1100, 1101, 2041, 2047, 2057, 2062, 2067, 2075, 2080, 2087, 2099, 4066, 4068, 4069, 4071, 4078, 4089],
                         'P10' : [302, 495, 553, 557, 560, 563, 1087, 1090, 1093, 1098, 1099, 2050, 2057, 2062, 2075, 2087],
                         'P13' : [296, 360, 431, 437, 439, 440, 517, 545, 548, 551, 553, 557, 560, 563, 567, 569, 572, 574, 575, 577, 579, 580, 582, 584, 585, 586, 587, 588, 589, 590, 591, 593, 594, 597, 598, 601, 602, 605, 606, 1058, 1060, 1063, 1064, 1066, 1069, 1071, 1076, 1078, 1079, 1081, 1092, 2038, 2041, 2047, 2054, 2064, 2084, 2091, 2094, 2096, 2097, 2102, 2103, 4063, 4069, 4074, 4076, 4081, 4083, 4087],
                         'P14' : [268, 280, 301, 504, 505, 1050],
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
    
    # select all cruises from above
    flat_nums = [element for sublist in (list(go_ship_cruise_nums_2023.values())) for element in sublist] # flatten dictionary into one list
    flat_nums = list(set(flat_nums)) # remove duplicates from list
    go_ship = glodap[glodap["G2cruise"].isin(flat_nums)]    
    
    return go_ship, go_ship_cruise_nums_2023
    
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
    glodap = glodap[(glodap['G2salinity'] >= 30) & (glodap['G2salinity'] <= 37)]
    
    # reset index
    glodap_out = glodap.reset_index(drop=True)
    
    return glodap_out
        
def kl_divergence(espers):
    """
    Uses KL divergence to determine which equations predict best, where a lower
    KL divergence means that the two datasets are closer.
    
    Keyword arguments:
        espers = ESPER prediction dataframe for all 16 equations for each of
                 the three methods
        
    Returns:
        kl_div = array of KL divergences for each of the equation/method
                 combinations.

    """
    kl_div = np.zeros([16,3])

    for j in range(0,3):
        if j == 0:
            esper_type = 'LIRtalk'
        elif j == 1:
            esper_type = 'NNtalk'
        else:
            esper_type = 'Mtalk'
            
        for i in range(1,17):
            LIR_name = esper_type + str(i)
            ab = espers[['G2talk', LIR_name]].dropna(axis=0)
            a = np.asarray(ab.G2talk, dtype=float)
            a /= np.sum(a)
            b = np.asarray(ab[LIR_name], dtype=float)
            b /= np.sum(b)
        
            vec = special.rel_entr(a,b)
            kl_div[i-1,j] = np.sum(vec)
            
    kl_div = pd.DataFrame(kl_div, columns = ['LIR','NN','Mixed'])
    kl_div.index += 1
    
    return kl_div
    
    
    
        