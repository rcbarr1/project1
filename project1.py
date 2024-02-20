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
    Subsets GLODAPv2.2023 for data that is along a WOCE, CLIVAR, GO-SHIP, or
    SOCCOM transect. Additionally, outputs a dictionary
    with all transects as keys where associated values are the GLODAPv2.2023
    G2cruise numbers of all cruises that contain data along all or part of that
    transect. Marks cruises in Northwest Pacific conducted by Japanese
    researchers with "_J" to indicate that they may not be comparable to U.S.
    cruise data due to methodological differences in TA measurements.
    
    Keyword arguments:
        glodap = pandas dataframe containing glodap dataset with original 
                 headers
        
    Returns:
        go_ship = subset of "glodap" dataframe that only contains entries from
                 WOCE, CLIVAR, GO-SHIP, or SOCCOM cruises
                 
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
                         'P02' : [272, 1035],
                         'P02_J' : [406, 407, 408, 459],
                         'P03' : [],
                         'P03_J' : [497, 1070, 1086, 1096, 2098, 5017],
                         'P04' : [319],
                         'P06' : [243, 486, 273, 3029, 3030],
                         'P09' : [],
                         'P09_J' : [412, 515, 546, 547, 549, 550, 552, 554, 555, 556, 558, 559, 561, 562, 564, 565, 566, 568, 570, 571, 573, 576, 581, 583, 592, 595, 596, 599, 600, 603, 604, 607, 608, 609, 1056, 1057, 1058, 1067, 1071, 1079,  1080, 1082, 1083, 1087, 1090, 1093, 1100, 1101, 2041, 2047, 2057, 2062, 2067, 2075, 2080, 2087, 2099, 4066, 4068, 4069, 4071, 4078, 4089],
                         'P10' : [302],
                         'P10_J' : [495, 553, 557, 560, 563, 594, 1087, 1090, 1093, 1098, 1099, 2050, 2057, 2062, 2075, 2087, 4066],
                         'P13' : [296],
                         'P13_J' : [439, 440, 517, 545, 548, 551, 553, 557, 560, 563, 567, 569, 572, 574, 575, 577, 579, 580, 582, 584, 585, 586, 587, 588, 589, 590, 591, 593, 594, 597, 598, 601, 602, 605, 606, 1058, 1060, 1063, 1064, 1066, 1069, 1071, 1076, 1078, 1079, 1081, 1092, 2038, 2041, 2047, 2054, 2064, 2084, 2091, 2094, 2096, 2097, 2102, 2103, 4063, 4069, 4074, 4076, 4081, 4083, 4087],
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
    
    # ignore rows that have TA qc as 0
    glodap = glodap[glodap['G2talkqc'] != 0]
    
    #filter data with salinity to only look at open ocean
    glodap = glodap[(glodap['G2salinity'] >= 30) & (glodap['G2salinity'] <= 37)]
    
    # reset index
    glodap_out = glodap.reset_index(drop=True)
    
    return glodap_out
        
def ensemble_mean(espers):
    """
    Calculates ensemble mean of ESPER alkalinity predictions and mean of
    uncertainties of alkalinity predictions.
    
    Keyword arguments:
        espers = ESPER prediction dataframe for all 16 equations for each of
                 the three methods
                 
    Returns:
        espers_out = "espers" dataframe with added columns "Ensemble_Mean_TA"
                      and "Ensemble_Mean_TA_Uncert"
        
    """
    
    talk_subset_LIR = espers[['LIRtalk1', 'LIRtalk2', 'LIRtalk3', 'LIRtalk4', 
                              'LIRtalk5', 'LIRtalk6', 'LIRtalk7', 'LIRtalk8',
                              'LIRtalk9', 'LIRtalk10', 'LIRtalk11',
                              'LIRtalk12', 'LIRtalk13', 'LIRtalk14',
                              'LIRtalk15', 'LIRtalk16']]
    
    if 'NNtalk1' in espers.columns:
        talk_subset_NN = espers[['NNtalk1', 'NNtalk2', 'NNtalk3', 'NNtalk4',
                                 'NNtalk5', 'NNtalk6', 'NNtalk7', 'NNtalk8',
                                 'NNtalk9', 'NNtalk10', 'NNtalk11', 'NNtalk12',
                                 'NNtalk13', 'NNtalk14', 'NNtalk15', 'NNtalk16']]
    
        talk_subset_M = espers[['Mtalk1', 'Mtalk2', 'Mtalk3', 'Mtalk4',
                                'Mtalk5', 'Mtalk6', 'Mtalk7', 'Mtalk8', 
                                'Mtalk9', 'Mtalk10', 'Mtalk11', 'Mtalk12',
                                'Mtalk13', 'Mtalk14', 'Mtalk15', 'Mtalk16']]

    uncertainty_subset_LIR = espers[['LIRtalk_uncert1', 'LIRtalk_uncert2',
                                     'LIRtalk_uncert3', 'LIRtalk_uncert4',
                                     'LIRtalk_uncert5', 'LIRtalk_uncert6',
                                     'LIRtalk_uncert7', 'LIRtalk_uncert8',
                                     'LIRtalk_uncert9', 'LIRtalk_uncert10',
                                     'LIRtalk_uncert11', 'LIRtalk_uncert12',
                                     'LIRtalk_uncert13', 'LIRtalk_uncert14',
                                     'LIRtalk_uncert15', 'LIRtalk_uncert16']]
    
    if 'NNtalk_uncert1' in espers.columns:
        uncertainty_subset_NN = espers[['NNtalk_uncert1', 'NNtalk_uncert2',
                                        'NNtalk_uncert3', 'NNtalk_uncert4',
                                        'NNtalk_uncert5', 'NNtalk_uncert6',
                                        'NNtalk_uncert7', 'NNtalk_uncert8',
                                        'NNtalk_uncert9', 'NNtalk_uncert10',
                                        'NNtalk_uncert11', 'NNtalk_uncert12',
                                        'NNtalk_uncert13', 'NNtalk_uncert14',
                                        'NNtalk_uncert15', 'NNtalk_uncert16']]
    
        uncertainty_subset_M = espers[['Mtalk_uncert1', 'Mtalk_uncert2',
                                       'Mtalk_uncert3', 'Mtalk_uncert4',
                                       'Mtalk_uncert5', 'Mtalk_uncert6',
                                       'Mtalk_uncert7', 'Mtalk_uncert8',
                                       'Mtalk_uncert9', 'Mtalk_uncert10',
                                       'Mtalk_uncert11', 'Mtalk_uncert12', 
                                       'Mtalk_uncert13', 'Mtalk_uncert14',
                                       'Mtalk_uncert15', 'Mtalk_uncert16']]

    espers['Ensemble_Mean_TA_LIR'] = talk_subset_LIR.mean(axis=1)
    espers['Ensemble_Std_TA_LIR'] = talk_subset_LIR.std(axis=1)
    espers['Ensemble_Mean_TA_Uncert_LIR'] = uncertainty_subset_LIR.mean(axis=1)
    if 'NNtalk1' in espers.columns:
        espers['Ensemble_Mean_TA_NN'] = talk_subset_NN.mean(axis=1)
        espers['Ensemble_Std_TA_NN'] = talk_subset_NN.std(axis=1)
        espers['Ensemble_Mean_TA_Uncert_NN'] = uncertainty_subset_NN.mean(axis=1)
        espers['Ensemble_Mean_TA_Mixed'] = talk_subset_M.mean(axis=1)
        espers['Ensemble_Std_TA_Mixed'] = talk_subset_M.std(axis=1)
        espers['Ensemble_Mean_TA_Uncert_Mixed'] = uncertainty_subset_M.mean(axis=1)
    
    espers_out = espers
    
    return espers_out

def trim_go_ship(espers, go_ship_cruise_nums_2023):
    """
    1. Selects only measured (not calculated) alkalinity
    2. Subsets data to only use open ocean salinity
    3. Save in a dataframe for each transect
    
    Keyword arguments:
        espers = ESPER prediction dataframe for all 16 equations for each of
                 the three methods
        go_ship_cruise_nums_2023 = dictionary where keys are transects and
                 values are G2cruise numbers, as defined by GLODAPv2.2023, that
                 contain data along all or part of the associated transect
                 (from go_ship_only function)
        
        
    Returns:
        trimmed = dictionary of all transect dataframes
        
    """
    # XX.YY = cruise.station to trim

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
    for i in range(42,59):
        A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[42:58]
    for i in range(118,142):
        A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[118:141]
    for i in range(172,236):
        A17 = A17[~((A17.G2cruise == 297) & (A17.G2station == i))] # trim 297.[172:235]
    for i in range(77,82):
        A17 = A17[~((A17.G2cruise == 2013) & (A17.G2station == i))] # trim 2013.[77:81]

    A20 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['A20'])] # no trimming needed

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

    P02_J = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P02_J'])] # no trimming needed

    P03 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P03'])] # no trimming needed

    P03_J = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P03_J'])] # no trimming needed
    for i in range(209,219):
        P03_J = P03_J[~((P03_J.G2cruise == 497) & (P03_J.G2station == i))] # 497.[209:218]
    for i in range(3851,3867):
        P03_J = P03_J[~((P03_J.G2cruise == 1086) & (P03_J.G2station == i))] # 1086.[3851:3866]
    for i in range(4867,4878):
        P03_J = P03_J[~((P03_J.G2cruise == 1096) & (P03_J.G2station == i))] # 1096.[4867:4877]

    P06 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P06'])] # no trimming needed

    P09 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P09'])] # no trimming needed
        
    P09_J = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P09_J'])] # FINISH THIS
    for i in range(71,80):
        P09_J = P09_J[~((P09_J.G2cruise == 609) & (P09_J.G2station == i))] # 609.[71:79]
    for i in range(3056,3070):
        P09_J = P09_J[~((P09_J.G2cruise == 1067) & (P09_J.G2station == i))] # 1067.[3056:3069]
    for i in range(3482,3521):
        P09_J = P09_J[~((P09_J.G2cruise == 1071) & (P09_J.G2station == i))] # 1071.[3482:3520]
    for i in range(3958,4000):
        P09_J = P09_J[~((P09_J.G2cruise == 1079) & (P09_J.G2station == i))] # 1079.[3958:3999]
    for i in range(3920,3964):
        P09_J = P09_J[~((P09_J.G2cruise == 1087) & (P09_J.G2station == i))] # 1087.[3920:3963]
    for i in range(4273,4305):
        P09_J = P09_J[~((P09_J.G2cruise == 1090) & (P09_J.G2station == i))] # 1090.[4273:4304]
    for i in range(4624,4650):
        P09_J = P09_J[~((P09_J.G2cruise == 1093) & (P09_J.G2station == i))] # 1093.[4624:4649]
    for i in range(5486,5549):
        P09_J = P09_J[~((P09_J.G2cruise == 1100) & (P09_J.G2station == i))] # 1100.[5486:5548]
    for i in range(4237,4279):
        P09_J = P09_J[~((P09_J.G2cruise == 2041) & (P09_J.G2station == i))] # 2041.[4237:4278]
    for i in range(4535,4577):
        P09_J = P09_J[~((P09_J.G2cruise == 2047) & (P09_J.G2station == i))] # 2047.[4535:4576]
    for i in range(4995,5026):
        P09_J = P09_J[~((P09_J.G2cruise == 2057) & (P09_J.G2station == i))] # 2057.[4995:5025]
    for i in range(5183,5192):
        P09_J = P09_J[~((P09_J.G2cruise == 2062) & (P09_J.G2station == i))] # 2062.[5183:5191]
    for i in range(4974,5006):
        P09_J = P09_J[~((P09_J.G2cruise == 2075) & (P09_J.G2station == i))] # 2075.[4974:5005]
    for i in range(5330,5347):
        P09_J = P09_J[~((P09_J.G2cruise == 2080) & (P09_J.G2station == i))] # 2080.[5330:5346]
    for i in range(5641,5655):
        P09_J = P09_J[~((P09_J.G2cruise == 2087) & (P09_J.G2station == i))] # 2087.[5641:5654]
    for i in range(6174,6191):
        P09_J = P09_J[~((P09_J.G2cruise == 2099) & (P09_J.G2station == i))] # 2099.[6174:6190]
    for i in range(5695,5781):
        P09_J = P09_J[~((P09_J.G2cruise == 4066) & (P09_J.G2station == i))] # 4066.[5695:5780]
    for i in range(5769,5781):
        P09_J = P09_J[~((P09_J.G2cruise == 4069) & (P09_J.G2station == i))] # 4069.[5769:5780]
    for i in range(6413,6418):
        P09_J = P09_J[~((P09_J.G2cruise == 4078) & (P09_J.G2station == i))] # 4078.[6413:6417]

    P10 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P10'])]
    for i in range(80,91):
        P10 = P10[~((P10.G2cruise == 302) & (P10.G2station == i))] # 302.[80:90]

    P10_J = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P10_J'])] # DO THIS
    P10_J = P10_J[~((P10_J.G2cruise == 495) & (P10_J.G2station == 1003))] # 495.1003
    for i in range(3870,3964):
        P10_J = P10_J[~((P10_J.G2cruise == 1087) & (P10_J.G2station == i))] # 1087.[3870:3963]
    for i in range(4254,4305):
        P10_J = P10_J[~((P10_J.G2cruise == 1090) & (P10_J.G2station == i))] # 1090.[4254:4304]
    for i in range(4586,4650):
        P10_J = P10_J[~((P10_J.G2cruise == 1093) & (P10_J.G2station == i))] # 1093.[4586:4649]
    P10_J = P10_J[~((P10_J.G2cruise == 1098) & (P10_J.G2station == 5163))] # 1098.5163
    for i in range(5220,5238):
        P10_J = P10_J[~((P10_J.G2cruise == 1099) & (P10_J.G2station == i))] # 1099.[5220:5237]
    for i in range(4663,4727):
        P10_J = P10_J[~((P10_J.G2cruise == 2050) & (P10_J.G2station == i))] # 2050.[4663:4726]
    for i in range(4955,5026):
        P10_J = P10_J[~((P10_J.G2cruise == 2057) & (P10_J.G2station == i))] # 2057.[4955:5025]
    for i in range(5195,5233):
        P10_J = P10_J[~((P10_J.G2cruise == 2062) & (P10_J.G2station == i))] # 2062.[5195:5232]
    for i in range(4924,5006):
        P10_J = P10_J[~((P10_J.G2cruise == 2075) & (P10_J.G2station == i))] # 2075.[4924:5005]
    for i in range(5591,5638):
        P10_J = P10_J[~((P10_J.G2cruise == 2087) & (P10_J.G2station == i))] # 2087.[5591:5637]
    for i in range(5654,5692):
        P10_J = P10_J[~((P10_J.G2cruise == 4066) & (P10_J.G2station == i))] # 4066.[5654:5691]

    P13 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P13'])]

    P13_J = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P13_J'])] # FINISH THIS
    for i in range(1,3):
        P13_J = P13_J[~((P13_J.G2cruise == 439) & (P13_J.G2station == i))] # 439.[1:2]
    P13_J = P13_J[~((P13_J.G2cruise == 439) & (P13_J.G2station == 16))] # 439.16
    P13_J = P13_J[~((P13_J.G2cruise == 439) & (P13_J.G2station == 24))] # 439.24
    P13_J = P13_J[~((P13_J.G2cruise == 439) & (P13_J.G2station == 71))] # 439.71
    P13_J = P13_J[~((P13_J.G2cruise == 440) & (P13_J.G2station == 0))] # 440.0
    for i in range(1,43):
        P13_J = P13_J[~((P13_J.G2cruise == 517) & (P13_J.G2station == i))] # 517.[1:42]
    P13_J = P13_J[~((P13_J.G2cruise == 598) & (P13_J.G2station == 2428))] # 598.2428
    P13_J = P13_J[~((P13_J.G2cruise == 598) & (P13_J.G2station == 2432))] # 598.2432
    for i in range(2585,2593):
        P13_J = P13_J[~((P13_J.G2cruise == 1060) & (P13_J.G2station == i))] # 1060.[2585:2592]
    for i in range(2980,3002):
        P13_J = P13_J[~((P13_J.G2cruise == 1066) & (P13_J.G2station == i))] # 1066.[2980:3001]
    for i in range(2733,2738):
        P13_J = P13_J[~((P13_J.G2cruise == 1064) & (P13_J.G2station == i))] # 1064.[2733:2737]
    for i in range(3294,3316):
        P13_J = P13_J[~((P13_J.G2cruise == 1069) & (P13_J.G2station == i))] # 1069.[3294:3315]
    for i in range(3341,3363):
        P13_J = P13_J[~((P13_J.G2cruise == 1069) & (P13_J.G2station == i))] # 1069.[3341:3362]
    P13_J = P13_J[~((P13_J.G2cruise == 1071) & (P13_J.G2station == 3529))] # 1071.3529
    for i in range(3538,3576):
        P13_J = P13_J[~((P13_J.G2cruise == 1071) & (P13_J.G2station == i))] # 1071.[3538:3575]
    for i in range(3757,3769):
        P13_J = P13_J[~((P13_J.G2cruise == 1076) & (P13_J.G2station == i))] # 1076.[3757:3768]
    for i in range(3790,3811):
        P13_J = P13_J[~((P13_J.G2cruise == 1076) & (P13_J.G2station == i))] # 1076.[3790:3810]
    for i in range(3885,3991):
        P13_J = P13_J[~((P13_J.G2cruise == 1078) & (P13_J.G2station == i))] # 1078.[3885:3990]
    for i in range(3903,3950):
        P13_J = P13_J[~((P13_J.G2cruise == 1079) & (P13_J.G2station == i))] # 1079.[3903:3949]
    for i in range(4424,4455):
        P13_J = P13_J[~((P13_J.G2cruise == 1092) & (P13_J.G2station == i))] # 1092.[4424:4454]
    for i in range(4115,4137):
        P13_J = P13_J[~((P13_J.G2cruise == 2038) & (P13_J.G2station == i))] # 2038.[4115:4136]
    for i in range(4186,4224):
        P13_J = P13_J[~((P13_J.G2cruise == 2041) & (P13_J.G2station == i))] # 2041.[4186:4223]
    for i in range(4480,4518):
        P13_J = P13_J[~((P13_J.G2cruise == 2047) & (P13_J.G2station == i))] # 2047.[4480:4517]
    P13_J = P13_J[~((P13_J.G2cruise == 2047) & (P13_J.G2station == 4526))] # 2047.4526
    for i in range(5299,5303):
        P13_J = P13_J[~((P13_J.G2cruise == 2064) & (P13_J.G2station == i))] # 2064.[5299:5302]
    for i in range(5466,5472):
        P13_J = P13_J[~((P13_J.G2cruise == 2084) & (P13_J.G2station == i))] # 2084.[5466:5471]
    for i in range(5453,5455):
        P13_J = P13_J[~((P13_J.G2cruise == 2084) & (P13_J.G2station == i))] # 2084.[5453:5454]
    for i in range(5785,5801):
        P13_J = P13_J[~((P13_J.G2cruise == 2091) & (P13_J.G2station == i))] # 2091.[5785:5800]
    for i in range(5994,6019):
        P13_J = P13_J[~((P13_J.G2cruise == 2094) & (P13_J.G2station == i))] # 2094.[5994:6018]
    for i in range(6030,6046):
        P13_J = P13_J[~((P13_J.G2cruise == 2096) & (P13_J.G2station == i))] # 2096.[6030:6045]
    P13_J = P13_J[~((P13_J.G2cruise == 2097) & (P13_J.G2station == 6081))] # 2097.6081
    for i in range(6246,6262):
        P13_J = P13_J[~((P13_J.G2cruise == 2102) & (P13_J.G2station == i))] # 2102.[6246:6261]
    for i in range(5596,5600):
        P13_J = P13_J[~((P13_J.G2cruise == 4063) & (P13_J.G2station == i))] # 4063.[5596:5599]
    for i in range(5785,5791):
        P13_J = P13_J[~((P13_J.G2cruise == 4069) & (P13_J.G2station == i))] # 4069.[5785:5790]
    for i in range(5934,5942):
        P13_J = P13_J[~((P13_J.G2cruise == 4074) & (P13_J.G2station == i))] # 4074.[5934:5941]
    for i in range(5997,6019):
        P13_J = P13_J[~((P13_J.G2cruise == 4076) & (P13_J.G2station == i))] # 4076.[5997:6018]
    for i in range(6462,6481):
        P13_J = P13_J[~((P13_J.G2cruise == 4081) & (P13_J.G2station == i))] # 4081.[6462:6480]
    for i in range(6702,6764):
        P13_J = P13_J[~((P13_J.G2cruise == 4087) & (P13_J.G2station == i))] # 4087.[6702:6763]

    P14 = espers[espers["G2cruise"].isin(go_ship_cruise_nums_2023['P14'])]
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
    
    trimmed = {'A02' : A02, 'A05' : A05, 'A10' : A10, 'A12' : A12, 
               'A135' : A135, 'A16N' : A16N, 'A16S' : A16S, 'A17' : A17, 
               'A20' : A20, 'A22' : A22, 'A25' : A25, 'A29' : A29, 
               'AR07E' : AR07E, 'AR07W' : AR07W, 'ARC01E' : ARC01E,
               'ARC01W' : ARC01W, 'MED01' : MED01, 'I01' : I01, 'I03' : I03,
               'I05' : I05, 'I06' : I06, 'I07' : I07, 'I08N' : I08N,
               'I08S' : I08S, 'I09N' : I09N, 'I09S' : I09S, 'I10' : I10,
               'P01' : P01, 'P02' : P02, 'P02_J' : P02_J, 'P03' : P03,
               'P03_J' : P03_J, 'P06' : P06, 'P09' : P09, 'P09_J' : P09_J,
               'P10' : P10, 'P10_J' : P10_J, 'P13' : P13, 'P13_J' : P13_J,
               'P14' : P14, 'P15' : P15, 'P16N' : P16N, 'P16S' : P16S,
               'P17E' : P17E, 'P17N' : P17N, 'P18' : P18, 'P21' : P21,
               'S04I' : S04I, 'SR04' : SR04, 'S04P' : S04P, 'SR01' : SR01,
               'SR03' : SR03}
    
    return trimmed

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

def create_mc_cruise_offset(all_trimmed, num_mc_runs):
    """
    Creates Monte Carlo simulation of 2 µmol/kg offset for each cruise
    
    Keyword arguments:
        all_trimmed = flattened dataframe of trimmed go-ship/clivar/woce
        cruises (not in dictionary form by transect)
        
        num_mc_runs = number of times to repeat Monte Carlo simulation
        
    Returns:
        G2talk_mc = array where rows are each data point, columns are each one
        MC simulation, and in each column cruises are offset by an amount
        selected randomly from a normal distribution with a standard deviation
        of 2 µmol/kg

    """
    cruise_nums = list(all_trimmed.G2cruise.unique())

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
        print(j)
    
    return G2talk_mc


def create_mc_individual_offset(all_trimmed, num_mc_runs):
    """
    Uses KL divergence to determine which equations predict best, where a lower
    KL divergence means that the two datasets are closer.
    
    Keyword arguments:
        all_trimmed = flattened dataframe of trimmed go-ship/clivar/woce
        cruises (not in dictionary form by transect)
        
        num_mc_runs = number of times to repeat Monte Carlo simulation
        
    Returns:
        G2talk_mc = array where rows are each data point, columns are each one
        MC simulation, and in each column cruises are offset by an amount
        selected randomly from a normal distribution with a standard deviation
        of 2 µmol/kg

    """

    G2talk_mc = np.empty((len(all_trimmed.G2talk),num_mc_runs))

    for j in range(0,num_mc_runs): # loop to repeat x times for MC analysis
        go_ship_offset = all_trimmed.copy()
        
        # loop through all data points
        for i in range(0,len(all_trimmed.G2talk)):
            # get random number between from normal distribution with mean at 0 and
            # standard deviation at 2, representing ± 2 µmol/kg alkalinity
            offset = np.random.normal(loc = 0.0, scale = 2)
            offset = np.round(offset, decimals=1)
            
            # add offset to all rows tagged with cruise number i
            G2talk_mc[i,j] = go_ship_offset.loc[i,'G2talk'] + offset
        print(j)
    
    return G2talk_mc
    
    
        