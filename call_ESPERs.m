%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: call_ESPERs.m
% Author: Reese Barrett
% Date: 2023-10-2y

% Description:
% - Reads  GLODAPv2.2022 .csv files modified for use in ESPERs
% - Calls each ESPER and runs each available equation of the 16 per ESPER
% to predict TA
% - Outputs data into format as .csv
%    
% To-Do:
% - read in .csv file
% - run through ESPERs
% - set up output to be useful (want all original properties + predicted TA
% at each lat/lon/depth/time + associated uncertainties
% test
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% read in modified GLODAP data from .csv generated by
% glodap_data_formatting.py
filepath = '/Users/Reese/Documents/project1/data/';
input_file = 'GLODAPv2.2022_for_ESPERs.csv';
output_file = 'GLODAP_with_ESPER_TA.csv';

glodap = readtable([filepath input_file]);

%% call ESPER_LIR

% call ESPER_LIR
% calls all 16 equations representing all possible combinations of input
% variables used to calculate TA (Salinity, Temperature, Nitrate, Silicate,
% and Oxygen)
[output_estimates_LIR, uncertainty_estimates_LIR] = ESPER_LIR(1, ...
    [glodap.G2longitude, glodap.G2latitude, glodap.G2depth], ...
    [glodap.G2salinity, glodap.G2temperature, glodap.G2nitrate, ...
    glodap.G2silicate, glodap.G2oxygen], [1 2 4 5 6], ...
    'Equations', [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16], ...
    'EstDates',[glodap.G2dectime]);


%% format ESPER_LIR output

% create output table
output_LIR = table;

% store outputs in table with each column representing the TA estimate
% and uncertainty created by the corresponding equation number
for i = 1:16
    TA_col_name = 'LIRtalk' + string(i);
    TA_uncert_col_name = 'LIRtalk_uncert' + string(i);
    output_LIR.(TA_col_name) = output_estimates_LIR.TA(:,i);
    output_LIR.(TA_uncert_col_name) = uncertainty_estimates_LIR.TA(:,i);
end

% format output from ESPER LIR

%% call ESPER_NN

[output_estimates_NN, uncertainty_estimates_NN] = ESPER_NN(1, ...
    [glodap.G2longitude, glodap.G2latitude, glodap.G2depth], ...
    [glodap.G2salinity, glodap.G2temperature, glodap.G2nitrate, ...
    glodap.G2silicate, glodap.G2oxygen], [1 2 4 5 6], ...
    'Equations', [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16], ...
    'EstDates',[glodap.G2dectime]);

%% format ESPER_NN output

% create output table
output_NN = table;

% store outputs in table with each column representing the TA estimate
% and uncertainty created by the corresponding equation number
for i = 1:16
    TA_col_name = 'NNtalk' + string(i);
    TA_uncert_col_name = 'NNtalk_uncert' + string(i);
    output_NN.(TA_col_name) = output_estimates_NN.TA(:,i);
    output_NN.(TA_uncert_col_name) = uncertainty_estimates_NN.TA(:,i);
end

% format output from ESPER NN

%% call ESPER_Mixed

[output_estimates_Mixed, uncertainty_estimates_Mixed] = ESPER_Mixed(1, ...
    [glodap.G2longitude, glodap.G2latitude, glodap.G2depth], ...
    [glodap.G2salinity, glodap.G2temperature, glodap.G2nitrate, ...
    glodap.G2silicate, glodap.G2oxygen], [1 2 4 5 6], ...
    'Equations', [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16], ...
    'EstDates',[glodap.G2dectime]);

%% format ESPER_Mixed output

% create output table
output_Mixed = table;

% store outputs in table with each column representing the TA estimate
% and uncertainty created by the corresponding equation number
for i = 1:16
    TA_col_name = 'Mtalk' + string(i);
    TA_uncert_col_name = 'Mtalk_uncert' + string(i);
    output_Mixed.(TA_col_name) = output_estimates_Mixed.TA(:,i);
    output_Mixed.(TA_uncert_col_name) = uncertainty_estimates_Mixed.TA(:,i);
end

%% concatenate tables together and output as a .csv file for analysis in Python
output = [glodap output_LIR output_NN output_Mixed];
writetable(output,[filepath output_file])





