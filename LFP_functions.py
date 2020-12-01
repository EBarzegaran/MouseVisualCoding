#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from scipy.ndimage.filters import gaussian_filter

def RF_mapping(results,presentations,TW,Units):
    # First find the time index of interest
    T = results['time'].mean(axis=1).mean(axis=1)*1000 # time in miliseconds
    Tind = np.where((T>TW[0]) & (T<TW[1]))[0]
    lfp_cond = (results['lfp']);
    lfp_cond = bipolar(gaussian_filter_trials(lfp_cond, 1));

    #lfp_cond = gaussian_filter_trials(csd(results['lfp'][:,Tind,:,:],1));
    lfp_cond = abs(lfp_cond[:,Tind,:,:].mean(axis=2)).mean(axis=1);
    cnd_id = results['conditioninfo'];

    PC = presentations[['orientation','x_position','y_position','stimulus_condition_id']]

    if len(Units)>1:
        LFP_averaged = lfp_cond[Units,:].max(axis=0);
    else:
        LFP_averaged = np.squeeze(lfp_cond[Units,:]);

    cnd_info = list(map(lambda x: PC[PC['stimulus_condition_id']==x].iloc[0],cnd_id))# each condition info
    cnd_info = pd.concat(cnd_info,axis=1).transpose()
    Data_organized = [];#zeros(3,9,9)
    Cnd_organized = [];
    for O in np.sort(cnd_info['orientation'].unique()):#range(0,len(cnd_info['orientation'].unique())):
        for X in np.sort(cnd_info['x_position'].unique()):#range(0,len(cnd_info['x_position'].unique())):
            for Y in np.sort(cnd_info['y_position'].unique()):#range(0,len(cnd_info['y_position'].unique())):
                Ind = cnd_info['stimulus_condition_id'][(cnd_info['orientation']==O) & (cnd_info['x_position']==X) & (cnd_info['y_position']==Y)].values
                #print(LFP_averaged.shape)
                Data_organized.append(LFP_averaged[Ind[0]-1])
                Cnd_organized.append(Ind[0])

    Data_final = np.array(Data_organized).reshape([cnd_info['orientation'].nunique(),cnd_info['x_position'].nunique(),cnd_info['y_position'].nunique()])
    return {'Data': Data_final,'CondInfo': Cnd_organized}

def organize_epoch(lfp, presentations,prestim = .5,poststim=.0):
    """
    This function get the lfp (numpy data array) and find the trials for a/ specific condition/conditions
    """


    # find optimum time window length for epoching the data
    PC = presentations
    CI = presentations['stimulus_condition_id']
    tw = []
    for i in range(0, len(CI)):
        # print(i)
        tw.append(len(np.where(np.logical_and(lfp['time'].values > (PC['start_time'].array[i] - prestim),
                                              lfp['time'].values < (PC['stop_time'].array[i] + poststim)))[0]))
    timelength =  max(tw)# I added 10 for trial variability

    # epoch the data
    Cond_id = []
    lfp_cond = np.zeros((min(lfp.shape), timelength, CI.value_counts().array[0], CI.nunique()))
    time = np.full((timelength, CI.value_counts().array[0], CI.nunique()), np.nan)
    for Cnd in range(0, CI.nunique()):
        if Cnd%10 ==0: print(Cnd)
        Cond_id.append(CI.unique()[Cnd])
        PC = presentations[CI == CI.unique()[Cnd]]
        for Ind in range(0, CI.value_counts().array[Cnd]):
            TW = np.logical_and(lfp['time'].values > (PC['start_time'].array[Ind] - prestim),
                                lfp['time'].values < (PC['stop_time'].array[Ind] + poststim))
            lfp_cond[:, 0:lfp[{"time": TW}].T.shape[1], Ind, Cnd] = lfp[
                {"time": TW}].T;  # sometimes the trial lengths are different
            time[0:lfp[{"time": TW}].T.shape[1], Ind, Cnd] = lfp["time"][TW].values - PC['start_time'].array[Ind]
    lfp_cond[0,:,:,:]=0;
    return {'lfp': lfp_cond, 'conditioninfo': Cond_id, 'time': time}

def bipolar(lfp, axis=0):
    # FIX the axis issue: by permuting your data
    sizes = np.array(lfp.shape)
    sizes[axis] += 2
    lfp1 = np.zeros(sizes)
    lfp1[0:sizes[axis] - 2, :, :, :] = lfp
    lfp2 = np.zeros(sizes)
    lfp2[2:sizes[axis], :, :, :] = lfp
    lfp_out = lfp2 - lfp1
    lfp_out = lfp_out[1:sizes[axis] - 1, :, :, :]
    return lfp_out

def csd(lfp, axis=0):
    # FIX the axis issue: by permuting your data
    sizes = np.array(lfp.shape);
    sizes[axis] += 2;
    lfp1 = np.zeros(sizes);
    lfp1[0:sizes[axis] - 2, :, :, :] = lfp;
    lfp2 = np.zeros(sizes);
    lfp2[2:sizes[axis], :, :, :] = lfp;
    lfp3 = np.zeros(sizes);
    lfp3[1:sizes[axis] - 1, :, :, :] = lfp;

    lfp_out = lfp2 + lfp1 - (2 * lfp3);
    # lfp_out -= lfp3;
    lfp_out = lfp_out[1:sizes[axis] - 1, :, :, :];
    lfp_out[0, :, :, :] = 0;
    return lfp_out;

def gaussian_filter_trials(lfp, S):
    sizes = np.array(lfp.shape);
    lfp_out = np.zeros(sizes);
    for I1 in range(0, sizes[2]):
        for I2 in range(0, sizes[3]):
            lfp_out[:, :, I1, I2] = gaussian_filter(lfp[:, :, I1, I2], sigma=S);
    return lfp_out

