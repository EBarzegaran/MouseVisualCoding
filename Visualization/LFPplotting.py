#!/usr/bin/env python
# coding: utf-8

# add required paths
import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))#os.getcwd()
sys.path.insert(0, os.path.join(current_path,'MouseVisCode'))#

# import functions
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import lfp_session

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results'  # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)

# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from brain observatory set

session_id1 = [732592105, 737581020, 739448407, 742951821, 744228101,
               750749662, 754312389, 754829445, 757216464, 757970808,
               759883607, 761418226, 799864342]

# parameters for epoching
cond_name = 'drifting_gratings'  # condition for iPDC analysis
down_sample_rate = 5  # down-sampling rate-> orginal data is 1250
pre_stim = 1  # prestimulus time window, in seconds
preproc_dict = {
    'cond_name': cond_name,
    'srate': down_sample_rate,
    'prestim': pre_stim,
}
stim_params = [{'contrast': [.8], 'temporal_frequency':[2.0]}]
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include

lfp_Y = {} # To store average LFPs for later plottings

for s_id in session_id1:
    print('Animal #{}'.format(s_id))
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    # apply preprocessing: only load if preprocessing is done before
    lfp_Y[s_id] = LFP.plot_LFPs(preproc_params=preproc_dict, stim_params= stim_params[0], ROI_list = ROI_list,TimeWin=[-.2, 1])

# ----------------------------------------------------------------------------------------------------------------------

# indicate the animal IDs from functional connectivity stimulus set
session_id2 = [766640955, 767871931, 768515987, 771160300, 771990200,
               774875821, 778240327, 778998620, 779839471, 781842082, 821695405]

# parameters for epoching
cond_name = 'drifting_gratings_75_repeats'  # condition for iPDC analysis
preproc_dict = {
    'cond_name': cond_name,
    'srate': down_sample_rate,
    'prestim': pre_stim,
}

for s_id in session_id2:
    print('Animal #{}'.format(s_id))
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    # apply preprocessing: only load if preprocessing is done before
    lfp_Y[s_id] = LFP.plot_LFPs(preproc_params=preproc_dict, stim_params= stim_params[0], ROI_list = ROI_list,TimeWin=[-.3, 1])

# -------------------------------Plot the grand average over animals----------------------------------------------------

lfp_Y_avg = lfp_session.aggregate_LFP_ROI(lfp_Y)
lfp_session.LFP_plot(lfp_Y_avg['Y'], [-.2, 1], lfp_session.ROIColors('layers'), 'test.png')