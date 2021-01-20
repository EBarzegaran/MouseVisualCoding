#!/usr/bin/env python
# coding: utf-8

import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from MouseVisCode import lfp_session
import time
from lfp_session import search_preproc

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results'  # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)

# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from brain observatory set
"""
session_id1 = [732592105, 737581020, 739448407, 742951821, 743475441, 744228101,
               750332458, 750749662, 754312389, 754829445, 757216464, 757970808,
               759883607, 761418226, 763673393, 799864342]
"""

session_id1 = [737581020, 739448407, 742951821, 744228101,
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


for s_id in session_id1:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    # apply preprocessing: only load if preprocessing is done before
    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False, do_CSD=True, do_probe=False)

    LFP.load_LFPprobes(preproc_dict)
    LFP.layer_selection()

# ----------------------------------------------------------------------------------------------------------------------

# indicate the animal IDs from functional connectivity stimulus set
session_id2 = [766640955, 767871931, 768515987, 771160300, 771990200,
               774875821, 778240327, 778998620, 779839471, 781842082, 821695405]

# parameters for epoching
cond_name = 'drifting_gratings_75_repeats'  # condition for iPDC analysis
down_sample_rate = 5
pre_stim = 1  # prestimulus time window, in seconds

# parameters for iPDC analsis
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include
Mord = 15  # Model order
ff = .98  # Filtering factor
pdc_method = 'iPDC'  # Method for connectivity estimation
# a list of dictionaries, each element of list indicate the parameters to consider on conditions for FC analysis,
# put the values of dict as list
stim_params = [{'contrast': [.8]}]

PDC = {}  # To store a list of PDCs calculated used pdc_analysis function

for s_id in session_id2:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    # apply preprocessing: only load if preprocessing is done before
    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False, do_CSD=True)

    # load the preprocessed data
    # LFP.load_LFPprobes(LFP.preprocess[0])

    # Then for each animal indicate a preprocessed data and first down sample spatially and apply iPDC
    # layer selection -> Be careful: if you want to run this function, you should indicate the cortical layers manually
    # LFP.layer_selection()

    # PDC[s_id] = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params[0])

# iPDC analysis
# To be added.
