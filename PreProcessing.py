#!/usr/bin/env python
# coding: utf-8

import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from MouseVisCode import lfp_session
#from pydynet import dynet_statespace

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results' # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)

# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from functional connectivity stimulus set
session_id = [766640955, 767871931, 768515987, 771160300, 771990200,
              774875821, 778240327, 778998620, 779839471, 781842082, 821695405]

# parameters for epoching
cond_name = 'drifting_gratings_75_repeats'   # condition for iPDC analysis
down_sample_rate = 5                        # down-sampling rate-> orginal data is 1250
pre_stim = 1                                 # prestimulus time window, in seconds

# parameters for iPDC analsis
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include
Mord = 15                                   # Model order
ff = .98                                    # Filtering factor
pdc_method = 'iPDC'                         # Method for connectivity estimation
# a list of dictionaries, each element of list indicate the parameters to consider on conditions for FC analysis
stim_params = [{'contrast': .8}]

PDC={}  # To store a list of PDCs calculated used pdc_analysis function

for s_id in session_id:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache,session_id=s_id, result_path=ResultPath)

    # apply preprocessing: Update needed -> only load if preprocessing is done before

    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=1, do_RF=False, do_CSD=False)

    LFP.load_LFPprobes(LFP.preprocess[0])

    # Then for each animal indicate a preprocessed data and first down sample spatially and apply iPDC
    #layer selection -> Be careful: if you want to run this function, you should indicate the cortical layers manually
    LFP.layer_selection()

    PDC[s_id] = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params[0])

#iPDC analysis
# To be added.

