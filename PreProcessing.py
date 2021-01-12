#!/usr/bin/env python
# coding: utf-8

import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from MouseVisCode import lfp_session

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results' # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)


# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


# indicate the animal IDs from functional connectivity stimulus set
session_id = {766640955,767871931,768515987,771160300,771990200,774875821,778240327,778998620,779839471,781842082,821695405}
#session_id = {781842082}

for s_id in session_id:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache,session_id=s_id, result_path=ResultPath)

    # apply preprocessing: Update needed -> only load if preprocessing is done before
    #LFP.RF = False
    #LFP.preprocessing(cond_name=None, down_sample_rate=5, Prestim=1, do_RF=False, do_CSD=True) for the brain observatory animals

    LFP.preprocessing(cond_name='drifting_gratings_75_repeats', down_sample_rate=5, Prestim=1, do_RF=False, do_CSD=False)

    LFP.load_LFPprobes(LFP.preprocess[0])

    # Then for each animal indicate a preprocessed data and first down sample spatially and apply iPDC
    #layer selection -> Be careful: if you want to run this function, you should indicate the cortical layers manually
    LFP.layer_selection()

#iPDC analysis
# To be added.