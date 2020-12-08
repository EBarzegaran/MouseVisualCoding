#!/usr/bin/env python
# coding: utf-8

import os
import MouseVisCode.Probe_functions as ProbeF
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

ResultPath = '/Users/elhamb/Documents/Data/AllenBrainObserver_new/Results' # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)


# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


# -indicate the animal IDs
#session_id = {767871931,768515987,771160300,771990200,774875821,778240327,779839471,781842082,821695405}
session_id = {732592105}

for S_id in session_id:
    print(S_id)
    # -Load Data for a session
    session = cache.get_session_data(S_id)
    
    # -set necessary paths
    if not os.path.exists(os.path.join(ResultPath, str(S_id))):
        os.mkdir(os.path.join(ResultPath, str(S_id)))

    Resultspath = os.path.join(ResultPath, str(S_id))
    
    for I in range(0, session.probes.index.values.shape[0]):
        probe_id = session.probes.index.values[I]

        # -load lfp data
        lfp = session.get_lfp(probe_id)

        # -first extract probe info and save
        #ProbeF.extract_probeinfo(session, lfp, probe_id, Resultspath,doRF=True)

        # -extract and prepare the data for a condition
        cond_name = 'drifting_gratings'#'drifting_gratings_75_repeats'    #'flashes'
        Prestim = 1             # prestimulus time in sec
        down_rate = 5           # down sampling -> the original sampling rate is 1250 Hz
        ProbeF.prepare_condition(session, lfp, probe_id, cond_name, Resultspath, Prestim, down_rate)
        
        # -plot the receptive field mapping for the probe
        # You can alternatively run this within the function ProbeF.extract_probeinfo by setting doRF to True
        #ProbeF.RF_mapping_plot(session, lfp, probe_id, True, Resultspath)
        
        # -CSD plot for the probe
        ProbeF.CSD_plots(session,lfp,probe_id,Resultspath)


