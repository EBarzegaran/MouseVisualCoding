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
import probe_functions as ProbeF
import matplotlib.pyplot as plt

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results'  # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)

# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from brain observatory set
session_id1 = [732592105, 737581020, 739448407,
               742951821, 744228101,
               750749662, 754312389, 754829445, 757216464, 757970808,
               759883607, 761418226, 799864342]

# parameters for epoching
cond_name = 'drifting_gratings'  # condition for iPDC analysis
down_sample_rate = 1  # down-sampling rate-> orginal data is 1250
pre_stim = 0  # prestimulus time window, in seconds
preproc_dict = {
    'cond_name': cond_name,
    'srate': down_sample_rate,
    'prestim': pre_stim,
}
stim_params = [{'contrast': [.8], 'temporal_frequency':[2.0]}]
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include


for s_id in session_id1:
    print('Animal #{}'.format(s_id))
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    nroi = len(LFP.probes.keys())
    fig, axs = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi))
    fig2, axs2 = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi))
    # do individually for each probe, because it is a large data
    i = 0
    for probe_id in LFP.probes.keys():
        lfp = LFP.session.get_lfp(probe_id)
        Results = ProbeF.prepare_condition(LFP.session, LFP.session_id, lfp, probe_id, cond_name, LFP.result_path,
                                           pre_stim, down_sample_rate, False)

        result = ProbeF.layer_reduction(Results.Y, Results.srate, probe_id, LFP.result_path)
        axs[i].plot(result['labels'], result['high_gamma_amp'])
        axs[i].axvline(x=result['labels'][result['high_gamma_amp'].argmax()], linewidth=1, linestyle='--', color='k')
        axs[i].set_ylabel(Results.ROI)
        axs[i].set_xticks(result['labels'][range(0, len(result['labels']), 2)])

        axs2[i].pcolor(result['Pwelch_Freq'], result['labels'], result['Pwelch_norm'])
        # axs2[i].axhline(x=result['labels'][result['low_gamma_amp'].argmax()], linewidth=1, linestyle='--', color='k')
        axs2[i].set_ylabel(Results.ROI)
        axs2[i].invert_yaxis()
        axs2[i].set_yticks(result['labels'][range(0, len(result['labels']), 2)])
        axs2[i].tick_params(axis='both', which='major', labelsize=8)
        i += 1
    fig.savefig(os.path.join(LFP.result_path, 'HighGamma_layers.eps'), bbox_inches='tight')
    plt.close(fig)

    axs2[i - 1].set_xlabel('Frequency(Hz)')
    fig2.savefig(os.path.join(LFP.result_path, 'Pwelch_layers.png'), bbox_inches='tight')
    plt.close(fig2)

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
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    nroi = len(LFP.probes.keys())
    fig, axs = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi))
    fig2, axs2 = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi))
    # do individually for each probe, because it is a large data
    i=0
    for probe_id in LFP.probes.keys():
        lfp = LFP.session.get_lfp(probe_id)
        Results = ProbeF.prepare_condition(LFP.session, LFP.session_id, lfp, probe_id, cond_name, LFP.result_path,
                                           pre_stim, down_sample_rate, False)

        result = ProbeF.layer_reduction(Results.Y, Results.srate, probe_id, LFP.result_path)
        axs[i].plot(result['labels'], result['high_gamma_amp'])
        axs[i].axvline(x=result['labels'][result['high_gamma_amp'].argmax()], linewidth=1, linestyle='--', color='k')
        axs[i].set_ylabel(Results.ROI)
        axs[i].set_xticks(result['labels'][range(0,len(result['labels']),2)])

        axs2[i].pcolor(result['Pwelch_Freq'],result['labels'], result['Pwelch_norm'])
        #axs2[i].axhline(x=result['labels'][result['low_gamma_amp'].argmax()], linewidth=1, linestyle='--', color='k')
        axs2[i].set_ylabel(Results.ROI)
        axs2[i].invert_yaxis()
        axs2[i].set_yticks(result['labels'][range(0, len(result['labels']), 2)])
        axs2[i].tick_params(axis='both', which='major', labelsize=8)
        i += 1
    fig.savefig(os.path.join(LFP.result_path,'HighGamma_layers.eps'), bbox_inches='tight')
    plt.close(fig)

    axs2[i-1].set_xlabel('Frequency(Hz)')
    fig2.savefig(os.path.join(LFP.result_path, 'Pwelch_layers.png'), bbox_inches='tight')
    plt.close(fig2)