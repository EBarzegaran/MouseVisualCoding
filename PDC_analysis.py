import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from MouseVisCode import lfp_session
import time

ResultPath = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results'  # where to save the results
# set necessary paths
if not os.path.exists(ResultPath):
    os.mkdir(ResultPath)

# this path determines where the downloaded data will be stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from brain observatory set: animal # 739448407 does not have good quality
session_id1 = [732592105, 737581020, 739448407, 742951821, 744228101,
               750749662, 754312389, 754829445, 757216464, 757970808,
               759883607, 761418226, 799864342]

# ---------------------------------------------------------------------------
# parameters for condition
cond_name = 'drifting_gratings'  # condition for iPDC analysis
down_sample_rate = 5  # down-sampling rate-> orginal data is 1250
pre_stim = 1  # prestimulus time window, in seconds
preproc_dict = { # which preprocessing to use for PDC analysis
    'cond_name': cond_name,
    'srate': down_sample_rate,
    'prestim': pre_stim
}
# ---------------------------------------------------------------------------
# parameters for iPDC analsis
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include
Mord = 15  # Model order
ff = .98  # Filtering factor
pdc_method = 'iPDC'  # Method for connectivity estimation
# a list of dictionaries, each element of list indicate the parameters to consider on conditions for FC analysis,
# Note: put the values of dict as list of params
stim_params_bo = [{'contrast': [.8], 'temporal_frequency':[2.0]}]


PDC = {}  # To store a list of PDCs calculated used pdc_analysis function

for s_id in session_id1:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    t = time.time()
    # apply preprocessing: only if preprocessing is not done before
    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False,
                      do_CSD=True, do_probe=False)

    # Apply PDC analysis on the loaded LFP between all ROIs and layers for each animal and save the results
    PDC[s_id] = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params_bo[0], preproc_params = preproc_dict)
    print(time.time()-t)

#----------------------------------------------------------------------------------
# indicate the animal IDs from functional connectivity stimulus set
session_id2 = [766640955, 767871931, 768515987, 771160300, 771990200,
               774875821, 778240327, 778998620, 779839471, 781842082, 821695405]

cond_name2 = 'drifting_gratings_75_repeats'  # condition for iPDC analysis
preproc_dict = { # which preprocessing to use for PDC analysis
    'cond_name': cond_name2,
    'srate': down_sample_rate,
    'prestim': pre_stim
}

for s_id in session_id2:
    print(s_id)
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=ResultPath)

    # apply preprocessing: only if preprocessing is not done before
    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False,
                      do_CSD=True, do_probe=False)

    # Apply PDC analysis on the loaded LFP between all ROIs and layers for each animal and save the results
    PDC[s_id] = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params_bo[0], preproc_params = preproc_dict)
