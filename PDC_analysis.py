
# add required folders
import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))#os.getcwd()
sys.path.insert(0, os.path.join(current_path,'MouseVisCode'))#
sys.path.insert(0,os.path.join(current_path,'External','pydynet'))

# import required functions
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import lfp_session
import pdc_functions as PDCF
import _pickle as cPickle

# set necessary paths
result_path = '/Volumes/Elham-Unifr/Data/AllenBrainAll/Results'  # where to save the results
if not os.path.exists(result_path):
    os.mkdir(result_path)

# this path determines where the downloaded data is stored
manifest_path = os.path.join("/Volumes/Elham-Unifr/Data/AllenBrainAll/ecephys_project_cache", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# indicate the animal IDs from brain observatory set: animal # 739448407 does not have good quality
session_id1 = [732592105,737581020, 739448407, 742951821, 744228101,
               750749662, 754312389, 754829445, 757216464, 757970808,
               759883607, 761418226, 799864342]

# parameters for preprocessing
cond_name = 'drifting_gratings'  # condition for iPDC analysis
down_sample_rate = 5  # down-sampling rate-> orginal data is 1250
pre_stim = 1  # prestimulus time window, in seconds
preproc_dict_BO = { # which preprocessing to use for PDC analysis
    'cond_name': cond_name,
    'srate': down_sample_rate,
    'prestim': pre_stim
}

# parameters for iPDC analsis
ROI_list = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']  # ROIs to include
Mord = 15  # Model order
ff = .98  # Filtering factor
pdc_method = 'iPDC'  # Method for connectivity estimation
# a list of dictionaries, each element of list indicate the parameters to consider on conditions for FC analysis,
# Note: put the values of dict as list of params
stim_params_BO = [{'contrast': [.8], 'temporal_frequency':[2.0]}]

PDC_ROI = {}  # To store a list of PDCs calculated used pdc_analysis function

for s_id in session_id1:
    print('Session_id:{}'.format(s_id))
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=result_path)

    # apply preprocessing: only if preprocessing is not done before
    LFP.preprocessing(cond_name=cond_name, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False,
                      do_CSD=True, do_probe=False)

    # Apply PDC analysis on the loaded LFP between all ROIs and layers for each animal and save the results
    PDC = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params_BO[0], preproc_params = preproc_dict_BO, redo=True)
    PDC_ROI[s_id] = PDCF.PDC_to_ROI(PDC)

PDC_all_BO = PDCF.aggregate_PDC_ROI(list(PDC_ROI.values()))
PDCparam_dict = PDC['PDCparam_dict']

# --------------------------same analysis on functional connectivity set-------------------------------------

# indicate the animal IDs from functional connectivity stimulus set
session_id2 = [766640955, 767871931, 768515987, 771160300, 771990200,
               774875821, 778240327, 778998620, 779839471, 781842082, 821695405]
cond_name2 = 'drifting_gratings_75_repeats'  # condition for iPDC analysis
preproc_dict_FC = { # which preprocessing to use for PDC analysis
    'cond_name': cond_name2,
    'srate': down_sample_rate,
    'prestim': pre_stim
}

for s_id in session_id2:
    print('Session_id:{}'.format(s_id))
    # -Load Data for a session
    LFP = lfp_session.LFPSession(cache=cache, session_id=s_id, result_path=result_path)

    # apply preprocessing: only if preprocessing is not done before
    LFP.preprocessing(cond_name=cond_name2, down_sample_rate=down_sample_rate, pre_stim=pre_stim, do_RF=False,
                      do_CSD=True, do_probe=False)

    # Apply PDC analysis on the loaded LFP between all ROIs and layers for each animal and save the results
    PDC = LFP.pdc_analysis(ROI_list=ROI_list, Mord=Mord, ff=ff, pdc_method=pdc_method, stim_params=stim_params_BO[0], preproc_params = preproc_dict_FC)
    PDC_ROI[s_id] = PDCF.PDC_to_ROI(PDC)

PDC_all_FC = PDCF.aggregate_PDC_ROI([PDC_ROI[x] for x in session_id2])

PDC_all = PDCF.aggregate_PDC_ROI(list(PDC_ROI.values()))

# --------------------------------------Save the average and full data--------------------------------------------------
# save the data in a average folder
AverageResult = os.path.join(result_path,'AverageResults')
if not os.path.isdir(AverageResult):
    os.mkdir(AverageResult)

# full is the PDC_ROI & PDCparam_dict_FC & PDCparam_dict and preproc_dict_BO & preproc_dict_FC
file_name = PDCF.search_PDC("FullData_ROI", AverageResult, PDCparam_dict, preproc_dict_BO) # full data
filehandler = open(file_name, "wb")
cPickle.dump({'PDCs': PDC_ROI,'PDCparam_dict': PDCparam_dict,
              'preproc_dict_BO': preproc_dict_BO, 'preproc_dict_FC' : preproc_dict_FC}, filehandler)
filehandler.close()

file_name = PDCF.search_PDC("AverageData_ROI", AverageResult, PDCparam_dict, preproc_dict_BO) # Averaged data
filehandler = open(file_name, "wb")
cPickle.dump({'PDC_Average_BO': PDC_all_BO, 'PDC_Average_FC': PDC_all_FC,'PDC_Average_all': PDC_all,
              'preproc_dict_BO': preproc_dict_BO, 'preproc_dict_FC' : preproc_dict_FC}, filehandler)
filehandler.close()