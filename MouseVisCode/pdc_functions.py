import os
import json
import _pickle as cPickle
from functools import reduce
import numpy as np

def save_PDC(PDC_dict, path):
    """
    Save a PDC dictionary: an output of LFPsession.PDC_analysis
    :param PDC_dict:
    :param path:
    :return:
    """
    preproc_params = PDC_dict['preproc_dict']
    PDC_params = PDC_dict['PDCparam_dict']

    filename = search_PDC(PDC_dict['session_id'], path, PDC_params, preproc_params)
    # save the iPDCs and the parameters used as a dictionary
    filehandler = open(filename, "wb")
    cPickle.dump(PDC_dict, filehandler)
    filehandler.close()

    return filename


def search_PDC(session_id, result_path, PDCparams, preproc_dict):
    results = json.dumps(PDCparams['stim_param'])
    stim = ""
    for character in results:
        if character not in ['{', '}', ']', '[', ':', '"', ' ']:
            stim += character
    stim = stim.replace(',', '_')

    filename = os.path.join(result_path, '{}_{}_sr{}_prestim{}_Mord{}_ff{}_{}_{}.pkl'.format(
        session_id, preproc_dict['cond_name'], preproc_dict['srate'], preproc_dict['prestim'],
        PDCparams['Mord'], PDCparams['ff'], PDCparams['pdc_method'], stim))

    return filename


def PDC_to_ROI(PDC_dict):
    """
    Only keep the Intra area connectivity for further analysis
    :param PDC_dict: the dictionary that is the output of lfp_session.PDC_analysis
    :return: another dictionary with intra area PDCs
    """
    PDC_ROI = {}
    for roi in PDC_dict['ROIs']:
        # first extract ROI_L1 to ROI_L6
        Ind_src = [x.find('{}_L'.format(roi)) == 0 for x in PDC_dict['PDC'].source.values]
        Ind_trg = [x.find('{}_L'.format(roi)) == 0 for x in PDC_dict['PDC'].target.values]
        PDC_ROI [roi] = PDC_dict['PDC'][Ind_trg, Ind_src, :, :]

    PDC_out = {'session_id': PDC_dict['session_id'], 'ROIs':PDC_dict['ROIs'], 'PDC_ROI': PDC_ROI}
    # later think about probe info ############################----NOTE----##############################
    return PDC_out


def aggregate_PDC_ROI(PDC_ROI_list):
    """
    average Intra area PDCs over common ROIs

    :param PDC_ROI_list: a list of PDC_ROIs, outputs of PDC_to_ROI function
    :return: an average dictionary? of averaged ROIs
    """

    ROIs_All = reduce(lambda x, y: list(set().union(x, y)), [x['ROIs'] for x in PDC_ROI_list])
    PDC_ROI_all = {'session_ids': [x['session_id'] for x in PDC_ROI_list],
                   'ROIs': ROIs_All,
                   'PDCs': {}}

    # first indicate the ROIs in the list
    for roi in ROIs_All:
        s_ids = np.where(np.array([x['ROIs'].count(roi)>0 for x in PDC_ROI_list]))[0]
        # -for animals with that ROI: make a list and concat them-
        PDC_temp = [PDC_ROI_list[x]['PDC_ROI'][roi] for x in s_ids]
        # -time indexes with non NaN values and round them 3 digit to be uniform-
        NNan_ind = [np.logical_not(np.isnan(x.time.values)) for x in PDC_temp]
        NNan_ind = reduce(lambda x, y: np.logical_and(x[:min(len(x), len(y))], y[:min(len(x), len(y))]), NNan_ind)
        PDC_temp2 = []

        for pdc in PDC_temp: # loop over animals
            pdc.time.values =  np.round(pdc.time.values,3)
            PDC_temp2.append(pdc.isel(time = np.where(NNan_ind)[0]))

        # -calculate average over animals-
        if len(PDC_temp2)>1 :  # if more than 1 animal, calculate average
            PDC_avg = reduce(lambda x, y: x+y, PDC_temp2)/len(PDC_temp)
            PDC_ROI_all['PDCs'][roi]= PDC_avg
        else :
            PDC_ROI_all['PDCs'][roi] = PDC_temp2[0]

    return PDC_ROI_all