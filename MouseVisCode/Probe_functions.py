import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import LFP_functions as LFPF
import scipy.signal as signal
import scipy.io as sio
import pickle


from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from scipy.ndimage.filters import gaussian_filter


def extract_probeinfo(session, lfp, probe_id, Resultspath, doRF):
    structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
    interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]

    channel1 = session.channels[session.channels['probe_id'] == probe_id]
    channel1 = channel1.reset_index()
    A = channel1[channel1['id'].isin(lfp["channel"].values)]
    A = A[['id', 'anterior_posterior_ccf_coordinate', 'left_right_ccf_coordinate', 'dorsal_ventral_ccf_coordinate']]
    A.rename(columns={'anterior_posterior_ccf_coordinate': 'AP_CCF', 'left_right_ccf_coordinate': 'ML_CCF',
                      'dorsal_ventral_ccf_coordinate': 'DV_CCF'}, inplace=True)

    # create folder if not exist
    if not os.path.exists(os.path.join(Resultspath, 'MatlabData')):
        os.path.join(Resultspath, 'MatlabData')
    # do RF mapping
    if doRF:
        rf_results = RF_mapping_plot(session, lfp, probe_id, True, Resultspath)
    else:
        rf_results = []

    # Save the result
    if not os.path.isdir(os.path.join(Resultspath, 'PrepData')):
        os.mkdir(os.path.join(Resultspath, 'PrepData'))

    sio.savemat(os.path.join(Resultspath, 'PrepData', '{}_ProbeInfo.mat'.format(probe_id)),
                {'Coords': A.to_dict('list'), 'structure_acronyms': structure_acronyms, 'intervals': intervals,
                 'RF_Results': rf_results})

    a_file = open(os.path.join(Resultspath,'PrepData','{}_ProbeInfo.pkl'.format(probe_id)), "wb")
    pickle.dump({'Coords':A.to_dict('list'),'structure_acronyms':structure_acronyms,'intervals':intervals},a_file)
    a_file.close()

def prepare_condition(session, lfp, probe_id, cond_name, Resultspath, Prestim, down_rate):
    # sampling rate
    SRate = round(session.probes['lfp_sampling_rate'][session.probes.index.values == probe_id].values[0])

    # Extract ROIs
    structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
    interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]

    # --------------Read Grating and prepare the figures and then data for matlab-------------

    presentations = session.get_stimulus_table(cond_name)
    CI = presentations['stimulus_condition_id']

    results = LFPF.organize_epoch(lfp, presentations, prestim=Prestim, poststim=.0)
    # structure_acronyms[structure_acronyms.shape[0] - 2]

    Times = results['time'].mean(axis=1)[:, 7]
    Times[Times.shape[0] - 1] = Times[Times.shape[0] - 2] * 2 - Times[Times.shape[0] - 3]
    lfp_cond = (results['lfp'])
    lfp_cond = LFPF.bipolar(lfp_cond)  # LFPF.gaussian_filter_trials(lfp_cond, 1));

    # --------------------------MATLAB------------------------------
    Times = results['time'].mean(axis=1)[:, 7]
    Times[Times.shape[0] - 1] = Times[Times.shape[0] - 2] * 2 - Times[Times.shape[0] - 3]

    lfp_cond = (results['lfp'])
    #
    if lfp_cond == 'flashes':
        lfp_cond = LFPF.csd(LFPF.gaussian_filter_trials(lfp_cond, 1))  # just in case of flashes for layer assignment
    else:
        lfp_cond = LFPF.bipolar(lfp_cond)

    Y = lfp_cond
    # Y = Y[intervals[intervals.shape[0]-3]:intervals[intervals.shape[0]-2]+1];
    Y = np.moveaxis(Y, -2, 0)
    # Y = Y[:,np.arange(0,Y.shape[1],2),:]
    # Y = Y[:,np.arange(Y.shape[1],0,-1),:] % reverse the electrode order! I do this in matlab

    # downsampling the signal
    # Here I downsample the signal, to allow better estimation of FC in lower frequencies using AR models:
    Y = signal.decimate(Y, down_rate, axis=2)
    Times = Times[np.arange(0, Times.shape[0], down_rate)]
    dSRate = SRate / down_rate

    # condition info
    cnd_id = results['conditioninfo']
    cnd_info = list(map(lambda x: presentations[presentations['stimulus_condition_id'] == x].iloc[0],
                        cnd_id))  # each condition info
    cnd_info = pd.concat(cnd_info, axis=1).transpose()

    # save data
    if not os.path.exists(os.path.join(Resultspath, 'PrepData')):
        os.mkdir(os.path.join(Resultspath, 'PrepData'))

    sio.savemat(os.path.join(Resultspath, 'PrepData', '{}_{}{}.mat'.format(probe_id, cond_name, int(dSRate))),
                {'Y': Y, 'Times': Times, 'srate': dSRate, 'cnd_info': cnd_info.to_dict("list"), 'cnd_id': cnd_id
                    , 'ROI': structure_acronyms[structure_acronyms.shape[0] - 2]})

    a_file = open(os.path.join(Resultspath, 'PrepData', '{}_{}{}.pkl'.format(probe_id, cond_name, int(dSRate))), "wb")
    pickle.dump({'Y': Y, 'Times': Times, 'srate': dSRate, 'cnd_info': cnd_info.to_dict("list"), 'cnd_id': cnd_id
                    , 'ROI': structure_acronyms[structure_acronyms.shape[0] - 2]}, a_file)
    a_file.close()


def CSD_plots(session, lfp, probe_id, Resultspath):
    # --------------------------------------------------------
    # In case you want to check the original csd, run this:
    csd = session.get_current_source_density(probe_id)
    filtered_csd = gaussian_filter(csd.data, sigma=4)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.set_cmap('coolwarm')
    ax.pcolor(csd["time"], csd["vertical_position"], filtered_csd)

    ax.set_xlabel("time relative to stimulus onset (s)", fontsize=20)
    ax.set_ylabel("vertical position (um)", fontsize=20)

    # plt.show()
    structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
    interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]
    plt.title(structure_acronyms[structure_acronyms.shape[0] - 2])
    fig.savefig('{}/Probe_{}_CSD_original.png'.format(Resultspath, probe_id), dpi=300)

    # ------------------THIS IS FOR FLASHES CONDITION-----------
    # -compute CSD of the flashes condition
    presentations = session.get_stimulus_table('flashes')
    CI = presentations['stimulus_condition_id']

    Prestim = .05  # prestimulus time in sec
    results = LFPF.organize_epoch(lfp, presentations, Prestim, 0)

    # -Prepare Variables-
    CI = np.array(results['conditioninfo'])
    CI = pd.DataFrame(CI, columns=['CID'])['CID']
    lfp_cond = (results['lfp'])
    lfp_cond = lfp_cond[55:80]  # intervals[intervals.shape[0]-3]:intervals[intervals.shape[0]-2]+1];
    # lfp_cond = LFPF.bipolar(LFPF.gaussian_filter_trials(lfp_cond, 1));
    lfp_cond = LFPF.csd(LFPF.gaussian_filter_trials(lfp_cond, 1))
    Time = results['time'].mean(axis=1)

    intervals2 = intervals
    intervals2[intervals2 < 0] = 0
    # -Figure configs-
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    plt.set_cmap("coolwarm")

    # ---
    Ma = []
    for Cnd in np.array(range(CI.nunique() - 1, -1, -1)):  # range(0,CI.nunique()):
        TimeWin = Time[:, Cnd]
        ZeroPoint = abs(TimeWin[~np.isnan(TimeWin)] - .0).argmin()
        ax = axs[Cnd]
        lfp_cond_M = np.nanmean(lfp_cond[:, :, :, Cnd], axis=2)
        # -Plot-
        p = ax.pcolormesh(lfp_cond_M)
        # -Ytick-
        ax.set_yticks(np.arange(0, lfp_cond.shape[0]))
        ax.set_yticklabels(np.arange(lfp_cond.shape[0], 0, -1))
        # -ToSetXtickLabels-
        num_time_labels = 12
        time_label_indices = np.around(np.linspace(1, len(TimeWin), num_time_labels)).astype(int) - 1
        time_labels = np.round(TimeWin[time_label_indices], 2)
        ax.set_xticks(time_label_indices)
        ax.set_xticklabels(time_labels)
        ax.set_xlabel("time (s)", fontsize=20)
        ax.set_xlim([0, min(np.argwhere(np.isnan(TimeWin)))])
        ax.plot([ZeroPoint, ZeroPoint], [0, lfp_cond.shape[0]], 'w--')
        # -TitleAndColormap-
        ax.set_title('Condition #{}'.format(CI.unique()[Cnd]))
        # fig.set_clim(-1,1)
        cbar = fig.colorbar(p, ax=ax)  # format=ticker.FuncFormatter(fmt))
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        Ma.append(abs(lfp_cond_M).max() / 1.5)
        p.set_clim(-max(Ma), max(Ma))

    plt.set_cmap("coolwarm")
    plt.tight_layout()
    fig.savefig('{}/Probe_{}_flashes_csd.png'.format(Resultspath, probe_id), dpi=300)


def RF_mapping_plot(session, lfp, probe_id, doplot, Resultspath):
    # THIS IS RF MAPPING USING GABORS

    presentations = session.get_stimulus_table('gabors')
    CI = presentations['stimulus_condition_id']
    Prestim = .0  # prestimulus time in sec, Notes: there is no ISI for gabors condition
    results = LFPF.organize_epoch(lfp, presentations, Prestim, 0)

    cnd_id = results['conditioninfo']
    cnd_info = list(map(lambda x: presentations[presentations['stimulus_condition_id'] == x].iloc[0],
                        cnd_id))  # each condition info
    cnd_info = pd.concat(cnd_info, axis=1).transpose()

    Data_final = np.zeros([min(lfp.shape), 3, 9, 9])


    for E in range(0, min(lfp.shape)):
        if E % 5 == 0: print(E)
        Results2 = LFPF.RF_mapping(results, presentations, [30, 110], np.array([E]))
        Data_final[E, :, :, :] = Results2['Data']

    # prepare
    # cnd_id = Results2['CondInfo']
    if doplot:
        fig, axes = plt.subplots(round(Data_final.shape[0] / 2), 3, figsize=(4, round(Data_final.shape[0] / 2) * 2))
        plt.set_cmap('Reds')
        Elecs = np.array(range(0, Data_final.shape[0], 2))
        for j in range(0, round(Data_final.shape[0] / 2)):
            M = Data_final[Elecs[j], :, :, :].max()
            for i in range(0, 3):
                p = axes[j, i].pcolormesh(gaussian_filter(Data_final[Elecs[j], i, :, :], sigma=.5))
                p.set_clim(0, M / 1.2)
                axes[j, i].set_aspect('equal')
                if i == 0:
                    axes[j, i].set_ylabel('E{}'.format(Elecs[j]))
                if j == 0:
                    axes[j, i].set_title('Orient = {}'.format(np.sort(presentations['orientation'].unique())[i]))
        plt.tight_layout
        fig.savefig('{}/Probe_{}_RF_mapping_all.png'.format(Resultspath, probe_id), dpi=300)
        plt.close()
        return {'Data': Data_final, 'CondInfo': cnd_info.to_dict('list'), 'CondOrganized': Results2['CondInfo']}
    else:
        return {'Data': Data_final, 'CondInfo': cnd_info.to_dict('list'), 'CondOrganized': Results2['CondInfo']}
