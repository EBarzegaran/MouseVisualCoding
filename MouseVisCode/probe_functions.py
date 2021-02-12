import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lfp_functions as LFPF
import scipy.signal as signal
import _pickle as cPickle
import xarray as xr
from scipy.ndimage.filters import gaussian_filter
from scipy import signal

from sklearn.decomposition import FastICA, PCA


def extract_probeinfo(session, lfp, probe_id, Resultspath, doRF):
    """
    extract the channels info of a probe including their ROI, their coordinates, their RF maps
    :param session: session class from allensdk
    :param lfp: lfp matrix from session.get_lfp
    :param probe_id:
    :param Resultspath: path to save the results
    :param doRF: should the function run RF mapping?
    :return: save the results in a binary format file
    """
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

    a_file = open(os.path.join(Resultspath, 'PrepData', '{}_ProbeInfo.pkl'.format(probe_id)), "wb")
    # cPickle.dump({'Coords':A.to_dict('list'),'structure_acronyms':structure_acronyms,'intervals':intervals},a_file)
    cPickle.dump({'Coords': A, 'structure_acronyms': structure_acronyms, 'intervals': intervals}, a_file)
    a_file.close()


def prepare_condition(session, session_id, lfp, probe_id, cond_name, Resultspath, Prestim, down_rate, do_save=True):
    """
    epoch the data and apply down-sampling
    :param session: allensdk session
    :param lfp: lfp matrix from session.get_lfp
    :param probe_id:
    :param cond_name: name of the condition to be processed
    :param Resultspath: path to save the results
    :param Prestim: pre-stimulus time window in seconds
    :param down_rate: temporal down-sampling rate
    :return: save the results in a binary format file and return the resulting sampling rate
    """
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

    # --------------------------Output File------------------------------
    # Times = results['time'].mean(axis=1)[:, 0]
    Time_nanzero = np.nansum(results['time'], axis=0) != 0
    Times = [np.mean(results['time'][:, Time_nanzero[:, x], x], axis=1) for x in range(0, Time_nanzero.shape[1])][0]

    Times[Times.shape[0] - 1] = Times[Times.shape[0] - 2] * 2 - Times[Times.shape[0] - 3]

    lfp_cond = (results['lfp'])
    if cond_name == 'flashes':
        lfp_cond = LFPF.csd(LFPF.gaussian_filter_trials(lfp_cond, 1))  # just in case of flashes for layer assignment
    else:
        lfp_cond = LFPF.bipolar(lfp_cond)  # LFPF.gaussian_filter_trials(lfp_cond, 1));

    Y = lfp_cond
    Y = np.moveaxis(Y, -2, 0)

    # downsampling the signal
    # Here I downsample the signal, to allow better estimation of FC in lower frequencies using AR models:
    Y = signal.decimate(Y, down_rate, axis=2)
    Times = Times[np.arange(0, Times.shape[0], down_rate)]
    dSRate = SRate / down_rate

    # start and stop time of each trial
    time_start_stop_trl = results['time_start_stop']

    # condition info
    cnd_id = results['cnd_id']
    cnd_info = list(map(lambda x: presentations[presentations['stimulus_condition_id'] == x].iloc[0],
                        cnd_id))  # each condition info
    cnd_info = pd.concat(cnd_info, axis=1).transpose()

    # convert to class?
    LFPdata = LFPprobe(session_id, probe_id, structure_acronyms[structure_acronyms.shape[0] - 2], Y, dSRate,
                       results['channel'], Times, cnd_id, cnd_info, time_start_stop_trl)

    if do_save:
        # save the data
        if not os.path.exists(os.path.join(Resultspath, 'PrepData')):
            os.mkdir(os.path.join(Resultspath, 'PrepData'))

        a_file = open(os.path.join(Resultspath, 'PrepData',
                                   '{}_{}{}_pres{}s.pkl'.format(probe_id, cond_name, int(down_rate), Prestim)), "wb")

        cPickle.dump(LFPdata.__dict__, a_file)
        a_file.close()

        return structure_acronyms[structure_acronyms.shape[0] - 2]
    else:
        return LFPdata


def CSD_plots(session, lfp, probe_id, Resultspath):
    """
    Plots the CSD maps of only cortical areas for layer assignment. Uses Flashes condition
    :param session: allenssdk session object
    :param lfp: lfp matrix from session.get_lfp
    :param probe_id:
    :param Resultspath: path to save the figures
    :return:
    """
    # --------------------------------------------------------
    # In case you want to check the original csd, run this:
    csd = session.get_current_source_density(probe_id)
    filtered_csd = gaussian_filter(csd.data, sigma=4)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.set_cmap('coolwarm')
    ax.pcolor(csd["time"], csd["vertical_position"], filtered_csd)

    ax.set_xlabel("time relative to stimulus onset (s)", fontsize=20)
    ax.set_ylabel("vertical position (um)", fontsize=20)

    structure_acronyms, intervals = session.channel_structure_intervals(lfp["channel"])
    # interval_midpoints = [aa + (bb - aa) / 2 for aa, bb in zip(intervals[:-1], intervals[1:])]
    plt.title(structure_acronyms[structure_acronyms.shape[0] - 2])
    fig.savefig('{}/Probe_{}_CSD_original.png'.format(Resultspath, probe_id), dpi=300)
    plt.close()
    # ------------------THIS IS FOR FLASHES CONDITION-----------
    # -compute CSD of the flashes condition
    presentations = session.get_stimulus_table('flashes')
    CI = presentations['stimulus_condition_id']

    Prestim = .05  # prestimulus time in sec
    results = LFPF.organize_epoch(lfp, presentations, Prestim, 0)

    # -Prepare Variables-
    CI = np.array(results['cnd_id'])
    CI = pd.DataFrame(CI, columns=['CID'])['CID']
    lfp_cond = (results['lfp'])
    lfp_cond = LFPF.csd(LFPF.gaussian_filter_trials(lfp_cond, 1))

    # -test the intervals-
    VISs = [x.find('VIS') for x in [str(x) for x in structure_acronyms]]
    VIS_ind = np.where(np.array(VISs) == 0)[0]
    lfp_cond = lfp_cond[intervals[VIS_ind.min()]:intervals[VIS_ind.max() + 1]]

    # lfp_cond = lfp_cond[intervals[intervals.shape[0] - 3]:intervals[intervals.shape[0] - 2]]

    # -remove trials with nan values as time-
    Time_nanzero = np.nansum(results['time'], axis=0) != 0
    Time = [np.mean(results['time'][:, Time_nanzero[:, x], x], axis=1) for x in range(0, Time_nanzero.shape[1])]

    # -Figure configs-
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    plt.set_cmap("coolwarm")

    # ---
    Ma = []
    for Cnd in np.array(range(CI.nunique() - 1, -1, -1)):  # range(0,CI.nunique()):
        TimeWin = Time[Cnd]
        ZeroPoint = abs(TimeWin[~np.isnan(TimeWin)] - .0).argmin()
        ax = axs[Cnd]
        lfp_cond_M = np.nanmean(lfp_cond[:, :, Time_nanzero[:, Cnd], Cnd], axis=2)
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
        ax.set_title(
            'Cond#{}-{}-{}'.format(CI.unique()[Cnd], structure_acronyms[structure_acronyms.shape[0] - 2], len(VIS_ind)))
        # fig.set_clim(-1,1)
        cbar = fig.colorbar(p, ax=ax)  # format=ticker.FuncFormatter(fmt))
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        Ma.append(abs(lfp_cond_M).max() / 1.5)
        p.set_clim(-max(Ma), max(Ma))

    plt.set_cmap("coolwarm")
    plt.tight_layout()
    fig.savefig('{}/Probe_{}_flashes_csd.png'.format(Resultspath, probe_id), dpi=300)
    plt.close()


def RF_mapping_plot(session, lfp, probe_id, doplot, Resultspath):
    """
    THIS IS RF MAPPING USING GABORS condition
    :param session:
    :param lfp:
    :param probe_id:
    :param doplot:
    :param Resultspath:
    :return:
    """

    presentations = session.get_stimulus_table('gabors')
    CI = presentations['stimulus_condition_id']
    Prestim = .0  # prestimulus time in sec, Notes: there is no ISI for gabors condition
    results = LFPF.organize_epoch(lfp, presentations, Prestim, 0)

    cnd_id = results['cnd_id']
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


def layer_selection(layer_table, probe_id, result_path):
    """
    Select 6 cortical layers based on manual labels in a XLSX file
    :param layer_table: loaded xlsx table in form of a dataframe
    :param probe_id: the probe in that session
    :param result_path: Path to session, to load probe information
    :return: the channel IDs of six layers for that probe
    """
    # read the probe info and return error if do not find any
    file = open(os.path.join(result_path, 'PrepData', '{}_ProbeInfo.pkl'.format(probe_id)), "rb")
    dataPickle = file.read()
    file.close()
    probe_info = cPickle.loads(dataPickle)

    # indicate layers channel ids and return them
    Coord = probe_info['Coords']
    Coord.sort_values('id', inplace=True, ascending=True)
    Coord.reset_index(inplace=True)

    # select structure acronym: find the one starts with VIs?
    # Attention: indexing is should be -1
    # Ind_area = range(probe_info['intervals'][-2]-1,probe_info['intervals'][-3]-1,-1)
    VISs = [x.find('VIS') for x in [str(x) for x in probe_info['structure_acronyms']]]  # in case of fragmented ROI
    VIS_ind = np.where(np.array(VISs) == 0)[0]
    Ind_area = range(probe_info['intervals'][VIS_ind.max() + 1] - 1, probe_info['intervals'][VIS_ind.min()] - 1, -1)
    Ind_area2 = range(probe_info['intervals'].max() - 1, probe_info['intervals'].min() - 1, -1)
    Ind_area2 = (probe_info['intervals'][VIS_ind.max() + 1] - 1) - np.array(Ind_area2)

    # Select the layers:
    #channel_ids = Coord['id'].loc[Ind_area]  # select only VISp area
    Ind_layers = layer_table['P{}'.format(probe_id)].loc[['L{}'.format(x) for x in range(1, 7)]] - 1
    #channel_Inds = probe_info['intervals'].max() - 1 - np.array([np.where(Ind_area2 == i)[0][0] for i in Ind_layers])

    # return the channels ids
    if np.isnan(Ind_layers).to_numpy().sum() == 0:
        channel_Inds = probe_info['intervals'].max() - 1 - np.array(
            [np.where(Ind_area2 == i)[0][0] for i in Ind_layers])
        return Coord['id'].iloc[channel_Inds]
    else:
        return []


def LFP_plot(Y, TimeWin, figure_path):
    nroi = len(Y.keys())
    fig, axs = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi), sharex=True)

    for i in range(0, nroi):
        roi = list(Y.keys())[i]
        T = Y[roi].time.values
        T_ind = np.where((T >= TimeWin[0]) & (T <= TimeWin[1]))[0]
        y = Y[roi].isel(time=T_ind)
        y = np.moveaxis(y.__array__(), -1, 0)
        dims = y.shape
        y2 = y.reshape(dims[0] * dims[1], dims[2], dims[3])
        MEAN = np.nanmean(y2, axis=0).transpose()
        SEM = (np.nanstd(y2, axis=0) / (y2.shape[0] ** .5)).transpose()
        offset = MEAN.max(axis=(0, 1))
        for l in range(0, MEAN.shape[1]):
            axs[i].plot(T[T_ind], MEAN[:, l] - (offset * l), linewidth=1, label='L{}'.format(l))
            axs[i].fill_between(T[T_ind], MEAN[:, l] - (offset * l) + SEM[:, l], MEAN[:, l] - (offset * l) - SEM[:, l],
                                alpha=.5)
            axs[i].set_title(roi)
            axs[i].set_yticks([])
            axs[i].axvline(x=0, linewidth=1, linestyle='--', color='k')
            if i == nroi - 1:
                axs[i].set_xlabel('Time')
                axs[i].set_xlim(TimeWin[0], TimeWin[1])
                axs[i].legend(loc='right')

    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


class LFPprobe(object):
    """
    A class to store the LFPs of each probe
    """

    def __init__(self, session_id, probe_id, ROI, Y, srate, channels=None, time=None, cnd_id=None, cnd_info=None, time_start_stop_trl=None):
        """

        :param ID: Probe ID
        :param ROI: ROI name
        :param Y: trial x channel x time x cnd
        :param srate: sampling rate
        """
        self.session_id = session_id
        self.probe_id = probe_id
        self.ROI = ROI
        if isinstance(Y, np.ndarray):
            self.Y = xr.DataArray(Y, dims=['trial', 'channel', 'time', 'cnd_id'],
                                  coords=dict(trial=range(0, Y.shape[0]), channel=channels, time=time, cnd_id=cnd_id))
        else:
            self.Y = Y
        self.srate = srate
        self.cnd_info = cnd_info
        self.time_start_stop_trl = time_start_stop_trl

    @classmethod  # Alternative constructor: read from file
    def from_file(cls, filename):
        with open(filename, "rb") as file:
            dataPickle = file.read()
            file.close()

        Arg_dict = cPickle.loads(dataPickle)

        return cls(**Arg_dict)


def layer_reduction(Y, FS, probe_id, result_path):
    """
    Select 6 cortical layers based on dimensionality reduction methods
    :param Y:
    :param probe_id: the probe in that session
    :param result_path: Path to session, to load probe information
    :return: ?
    """
    # read the probe info and return error if do not find any
    file = open(os.path.join(result_path, 'PrepData', '{}_ProbeInfo.pkl'.format(probe_id)), "rb")
    dataPickle = file.read()
    file.close()
    probe_info = cPickle.loads(dataPickle)

    # select structure acronym: find the one starts with VIs?
    # Attention: indexing is should be -1
    # Ind_area = range(probe_info['intervals'][-2]-1,probe_info['intervals'][-3]-1,-1)
    VISs = [x.find('VIS') for x in [str(x) for x in probe_info['structure_acronyms']]]  # in case of fragmented ROI
    VIS_ind = np.where(np.array(VISs) == 0)[0]
    offset = 5
    Ind_area = range(min(probe_info['intervals'][VIS_ind.max() + 1] - 1 + offset, probe_info['intervals'].max() - 1),
                     probe_info['intervals'][VIS_ind.min()] - 1, -1)
    Y2 = Y[:, Ind_area, :, :]
    Y2 = np.moveaxis(Y2.__array__(), 0, -1)
    dims = Y2.shape
    Y2 = Y2.reshape(dims[0], dims[1] * dims[2] * dims[3], order="F")
    Y2 = Y2[:, np.where(~np.isnan(Y2[1,:]))[0]]
    # only hig gamma for layer selection

    sos = signal.butter(10, 400, 'hp', fs=FS, output='sos')
    Y2_filtered = signal.sosfilt(sos, Y2)


    # spectrum
    res = signal.welch(Y2, fs=FS)
    Freqs = res[0]
    PWelch = np.array(res[1])
    PWelch_norm = np.divide(PWelch, np.linalg.norm(PWelch, axis=0))

    return {'high_gamma_amp': np.absolute(Y2_filtered).sum(axis=1),
            # 'low_gamma_amp': np.absolute(Y2_filtered_low).sum(axis=1),
            'Pwelch_norm': PWelch_norm,
            'Pwelch_Freq': Freqs,
            'labels': (probe_info['intervals'][VIS_ind.max() + 1] - 1) - np.array(Ind_area)}

    """
    ica = FastICA(n_components=6, random_state=0)
    X_transformed = ica.fit_transform(Coh.reshape(Coh.shape[0],Coh.shape[1]*Coh.shape[2]))
    A_ica = ica.mixing_

    pca = PCA(n_components=6)
    H = pca.fit_transform(Y2_filtered.transpose())
    A_pca = pca.components_
    plt.plot(A_pca.transpose())
    """
