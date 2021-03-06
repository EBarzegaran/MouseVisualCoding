
import os
import probe_functions as ProbeF
import pdc_functions as PDCF
import _pickle as cPickle
import pandas as pd
import numpy as np
from functools import reduce
import dynet_statespace as dsspace
import dynet_con as dcon
import xarray as xr
import matplotlib.pyplot as plt
from functools import reduce

class LFPSession(object):
    """
    Class to access, store, and retrieve LFP session data, apply pre-processing and estimate iPDC
    """

    def __init__(self,cache,session_id,result_path):
        """
        Initialize the class based on AllenBrainSDK session
        :param cache: cache from EcephysProjectCache.from_warehouse(manifest=manifest_path)
        :param session_id: ID for allenSDK session
        :param result_path: Path to save the results
        """
        self.session_id = session_id

        # Add the resultpath folder for this session #### be careful about this variable when saving and loading (both Paths)
        if not os.path.exists(os.path.join(result_path, str(self.session_id))):
            os.mkdir(os.path.join(result_path, str(self.session_id)))
        self.result_path = os.path.join(result_path, str(self.session_id))

        # check if the LFP session already exist, load that session preprocessing info
        try:
            self.load_session()
        except FileNotFoundError:
            # self.cond_name = cond_name
            self.preprocess = []  # any preprocessing is done? list of the preprocessing params
            self.RF = False  # Channel info is stored?
            self.CSD = False  # CSD plots for layer assignment are done before?
            self.ROIs = {}  # empty dictionary indicating cortical ROI (VIS areas) and their relative probes
            self.session = cache.get_session_data(session_id)  # Get allenSDK session
            # variables for running time only
            self.probes = dict.fromkeys(self.session.probes.index.values) # Get the probes for this session, make a dictionary maybe
            self.loaded_cond = None  #Load LFP option
            self.layer_selected = False  # if the loaded LFP is spatially down-sampled

    ## Class methods read/write the LFPSession from/to file (note: only preprocessing info is important)
    def save_session(self):
        """
        Saves session and preprocessing information to a .obj file using cPickle
        :return: file path/name
        """
        filename = os.path.join(self.result_path, 'LFPSession_{}.obj'.format(self.session_id))
        filehandler = open(filename, "wb")
        # Do not save the loaded LFP matrices since they are too big
        temp = self
        temp.probes = dict.fromkeys(temp.probes.keys())
        temp.loaded_cond = None
        temp.layer_selected = False
        cPickle.dump(temp.__dict__, filehandler)
        filehandler.close()
        return filename

    def load_session(self): # be careful about this -> result_path
        filename = os.path.join(self.result_path, 'LFPSession_{}.obj'.format(self.session_id))
        file = open(filename, 'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)

    def __str__(self):
        return str(self.__dict__).replace(", '", ",\n '")

    ## Processing methods
    def preprocessing(self,cond_name='drifting_gratings', down_sample_rate=5, pre_stim = 1, do_RF=False, do_CSD=False, do_probe=False):
        """
        Runs the preprocessing on the session with the input parameters, if it has not been run before.

        :param cond_name: condition name to be preprocessed
        :param do_RF: do receptive field mapping plots? Attention: this may take a while if set True, note it is not RF mappning based on permutation
        :param down_sample_rate:
        :param pre_stim: prestimulus time in sec
        :return:
        """
        # first indicate if the
        preproc_dict = {
            'cond_name': cond_name,
            'srate': down_sample_rate,
            'prestim': pre_stim,
            }

        # Attention: remove the zero conditions

        if not search_preproc(self.preprocess,preproc_dict):

            for probe_id in self.probes.keys():

                # Load lfp data
                lfp =self.session.get_lfp(probe_id)

                # First extract probe info and save
                if do_RF:
                    ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, do_RF)
                    self.RF = True
                elif not self.RF or do_probe:
                    ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, False)


                # CSD plot for the probe
                if (not self.CSD) and do_CSD:
                    ProbeF.CSD_plots(self.session, lfp, probe_id, self.result_path)

                # Extract and prepare the data for a condition
                if cond_name is not None:
                    ROI = ProbeF.prepare_condition(self.session, self.session_id, lfp, probe_id, cond_name, self.result_path, pre_stim, down_sample_rate)
                    self.ROIs[ROI] = probe_id

            # Add the pre-process params as a dictionary to the list of preprocessed data
            if cond_name is not None:
                self.preprocess.append(preproc_dict)

            if (not self.CSD) and do_CSD:
                self.CSD = True

            if not self.RF or do_probe:
                self.RF = True

            # Save the session after preprocessing
            self.save_session()

    def load_LFPprobes(self, cond_dict):
        """
        loads in the preprocessed LFP signal
        :param cond_dict: a dictionary with the preprocessing params
        :return: Updates the self.probes values
        """
        preprocess_ind = search_preproc(self.preprocess, cond_dict)
        if not preprocess_ind: # checks if the condition is previously run
            print("no preprocessing with these parameters is done")
            return

        cond = self.preprocess[preprocess_ind[0]]
        for probe_id in self.probes.keys():
            # first prepare the file name
            filename = os.path.join(self.result_path, 'PrepData', '{}_{}{}_pres{}s.pkl'.format(
                probe_id, cond['cond_name'], int(cond['srate']),cond['prestim']))
            # second load each probe and add it to the ROI list
            self.probes[probe_id] = ProbeF.LFPprobe.from_file(filename)
        self.loaded_cond =  cond['cond_name']

    def layer_selection(self, Filename=None):
        """
        This will be done on the loaded_cond data
        :return:
        """
        if Filename==None:
            Filename = os.path.join(self.result_path,'PrepData','Cortical_Layers.xlsx')
        try:
            layer_table = pd.read_excel(Filename)
            # set the layer names as index of the dataframe
            layer_table.set_index('Layers', inplace=True)

        except OSError:
            # if the layer file did not exist then return with an error
            print("Prepare the cortical layer files first as PrepData/Cortical_Layers.xlsx")
            return

        for probe_id in self.probes.keys():
            print(probe_id)
            #ProbeF.layer_reduction(self.probes[probe_id].Y, probe_id, self.result_path)

            channel_id = ProbeF.layer_selection(layer_table, probe_id, self.result_path)
            # select the LFP of those channels, and relabel the xarray dimensions
            if len(channel_id) > 0:
                self.probes[probe_id].Y = self.probes[probe_id].Y.sel(channel=channel_id.to_list())
            else:
                self.probes[probe_id].Y = []

        self.layer_selected = True

    def pdc_analysis(self, ROI_list=None, Mord=10, ff=.99, pdc_method='iPDC', stim_params=None, Freqs=np.array(range(1, 101)), preproc_params=None, redo = False):
        """
        Calculates time- and frequency-resolved functional connectivity between the LFP signals based on STOK algorithm
        :param ROI_list: list of ROIs to be considered for this analysis
        :param Mord: Model order for ARMA model
        :param ff: filter factor between 0 and 1
        :param pdc_method: check the pydynet toolbox for that
        :param stim_params: Parameters of stimulus to be used to pool the data
        :param Freqs: a numpy array uncluding the Frequencies for connectivity analysis
        :return:
        """
        if ROI_list is None:
            ROI_list = ['VISp']
        if stim_params is None:
            stim_params = []

        #----------------------------------------------------------------------------
        # Check if the PDC exist, just load it
        # analysis params
        PDCparam_dict = {
            'ROI_list': ROI_list,
            'Mord': Mord,
            'ff': ff,
            'pdc_method': pdc_method,
            'stim_param': stim_params
        }

        filename = PDCF.search_PDC(self.session_id, self.result_path, PDCparam_dict, preproc_params)
        if os.path.isfile(filename) and not redo:
            # load the file and return it
            file = open(filename, 'rb')
            PDC_dict = cPickle.load(file)
            file.close()
            return PDC_dict

        #----------------------------------------------------------------------------
        # load the preprocessed LFPs and down sample spatially by selecting 6 layers
        self.load_LFPprobes(preproc_params)
        self.layer_selection()

        # select the conditions and pool their trials together
        Result_pool = self.pool_data(preproc_params=preproc_params, stim_params= stim_params, ROI_list = ROI_list)
        Y = Result_pool['Y']
        Srate = Result_pool['Srate']

        # pull together and ROI-layer index
        srate = np.unique(np.array(list(Srate.values())))
        if len(srate) != 1:
            print("Sampling rates do not match between probes, please check the preprocessing!")
            return

        # Put the data from all ROIs together for PDC calculations
        Y_temp = np.concatenate(list(Y.values()), axis=1)  # second dimension is the channels
        Y_temp = np.moveaxis(Y_temp, -1, 0)
        YS = list(Y_temp.shape)
        Y_pool = Y_temp.reshape([YS[0] * YS[1], YS[2], YS[3]])
        # remove possible zero and NaN values (trials)
        nzero_trl = Y_pool[:, :, 10] != 0
        nzero_trl_ind = reduce((lambda x, y: np.logical_or(x, y)), nzero_trl.transpose())
        nNan_trl_ind = np.isnan(Y_pool).sum(axis=2).sum(axis=1) == 0
        Y_pooled = Y_pool[nzero_trl_ind & nNan_trl_ind, :, :]


        # iPDC matrix
        KF = dsspace.dynet_SSM_STOK(Y_pooled, p=Mord, ff=ff)
        iPDC = dcon.dynet_ar2pdc(KF, srate, Freqs, metric=pdc_method, univ=1, flow=2, PSD =1)
        # iPDC to xarray
        Time = Y['VISp'].time.values
        ROI_ls = np.array(Result_pool['ROI_labels']).reshape(np.prod(np.array(Result_pool['ROI_labels']).shape))
        iPDC_xr = xr.DataArray(iPDC, dims=['target', 'source', 'freq' , 'time'],
                         coords=dict(target= ROI_ls, source= ROI_ls, freq=Freqs, time=Time))
        # ROIs for output
        ROIs = list(Y.keys())
        chnl_ids = np.array(Result_pool['channel_ids']).reshape(np.prod(np.array(Result_pool['channel_ids']).shape))
        prb_ids = np.array(Result_pool['probe_ids']).reshape(np.prod(np.array(Result_pool['probe_ids']).shape))

        # save and return the output
        PDC_dict = {'session_id':self.session_id, 'KF': KF, 'ROIs': ROIs, 'PDC': iPDC_xr,
                'probe_info': {'probe_ids': prb_ids, 'channel_ids': chnl_ids}, 'PDCparam_dict': PDCparam_dict, 'preproc_dict': preproc_params}

        PDCF.save_PDC(PDC_dict, self.result_path)

        # save?
        return PDC_dict

    def pool_data(self, preproc_params=None, stim_params= None, ROI_list = None):

        # select the conditions and pool their trials together
        Y = {}  # to prepare the data for PDC analysis
        Srate = {}  # to make sure that Srates match
        ROI_labels = []
        channel_ids = []
        probe_ids = []
        # All ROIs in this session
        All_ROIs = [(self.probes[x].ROI, x) for x in self.probes.keys()]

        for ROI in ROI_list:
            # find the ROIs and the one with Layer assignment
            ch_ind = [i for i, y in enumerate([x[0] for x in All_ROIs]) if y == ROI]
            if bool(ch_ind): # in case of multiple recordings from the same ROI, I only labeled the one with better data
                temp = [len(self.probes[All_ROIs[x][1]].Y)>0 for x in ch_ind]
                Emp_ind = np.where(np.array(temp))[0]# find empty probes -> because no layer was assigned
                if len(Emp_ind)>0:
                    ch_ind = ch_ind[Emp_ind[0]]
                    #ch_ind = ch_ind[temp.index(True)]
                else:
                    ch_ind = []

            if bool(ch_ind) or (ch_ind==0): #if there is a probe
                probe_id = All_ROIs[ch_ind][1]
                cnd_info = self.probes[probe_id].cnd_info
                Cnds_inds = []
                for k in stim_params.keys():
                    Cnds = [cnd_info[k] == x for x in stim_params[k]]
                    if len(Cnds) > 1:
                        Cnds_temp = reduce((lambda x, y: np.logical_or(x, y)), [c.to_numpy() for c in Cnds])
                        Cnds_inds.append(Cnds_temp)
                    else:
                        Cnds_inds.append(Cnds)
                Cnds_final = np.array(reduce((lambda x, y: np.logical_and(x, y)), Cnds_inds))
                Cnds_inds_final = cnd_info['stimulus_condition_id'].to_numpy()[Cnds_final.squeeze()]

                # Prepare for output
                Y[ROI] = self.probes[probe_id].Y.sel(cnd_id=Cnds_inds_final)
                Srate[ROI] = self.probes[probe_id].srate
                ROI_labels.append(['{}_L{}'.format(ROI, i) for i in range(1, 7)])
                channel_ids.append(Y[ROI].channel.values)
                probe_ids.append([probe_id for l in range(1, 7)])



        # Set other outputs
        Time = Y['VISp'].time.values
        ROIs = list(Y.keys())

        return {'Y': Y, 'Srate': Srate, 'ROI_labels':ROI_labels, 'channel_ids':channel_ids, 'probe_ids':probe_ids}

    def plot_LFPs(self, preproc_params=None, stim_params= None, ROI_list = None, TimeWin=None):

        self.load_LFPprobes(preproc_params)
        self.layer_selection()

        Result_pool = self.pool_data(preproc_params=preproc_params, stim_params=stim_params, ROI_list=ROI_list)
        figure_path = os.path.join(self.result_path, 'Average_LFP_{}_downs{}.png'.format(
             preproc_params['cond_name'], int(preproc_params['srate'])))

        colors = ROIColors('layers')
        LFP_plot(Result_pool['Y'],TimeWin, colors, figure_path)
        # Return averaged Y
        return dict((x,y.mean(axis=(0,3))) for x,y in Result_pool['Y'].items())


def search_preproc(list_pre, dic_pre):
    """
    Search if the preprocessing with the current parameters has been run before
    :param list_pre: self.preprocess
    :param dic_pre: dictionary with new params
    :return: The index of pre-processes with current params
    """
    result = []
    for x in list_pre:
        shared_items = [x[k] == dic_pre[k] for k in x if k in dic_pre]
        result.append(sum(shared_items)==len(dic_pre))
    return [i for i, x in enumerate(result) if x]
    # maybe also searches if the files exist?


class ROIColors(object):
    """
    A Class that defines uniform colorings for ROIs and layers for visualization
    """

    def __init__(self,color_type='uni'):
        """
        Initializes the colors class
        :param color_type: 'uni'/'layers' indicate if it should return only one color per ROI ('Uni')
                            or 6 colors per ROI, for 6 layers('Layers')
        """
        roi_colors_rgb = {'VISp': [.43, .25, .63], 'VISl': [0.03, 0.29, 0.48], 'VISrl': [0.26, 0.68, 0.76],
                          'VISal': [0.65, 0.46, 0.11], 'VISpm': [1, .7, .3], 'VISam': [0.8, 0.11, 0.11]}
        self.ROI_names = {'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL', 'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'}

        if color_type == 'uni':
            self.roi_colors_rgb = roi_colors_rgb
            self.roi_colors_hex = dict((x, '#%02x%02x%02x' % (int(v[0] * 255), int(v[1] * 255), int(v[2] * 255))) for x, v in
                                  roi_colors_rgb.items())
        elif color_type =='layers':
            offset = np.arange(-.25,.26,.1)
            roi_colors_rgb_layers = dict(
                (x, np.array([np.minimum(np.maximum(v + x, 0), 1) for x in offset])) for x, v in roi_colors_rgb.items())
            self.roi_colors_rgb = roi_colors_rgb_layers

            self.roi_colors_hex = dict((x,['#%02x%02x%02x' % (int(v[0]*255), int(v[1]*255), int(v[2]*255)) for v in k])
                for x,k in roi_colors_rgb_layers.items())

        else:
            print ('Wrong color type')
            return

        self.color_type = color_type


def LFP_plot(Y, TimeWin, colors, figure_path):
    """
    A general function to plot LFP averages
    :param Y: LFP data with dimensions :trials x layers x time x conditions
    :param TimeWin:
    :param colors:
    :param figure_path:
    :return:
    """
    nroi = len(Y.keys())
    fig, axs = plt.subplots(nrows=nroi, ncols=1, figsize=(6, 2 * nroi), sharex=True)
    # ordered ROIs: for uniformity puporse
    ordered_rois = ['VISp','VISl','VISrl','VISal','VISpm','VISam']
    ROIs = list(filter(lambda x: (x in list(Y.keys())), ordered_rois))

    # for each ROI plot mean and SEM
    for i in range(0, nroi):
        roi = ROIs[i]
        T = Y[roi].time.values
        T_ind = np.where((T >= TimeWin[0]) & (T <= TimeWin[1]))[0]
        y = Y[roi].isel(time=T_ind)
        y = np.moveaxis(y.__array__(), -1, 0)
        dims = y.shape
        y2 = y.reshape(dims[0] * dims[1], dims[2], dims[3])
        MEAN = np.nanmean(y2, axis=0).transpose()
        SEM = (np.nanstd(y2, axis=0) / (y2.shape[0] ** .5)).transpose()
        offset = abs(MEAN).max(axis=(0, 1))
        yticks = np.zeros([MEAN.shape[1],1])
        for l in range(0, MEAN.shape[1]):
            MEAN_plot = MEAN[:, l] - (offset * l)
            axs[i].plot(T[T_ind], MEAN_plot,
                        linewidth=1, label='L{}'.format(l), color=colors.roi_colors_hex[roi][l])
            axs[i].fill_between(T[T_ind], MEAN[:, l] - (offset * l) + SEM[:, l], MEAN[:, l] - (offset * l) - SEM[:, l],
                                alpha=.5, color=colors.roi_colors_hex[roi][l])
            yticks[l]= MEAN_plot[T[T_ind]<0].mean()
        axs[i].set_title(colors.ROI_names[roi])
        axs[i].set_yticks(yticks)
        axs[i].set_yticklabels(['L{}'.format(i+1) for i in range(0, MEAN.shape[1])])
        axs[i].axvline(x=0, linewidth=1, linestyle='--', color='k')
        axs[i].grid(True)

        if i == nroi - 1:
            axs[i].set_xlabel('Time(S)',fontweight='bold')
            axs[i].set_xlim(TimeWin[0], TimeWin[1])
            #axs[i].legend(loc='right')

    plt.savefig(figure_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def aggregate_LFP_ROI(Y_list):
    """

    :param Y_list:
    :return:
    """
    ROIs_All = reduce(lambda x, y: list(set().union(x, y)), [x.keys() for x in Y_list.values()])

    Y_ROI_all = {'session_ids': Y_list.keys(),
                   'ROIs': ROIs_All,
                   'Y': {}}
    # first indicate the ROIs in the list
    for roi in ROIs_All:
        s_ids = np.where(np.array([list(x.keys()).count(roi) > 0 for x in Y_list.values()]))[0]
        # -for animals with that ROI: make a list and concatenate them-
        LFP_temp = [Y_list[list(Y_list.keys())[x]][roi] for x in s_ids]
        # -time indexes with non NaN values and round them 3 digit to be uniform-
        NNan_ind = [np.logical_not(np.isnan(x.time.values)) for x in LFP_temp]
        NNan_ind = reduce(lambda x, y: np.logical_and(x[:min(len(x), len(y))], y[:min(len(x), len(y))]), NNan_ind)
        LFP_temp2 = []

        for lfp in LFP_temp:  # loop over animals
            lfp.time.values = np.round(lfp.time.values, 3)
            lfp.channel.values = np.arange(0,len(lfp.channel.values))
            LFP_temp2.append(lfp.isel(time=np.where(NNan_ind)[0]))

        # -calculate average over animals-??
        #Y_ROI_all['Y'][roi] = np.array(LFP_temp2).mean(axis=0)
        Y_temp = np.expand_dims(np.array(LFP_temp2),axis=3)
        Y_ROI_all['Y'][roi] = xr.DataArray(Y_temp, dims=['trial', 'channel', 'time', 'cnd_id'],
                              coords=dict(trial=range(0, Y_temp.shape[0]), channel=lfp.channel.values, time=lfp.time.values[:Y_temp.shape[2]], cnd_id=[1]))

    return Y_ROI_all