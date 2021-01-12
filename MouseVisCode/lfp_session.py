
import os
import MouseVisCode.probe_functions as ProbeF
import _pickle as cPickle
import pandas as pd


class LFPSession(object):
    """
    Class to access, store, and retrieve LFP session data, apply pre-processing and estimate iPDC
    ------------------------------------------------------------------------
    ------------------------------------------------------------------------
    """

    def __init__(self,cache,session_id,result_path):

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
        filename = os.path.join(self.result_path, 'LFPSession_{}.obj'.format(self.session_id))
        filehandler = open(filename, "wb")
        # Do not save the loaded LFP matrices since they are too big
        temp = self
        temp.probes = dict.fromkeys(temp.probes.keys())
        temp.loaded_cond = None
        temp.layer_selected = False
        cPickle.dump(temp.__dict__, filehandler)
        filehandler.close()

    def load_session(self): # be careful about this -> result_path
        filename = os.path.join(self.result_path, 'LFPSession_{}.obj'.format(self.session_id))
        file = open(filename, 'rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = cPickle.loads(dataPickle)

    def __str__(self):
        return str(self.__dict__).replace(", '", ",\n '")

    ## Processing methods
    def preprocessing(self,cond_name='drifting_gratings', down_sample_rate=5, Prestim = 1, do_RF=False, do_CSD=False, do_probe=False):
        """
        Runs the preprocessing on the session with the input parameters, if it has not been run before.

        :param cond_name: condition name to be preprocessed
        :param do_RF: do receptive field mapping plots? Attention: this may take a while if set True, note it is not RF mappning based on permutation
        :param down_sample_rate:
        :param Prestim: prestimulus time in sec
        :return:
        """
        # first indicate if the
        preproc_dic = {
            'cond_name': cond_name,
            'srate': down_sample_rate,
            'prestim': Prestim,
            }
        #'layer_selected':False# include in the file name

        if not search_preproc(self.preprocess,preproc_dic):

            for probe_id in self.probes.keys():

                # Load lfp data
                lfp =self.session.get_lfp(probe_id)

                # First extract probe info and save
                if do_RF:
                    ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, do_RF)
                    self.RF = True
                elif not self.RF or do_probe:
                    ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, False)
                    self.RF = True

                # CSD plot for the probe
                if (not self.CSD) and do_CSD:
                    ProbeF.CSD_plots(self.session, lfp, probe_id, self.result_path)

                # Extract and prepare the data for a condition
                if cond_name is not None:
                    ROI = ProbeF.prepare_condition(self.session, self.session_id, lfp, probe_id, cond_name, self.result_path, Prestim,down_sample_rate)
                    self.ROIs[ROI] = probe_id

            # Add the pre-process params as a dictionary to the list of preprocessed data
            if cond_name is not None:
                self.preprocess.append(preproc_dic)

            if (not self.CSD) and do_CSD:
                self.CSD = True

            # Save the session after preprocessing
            self.save_session()

    def load_LFPprobes(self, cond):
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
            channel_id = ProbeF.layer_selection(layer_table, probe_id, self.result_path)

            # select the LFP of those channels, and relabel the xarray dimensions
            self.probes[probe_id].Y = self.probes[probe_id].Y.sel(channel=channel_id.to_list())

        self.layer_selected = True


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