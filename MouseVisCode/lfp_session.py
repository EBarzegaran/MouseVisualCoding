
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import MouseVisCode.probe_functions as ProbeF

class LFPSession(object):

    """
    Class to store and retrieve LFP data and estimate iPDC
    """

    def __init__(self,cache,session_id,result_path,cond_name='drifting_gratings'):

        self.session_id = session_id

        # Add the resultpath folder for this session #### be careful about this variable when saving and loading (both Paths)
        if not os.path.exists(os.path.join(result_path, str(self.session_id))):
            os.mkdir(os.path.join(result_path, str(self.session_id)))
        self.result_path = os.path.join(result_path, str(self.session_id))

        # Get allenSDK session. This variable only exist when running code, will not save in the file
        self.session = cache.get_session_data(session_id)

        # Get the probes for this session
        self.probe_ids = self.session.probes.index.values

        self.cond_name = cond_name
        self.preprocessed = False
        self.layer_selected = False


    def preprocessing(self,do_RF=True,down_sample_rate=5):

        for probe_id in self.probe_ids:

            # load lfp data
            lfp =self.session.get_lfp(probe_id)

            # first extract probe info and save
            ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, do_RF)

            # extract and prepare the data for a condition
            Prestim = 1  # prestimulus time in sec
            ProbeF.prepare_condition(self.session, lfp, probe_id, self.cond_name, self.result_path, Prestim,down_sample_rate)

            # CSD plot for the probe
            ProbeF.CSD_plots(self.session, lfp, probe_id, self.result_path)

        self.preprocessed=True


    def layerselection(self):
        for I in range(0, self.session.probes.index.values.shape[0]):
            # if the layer file did not exist then return with a warning
            print(I)# downsample the
        self.layer_selected = True

## class method read/write the LFPSession from/to file