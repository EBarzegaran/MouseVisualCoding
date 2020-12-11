
import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import MouseVisCode.Probe_functions as ProbeF

class LFPSession(object):

    """
    Class to store and retrieve LFP data and estimate iPDC
    """

    def __init__(self,session,result_path,cond_name='drifting_gratings',preprocessed = False):

        self.session_id = session['ecephys_session_id']

        # Add the resultpath folder for this session #### be careful about this variable when saving and loading (both Paths)
        if not os.path.exists(os.path.join(result_path, str(self.session_id))):
            os.mkdir(os.path.join(result_path, str(self.session_id)))
        self.result_path = os.path.join(result_path, str(self.session_id))

        # Get allenSDK session. This variable only exist when running code, will not save in the file
        self.session = session

        # Get the probes for this session
        self.probe_ids = session.probes.index.values.shape[0]

        self.cond_name = cond_name
        self.preprocessed = preprocessed


    def preprocessing(self,doRF):
        # Get allenSDK session (exceptions for wrong paths)

        for I in range(0,self.session.probes.index.values.shape[0]):
            probe_id =self.session.probes.index.values[I]

            # -load lfp data
            lfp =self.session.get_lfp(probe_id)

            # -first extract probe info and save
            ProbeF.extract_probeinfo(self.session, lfp, probe_id, self.result_path, doRF)

            # -extract and prepare the data for a condition
            cond_name = 'drifting_gratings_75_repeats'  # 'drifting_gratings'  #'flashes'
            Prestim = 1  # prestimulus time in sec
            down_rate = 5  # down sampling -> the original sampling rate is 1250 Hz
            ProbeF.prepare_condition(self.session, lfp, probe_id, self.cond_name, self.result_path, Prestim, down_rate)

            # -CSD plot for the probe
            ProbeF.CSD_plots(self.session, lfp, probe_id, self.result_path)

        self.preprocessed=True

## class method read from file