
import h5py
import pandas as pd
import scipy.signal
import numpy as np


def loadECGHDF52Dataframe(hdf5_file_name):
    ecg_1d_data = {'file_name':[],'number':[],'year':[],'leads':[]}
    with h5py.File(hdf5_file_name, 'r') as hf:
        for key in hf.keys():
            # ecg_1d_data[key] = hf[key][:]
            leads_data = hf[key][:]

            # convert uv to mV
            leads_data = leads_data / 1000
            # resample to 1024
            leads_data = scipy.signal.resample(leads_data, 1024, axis=1)
            
            if np.isnan(leads_data).any():
                print(key, 'nan')
                continue
            else:
                ecg_1d_data['leads'].append(leads_data)
                ecg_1d_data['file_name'].append(key)
                ecg_1d_data['number'].append(int(key.split('_')[1]))
                ecg_1d_data['year'].append(int(key.split('_')[2][0:4]))
            if hf[key].shape[0] != 12:
                print(key, hf[key].shape)
    
    ecg_df = pd.DataFrame(ecg_1d_data)
    return ecg_df
