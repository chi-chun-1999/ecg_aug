#%%
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#%%
from transforms import *
from composition import *
from util.data import loadECGHDF52Dataframe
import torch
import matplotlib.pyplot as plt
from util.show import show_ecg

#%%
# load ecg data
ecg_1d_file_name = '../../ecg_regression4liver/data/h5py/Type1Digitalv1/Type1Digtal-Save_Img_dmnoviolate_all.hdf5'

ecg_df = loadECGHDF52Dataframe(ecg_1d_file_name)
# %%

tmp = ecg_df['leads'][0:5]

ori_sample = torch.tensor(tmp)
plt.plot(ori_sample[0,0,:].detach().numpy())

#%%

compose_transform = Compose([
                            Standardize(0.5, 1),
                            OneOf([
                                 BaselineWander(5, 1), 
                                 RandDisplacement(1,1)
                                 ]),

                            SomeOf([
                                SineNoisePartial(0.5,1),
                                WhiteNoisePartial(0.5,1),
                                GaussianNoisePartial(0.5,1)
                                    ],2,1)
                             ])

transform = compose_transform(ori_sample)

show_ecg(transform[1])
show_ecg(ori_sample[1])
# %%
