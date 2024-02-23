#%%
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#%%
from tramsforms import *
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


#%% test Standardize

std = Standardize(0.5, 1)

transform = std(ori_sample)
sample = transform

# sample = torch.tensor(tmp)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()

show_ecg(transform[1])
show_ecg(ori_sample[1])


#%%

# test NoOp

trans = NoOp(0.5, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1])
show_ecg(sample[1])

# %%

# test RandTemporalWarp

trans = RandTemporalWarp(2, 1)

transform = trans(sample)


# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))

# %%

# test BaselineWander

trans = BaselineWander(5, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))

# %%

# test GaussianNoise

trans = GaussianNoise(1, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))


# %%

# test RandCrop

trans = RandCrop(0.1, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))

#%%

trans = RandDisplacement(1, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))
# %%

# test SpecAug

trans = SpecAug(0.1, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))
# %%

# test MagnitudeScale

trans = MagnitudeScale(-2, 1)

transform = trans(sample)

# plt.plot(transform[1,0,:].detach().numpy())
# plt.show()
# plt.plot(sample[1,0,:].detach().numpy())
# plt.show()
show_ecg(transform[1],figsize=(15,15))
show_ecg(sample[1],figsize=(15,15))

# %%
