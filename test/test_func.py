#%%
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


#%%
from transforms import *
from util.data import loadECGHDF52Dataframe
import torch
import matplotlib.pyplot as plt
from util.show import show_ecg
from functional import *

#%%
# load ecg data
ecg_1d_file_name = '../../ecg_regression4liver/data/h5py/Type1Digitalv1/Type1Digtal-Save_Img_dmnoviolate_all.hdf5'

ecg_df = loadECGHDF52Dataframe(ecg_1d_file_name)

#%%

std_sample = torch.tensor(ecg_df['leads'][0:32])
std = Standardize(0.5, 1)

std_sample = std(std_sample)
# %%
aug_ecg = square_noise(std_sample, torch.tensor(3))

plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,0,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,1,:].detach().numpy())
plt.show()

# %%

aug_ecg = white_noise(std_sample, 0.075)

plt.figure(figsize=(15,5))
plt.ylim(-2, 2)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,10,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-2, 2)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(std_sample[1,10,:].detach().numpy())
plt.show()
# %%

aug_ecg = sine_noise_partial(std_sample, 0.5)

plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,0,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,1,:].detach().numpy())
plt.show()
# %%

aug_ecg = square_noise_partial(std_sample, 0.5)

plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,0,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,1,:].detach().numpy())
plt.show()
# %%

aug_ecg = white_noise_partial(std_sample, 0.5)

plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,0,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,1,:].detach().numpy())
plt.show()

#%%

aug_ecg = gaussian_noise_partial(std_sample, 0.5)

plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(aug_ecg[1,0,:].detach().numpy())
plt.show()
plt.figure(figsize=(15,5))
plt.ylim(-5, 5)
plt.plot([i for i in range(1024)],[0 for i in range(1024)],'r')
plt.plot(std_sample[1,0,:].detach().numpy())
plt.show()
# %%
