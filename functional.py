import numpy as np
from torch.nn import functional as F

import torch
from torch.autograd import Function
from scipy import signal as scipy_signal
import biosppy

FREQ = 1024/4.8

def rand_temporal_warp(x, mag, warp_obj):
    mag = 100*(mag**2)
    return warp_obj(x, mag)

def baseline_wander(x, mag):
    BS, C, L = x.shape

    # form baseline drift
    strength = 0.25*torch.sigmoid(mag) * (torch.rand(BS).to(x.device).view(BS,1,1))
    strength = strength.view(BS, 1, 1)

    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)  # typical breaths per second for an adult
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    drift = strength*torch.sin(torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()).to(x.device)
    return x + drift

def gaussian_noise(x, mag):
    BS, C, L = x.shape
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.25*stdval*torch.sigmoid(mag)*torch.randn(BS, C, L).to(x.device)
    return x + noise

def rand_crop(x, mag):
    x_aug = x.clone()
    # get shapes
    BS, C, L = x.shape
    mag = mag.item()

    nmf = int(mag*L)
    start = torch.randint(0, L-nmf,[1]).long()
    end = (start + nmf).long()
    x_aug[:, :, start:end] = 0.
    return x_aug


def spec_aug(x, mag):
    num_ch = 12
    x_aug = x.clone()
    BS, C, L = x.shape
    mag = mag.item()
    
    # get shapes
    BS, NF, NT, _ = torch.stft(x[:,0,], n_fft=512, hop_length=4,return_complex=False).shape
    nmf = int(mag*NF)
    start = torch.randint(0, NF-nmf,[1]).long()
    end = (start + nmf).long()
    for i in range(12):
        stft_inp = torch.stft(x[:,i,], n_fft=512, hop_length=4,return_complex=True)
        idxs = torch.zeros(*stft_inp.shape).bool()
        stft_inp[torch.arange(BS).long(), start:end,:] = 0
        x_aug[:, i] = torch.istft(stft_inp, n_fft=512, hop_length=4)
    
   

    nmf = int(mag*L)
    start = torch.randint(0, L-nmf,[1]).long()
    end = (start + nmf).long()
    noise = 0.
    x_aug[:, :, start:end] = noise 
    return x_aug



def rand_displacement(x, mag, warp_obj):
    disp_mag = 100*(mag**2)
    return warp_obj(x, disp_mag)

def magnitude_scale(x, mag):
    BS, C, L = x.shape
    strength = torch.sigmoid(mag)*(-0.5 * (torch.rand(BS).to(x.device)).view(BS,1,1) + 1.25)
    strength = strength.view(BS, 1, 1)
    return x*strength

def square_noise(x, mag):
    
    '''
    mag: recommend [0, 0.5]
    freq: recommend [0.5, 1]
    
    '''
    BS, C, L = x.shape
    duration = L/FREQ
    
    # steps = torch.linspace(0,2*torch.pi*duration*freq, L).view(1,1,-1).to(x.device)
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    
    # noise = mag*scipy_signal.square(steps)
    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)*2  # typical breaths per second for an adult
    # print(frequency)
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    steps = torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()
    
    # print(steps)

    noise = torch.sigmoid(mag)*stdval*scipy_signal.square(steps)
    
    return x + noise
    
    
def white_noise(x, mag):
    BS, C, L = x.shape
    noise = torch.randn(1)*mag*torch.randn(BS, C, L).to(x.device)
    return x + noise


def sine_noise_partial(x, mag):

    """
    Args:
        x: [batchsize, num_channel, sequence_length]
        mag: value between 0 - 1.
        freq (float):
    Returns:
        X_sine_p:
    """
    X_sine_p = x.clone()
    BS, C, L = x.shape

    # duration = signal_length / FREQ
    # steps = np.linspace(0, 2 * np.pi * duration * freq, signal_length)
    # noise = mag*scipy_signal.square(steps)
    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)*2  # typical breaths per second for an adult
    # print(frequency)
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    steps = torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()
    sine_curve = torch.sin(steps)

    # w_ratio = np.random.rand() * mag # Random value between 0 - M
    w_ratio = torch.rand(1)*mag
    width = int(L * w_ratio)
    start = int(torch.rand(1) * (L - width))
    # for w in range(width):
    #     X_sine_p[:, :, start+w] += sine_curve[np.newaxis, np.newaxis, w]
    
    X_sine_p[:,:, start:start+width] += sine_curve[:,:, :width]

    return X_sine_p

def square_noise_partial(x, mag):
    
    X_square_p = x.clone()
    BS, C, L = x.shape

    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)*5 # typical breaths per second for an adult
    # print(frequency)
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    steps = torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()
    square_curve = scipy_signal.square(steps)

    # w_ratio = np.random.rand() * mag # Random value between 0 - M
    w_ratio = torch.rand(1)*mag
    width = int(L * w_ratio)
    start = int(torch.rand(1) * (L - width))
    # for w in range(width):
    #     X_sine_p[:, :, start+w] += sine_curve[np.newaxis, np.newaxis, w]

    X_square_p[:,:, start:start+width] += square_curve[:, :, :width]


    return X_square_p

def white_noise_partial(x, mag):

    X_wnp = x.clone()
    B, C, L = x.shape

    w_ratio = torch.rand(1)*mag
    width = int(L * w_ratio)
    start = int(torch.rand(1) * (L - width))

    white_noise_partial = torch.randn(B, C, width).to(x.device)*mag

    X_wnp[:,:, start:start+width] += white_noise_partial

    return X_wnp

def gaussian_noise_partial(x, mag, noise_mag=2):
    x_gnp = x.clone()
    BS, C, L = x_gnp.shape
    stdval = torch.std(x_gnp, dim=2).view(BS, C, 1).detach()
    w_ratio = torch.rand(1)*mag
    width = int(L * w_ratio)
    start = int(torch.rand(1) * (L - width))
    noise = 0.25*stdval*torch.sigmoid(torch.tensor(noise_mag))*torch.randn(BS, C, width).to(x_gnp.device)
    x_gnp[:,:, start:start+width] += noise
    return x_gnp