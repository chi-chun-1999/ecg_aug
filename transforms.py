############ Adapted from  https://github.com/moskomule/dda #############


""" 
Tramsforms

"""
from scipy.special import logit

import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, Bernoulli

from functional import (rand_temporal_warp, baseline_wander, gaussian_noise, rand_crop, spec_aug, rand_displacement, magnitude_scale, square_noise, sine_noise_partial, white_noise, sine_noise_partial, white_noise_partial, gaussian_noise_partial)

# from . import warp_ops
import warp_ops
import numpy as np


class _Tramsforms(nn.Module):
    """Base class of Tramsforms
    
    :param operation:
    :param magnitude:
    :param probability for tramsform
    """
    
    def __init__(self,
                 operation,
                 magnitude,
                 probability,
                 ecg_length=1024
                ):
        super().__init__()

        self.operation = operation
        self.magnitude = torch.tensor(magnitude)

        if probability < 0 or probability > 1:
            raise ValueError("Probability must be between 0 and 1")
        self.probability = probability
    
    def forward(self, input):
        
        # with probability p, apply the transform
        if torch.rand(1).item() < self.probability:
            mag = self.magnitude.to(input.device)
            transformed = self.operation(input, mag)
            return transformed
        else:
            return input
    

class NoOp(_Tramsforms):
    """NoOp
    
    :param operation:
    :param magnitude:
    :param probability for tramsform
    """
    
    def __init__(self,
                 magnitude,
                 probability,
                 ecg_length=1024
                ):
        super().__init__(self._operation, magnitude, probability, ecg_length)
        
    def forward(self, input):
        return input
    
    def _operation(self, x, mag):
        return x

class Standardize(_Tramsforms):
    """Standardize
    
    :param operation:
    :param magnitude:
    :param probability for tramsform
    """
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        super().__init__(self._operation,magnitude, probability, ecg_length)
        
    def forward(self, input):
        
        if torch.rand(1).item() < self.probability:
            transformed = self.operation(input)
            return transformed
        else:
            return input
    
    def _operation(self, x):
        
        if len(x.shape) == 2:
            if type(x) == torch.Tensor:
                stdval = torch.std(x, dim=1).view(-1, 1).detach()
                meanval = torch.mean(x, dim=1).view(-1, 1).detach()
            elif type(x) == np.ndarray:
                stdval = np.std(x, axis=1).reshape(-1, 1)
                meanval = np.mean(x, axis=1).reshape(-1, 1)
            return (x - meanval)/stdval
        elif len(x.shape) == 3:
            BS, C, L = x.shape

            stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
            meanval = torch.mean(x, dim=2).view(BS, C, 1).detach()
            return (x - meanval)/stdval
        
        else:
            raise ValueError("Input shape not supported")

        

class RandTemporalWarp(_Tramsforms):
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        """TemporalWarp
        
        :param operation:
        :param magnitude: recommand [0, 2]
        :param probability for tramsform
        """
        
        super().__init__(rand_temporal_warp, magnitude, probability, ecg_length)
        
        self.warp_obj = warp_ops.RandWarpAug([ecg_length])
    
    def forward(self, input: torch.Tensor):
        
        # input = input.double()
        input = input.type(torch.double)
        
        # print(input.dtype)
        
        # with probability p, apply the transform
        if torch.rand(1).item() < self.probability:
            # print('---- applying temporal warp ----')
            mag = self.magnitude.to(input.device)
            transformed = self.operation(input, mag, self.warp_obj)
            return transformed
        else:
            # print('---- not applying temporal warp ----')

            return input
        

class BaselineWander(_Tramsforms):
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        """BaselineWander
        
        :param operation:
        :param magnitude:
        :param probability for tramsform
        """
        super().__init__(baseline_wander, magnitude, probability, ecg_length)

class GaussianNoise(_Tramsforms):
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        """GaussianNoise
        
        :param operation:
        :param magnitude: recommand [-1, 1]
        :param probability for tramsform
        """
        super().__init__(gaussian_noise, magnitude, probability, ecg_length)

class RandCrop(_Tramsforms):
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        """RandCrop
        
        :param operation:
        :param magnitude: recommand to less than 0.1
        :param probability for tramsform
        """
        super().__init__(rand_crop, magnitude, probability, ecg_length)
        
    def forward(self, input):
        
        if torch.rand(1).item() < self.probability:
            mag = self.magnitude.to(input.device)
            transformed = self.operation(input, mag)
            return transformed
        
        else:
            return input
    
class RandDisplacement(_Tramsforms):
    
    def __init__(self,
                 magnitude:list,
                 probability=0.5,
                ecg_length=1024
                ):
        """RandDisplacement
        
        :param operation:
        :param magnitude: recommand to less than 1
        :param probability for tramsform
        """
        super().__init__(rand_displacement, magnitude, probability, ecg_length)
        
        self.warp_obj = warp_ops.DispAug([ecg_length])
        
    def forward(self, input):
        
        if torch.rand(1).item() < self.probability:
            mag = self.magnitude.to(input.device)
            transformed = self.operation(input, mag, self.warp_obj)
            return transformed
        
        else:
            return input

class SpecAug(_Tramsforms):
    
    def __init__(self,
                 magnitude=0.1,
                 probability=0.5,
                ecg_length=1024
                ):
        """SpecAug
        
        :param operation:
        :param magnitude: less than 0.1
        :param probability for tramsform
        """
        super().__init__(spec_aug, magnitude, probability, ecg_length)

class MagnitudeScale(_Tramsforms):
    
    def __init__(self,
                 magnitude,
                 probability=0.5,
                ecg_length=1024
                ):
        """MagnitudeScale
        
        :param operation:
        :param magnitude:
        :param probability for tramsform
        """
        super().__init__(magnitude_scale, magnitude, probability, ecg_length)

class SquareNoise(_Tramsforms):
    
    def __init__(self,
                 magnitude=1,
                 probability=0.5,
                ecg_length=1024
                ):
        """SquareNoise
        
        :param operation:
        :param magnitude: recommand [0, 0.5]
        :param probability for tramsform
        """
        super().__init__(square_noise, magnitude, probability, ecg_length)


class SineNoisePartial(_Tramsforms):
        
        def __init__(self,
                    magnitude,
                    probability=0.5,
                    ecg_length=1024
                    ):
            """SineNoisePartial
            
            :param operation:
            :param magnitude: recommand [0, 1]
            :param probability for tramsform
            """
            super().__init__(sine_noise_partial, magnitude, probability, ecg_length)

class WhiteNoise(_Tramsforms):
    
    def __init__(self,
                 magnitude=0.015,
                 probability=0.5,
                ecg_length=1024
                ):
        """WhiteNoise
        
        :param operation:
        :param magnitude: recommand [0, 0.5]
        :param probability for tramsform
        """
        super().__init__(white_noise, magnitude, probability, ecg_length)

class SineNoisePartial(_Tramsforms):
    
    def __init__(self,
                 magnitude=0.5,
                 probability=0.5,
                ecg_length=1024
                ):
        """SineNoisePartial
        
        :param operation:
        :param magnitude: recommand [0, 1]
        :param probability for tramsform
        """
        super().__init__(sine_noise_partial, magnitude, probability, ecg_length)

class SquareNoisePartial(_Tramsforms):

    def __init__(self,
                 magnitude=0.5,
                 probability=0.5,
                ecg_length=1024
                ):
        """SquareNoisePartial
        
        :param operation:
        :param magnitude: recommand [0, 0.5]
        :param probability for tramsform
        """
        super().__init__(square_noise, magnitude, probability, ecg_length)

class WhiteNoisePartial(_Tramsforms):
    
    def __init__(self,
                 magnitude=0.2,
                 probability=0.5,
                ecg_length=1024
                ):
        """WhiteNoisePartial
        
        :param operation:
        :param magnitude: recommand [0, 0.5]
        :param probability for tramsform
        """
        super().__init__(white_noise_partial, magnitude, probability, ecg_length)

class GaussianNoisePartial(_Tramsforms):
        
        def __init__(self,
                    magnitude=0.5,
                    probability=0.5,
                    ecg_length=1024
                    ):
            """GaussianNoisePartial
            
            :param operation:
            :param magnitude: recommand [0, 0.5]
            :param probability for tramsform
            """
            super().__init__(gaussian_noise_partial, magnitude, probability, ecg_length)