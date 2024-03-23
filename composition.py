import random

import torch

class BaseCompose:
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class Compose(BaseCompose):
    def __init__(self, transforms, p=1.0):
        super().__init__(transforms, p)
    
    def __call__(self, data):

        apply = random.random() < self.p

        if apply:
            for t in self.transforms:
                data = t(data)
        return data


class OneOf(BaseCompose):
    def __init__(self, transforms, p=1.0):
        super().__init__(transforms, p)
    
    def __call__(self, data):

        apply = random.random() < self.p

        if apply:
            t = random.choice(self.transforms)
            data = t(data)
        return data

class SomeOf(BaseCompose):
    def __init__(self, transforms, n=1, p=1.0):
        super().__init__(transforms, p)
        self.n = n
        self.transforms_index = []
    
    def randomize_paremeter(self):
        
        apply = random.random() < self.p
        if apply:
            all_transforms_index = list(range(len(self.transforms)))
            self.transforms_index = random.sample(all_transforms_index, self.n)
        else:
            self.transforms_index = []

        return self.transforms_index
    
    def __call__(self, data):
            
            self.randomize_paremeter()
            # print(self.transforms_index)

            for i in self.transforms_index:
                data = self.transforms[i](data)
            return data

