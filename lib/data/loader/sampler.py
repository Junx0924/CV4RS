from __future__ import print_function
from __future__ import division

import logging
import torch
import numpy as np
import random

class ClassBalancedSampler(torch.utils.data.sampler.Sampler):
    """
    Sampler that generates class balanced indices with classes chosen randomly.
    For example, choosing batch_size = 50 and nun_samples_per_class = 2
    will result in
    50 indices, which point to 2 samples from 50/2=25 randomly picked classes.
    """

    def __init__(self, image_dict, image_list,batch_size=50, num_samples_per_class=2):
        self.image_dict         = image_dict
        self.image_list         = image_list

        self.batch_size         = batch_size
        self.samples_per_class  = num_samples_per_class
        self.sampler_length     = len(image_list)//batch_size
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            for _ in range(self.batch_size//self.samples_per_class):
                class_key = random.choice(list(self.image_dict.keys()))
                index_pool = [item[-1] for item in self.image_dict[class_key]]
                subset.extend(np.random.choice(index_pool, self.samples_per_class, replace=False))
            yield subset

    def __len__(self):
        return self.sampler_length

