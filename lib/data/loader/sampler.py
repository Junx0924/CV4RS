from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import random

class ClassBalancedSampler(torch.utils.data.sampler.Sampler):
    """Sampler that generates class balanced indices .
    For example, choosing batch_size = 50 and nun_samples_per_class = 2
    will result in
    50 indices, which point to 2 samples from 50/2=25 randomly picked classes.

    Yields:
        list: indexes of images
    """    
    def __init__(self, num_images, image_dict,num_samples_per_class=2):
        """Initialize the class .

        Args:
            num_images (int): 
            image_dict (dict): key: class labels, values: the index of images
            num_samples_per_class (int, optional):  Defaults to 2.
        """        
        self.image_dict         = image_dict
        self.samples_per_class  = num_samples_per_class
        num_class = len(image_dict)
        batch_size         = num_samples_per_class * num_class
        self.sampler_length     = num_images//batch_size

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from each classes
            for class_key in self.image_dict.keys():
                index_pool =  self.image_dict[class_key]
                replace = True if len(index_pool) < self.samples_per_class else False
                subset.extend(np.random.choice(index_pool, self.samples_per_class, replace=replace))
            yield list(np.unique(subset))

    def __len__(self):
        return self.sampler_length

