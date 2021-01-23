
from __future__ import print_function
from __future__ import division

import random

import torch
import numpy as np
import lib.data.set as dataset
from .sampler import ClassBalancedSampler


def make(config, model, type, subset_indices = None, dset_type = None):
    """
    subset_indices: indices for selecting subset of dataset, for creating
        clustered dataloaders.
    type: 'init', 'eval' or 'train'.
    """
    ds_name = config['dataset_selected']
    batch_size = config['dataloader']["batch_size"]
    num_samples_per_class = config['num_samples_per_class']
    num_workers = config['dataloader']["num_workers"]
    transform = config['transform_parameters'][ds_name]
    root = config['dataset'][ds_name]['root']
    shuffle= config['dataloader']['shuffle']

    ds = dataset.select(
        root = root,
        dset_type = dset_type, # dset_type: train, query, gallery
        transform = transform,
        is_training = type == 'train'
    )
    if type == 'train':
        ds.set_subset(subset_indices)
        train_data_sampler= ClassBalancedSampler(ds.image_dict,ds.image_list, batch_size = batch_size,num_samples_per_class = num_samples_per_class)
        dl = torch.utils.data.DataLoader(
            ds,
            num_workers= num_workers,
            batch_sampler= train_data_sampler
            )
    else:
        # else init or eval loader
        dl = torch.utils.data.DataLoader(ds, **config['dataloader'])
    return dl


def make_from_clusters(C, subset_indices, model, config):
    import numpy as np
    from math import ceil
    dataloaders = [[None] for c in range(config['nb_clusters'])]
    for c in range(config['nb_clusters']):
        dataloaders[c] = make(
            config = config, model = model, type = 'train', subset_indices = subset_indices[C == c],
            dset_type = 'train')
        dataloaders[c].dataset.id = c
    return dataloaders


def merge(dls_non_iter):

    nb_batches_per_dl = [len(dl) for dl in dls_non_iter]
    nb_batches = max(nb_batches_per_dl)
    I = range(len(dls_non_iter))
    length = len(dls_non_iter)
    dls = [iter(dl) for dl in dls_non_iter]

    for j in range(nb_batches):
        for i in I:
            b = next(dls[i], None)
            if b == None:
                # initialize new dataloader in case no batches left
                dls[i] = iter(dls_non_iter[i])
                b = next(dls[i])
            yield b, dls[i]._dataset # changed .dataset to ._dataset (there was no attribute 'dataset')

