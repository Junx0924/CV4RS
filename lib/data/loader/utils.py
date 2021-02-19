
from __future__ import print_function
from __future__ import division

import random
import torch
import numpy as np
import lib.data.set as dataset
from .sampler import ClassBalancedSampler

def make(config, model, type, subset_indices = None, dset_type = None,include_aux_augmentations = False):
    """
    Args:
        subset_indices: indices for selecting subset of dataset,
                        for creating clustered dataloaders.
        type: 'init', 'eval' or 'train'.
    """
    ds_name = config['dataset_selected']
    ds = dataset.select(
        datapath = config['dataset'][ds_name]['root'],
        dset_type = dset_type, # dset_type: train, query, gallery
        transform = config['transform_parameters'][ds_name],
        is_training = type == 'train',
        include_aux_augmentations = include_aux_augmentations,
        use_hdf5 = config['use_hdf5']
    )
    if type == 'train':
        if config['project']=='sndl':
            dl = torch.utils.data.DataLoader(
                ds,
                num_workers= config['dataloader']["num_workers"],
                shuffle= True,
                pin_memory= True,
                batch_size= config['dataloader']['batch_size']
                )
        else:
            ds.set_subset(subset_indices)
            train_data_sampler= ClassBalancedSampler(len(ds),ds.image_dict, config['num_samples_per_class'])
            dl = torch.utils.data.DataLoader(
                ds,
                num_workers= config['dataloader']["num_workers"],
                batch_sampler= train_data_sampler
                )
    else:
        # else init or eval loader (shuffle = false)
        dl = torch.utils.data.DataLoader(ds, **config['dataloader'])
    return dl


def make_from_clusters(C, subset_indices, model, config):
    """
    For project divide_and_conquer
    Get different dataloaders for different clusters
        Args:
            C: cluster labels
            subset_indices: original data indexs for each cluster
    """
    import numpy as np
    from math import ceil
    dataloaders = [[None] for c in range(config['nb_clusters'])]
    for c in range(config['nb_clusters']):
        dataloaders[c] = make(
            config = config, 
            model = model,
            type = 'train',
            subset_indices = subset_indices[C == c],
            dset_type = 'train')
        dataloaders[c].dataset.id = c
    return dataloaders


def merge(dls_non_iter):
    """
    For project divide_and_conquer
    Merge a list of torch dataloaders to feed in tqmd
    dls_non_iter: a list of torch dataloaders
    """
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

