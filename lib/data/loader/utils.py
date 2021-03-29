
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from math import ceil
import lib.data.set as dataset
from .sampler import ClassBalancedSampler

def make(config, type, subset_indices = None, dset_type = None,include_aux_augmentations = False):
    """A helper function for creating dataset objects .

    Args:
        config (dict): [description]
        type (str): 'init', 'eval' or 'train'.
        subset_indices (list, optional): indices for selecting subset of dataset . Defaults to None.
        dset_type (str, optional): 'train', 'val','test'. Defaults to None.
        include_aux_augmentations (bool, optional): if set true, apply rotation to get augumented image data. Defaults to False.

    Returns:
        torch dataloader 
    """    
    ds_name = config['dataset_selected']
    ds = dataset.select(
        datapath = config['dataset'][ds_name]['root'],
        dset_type = dset_type, # dset_type: train, val,test
        transform = config['transform_parameters'][ds_name],
        is_training = type == 'train',
        include_aux_augmentations = include_aux_augmentations,
        use_npmem = config['use_npmem']
    )
    if type == 'train':
        if 'num_samples_per_class' not in config.keys():
            dl = torch.utils.data.DataLoader(
                ds,
                num_workers= config['dataloader']["num_workers"],
                shuffle= True,
                pin_memory= True,
                batch_size= config['dataloader']['batch_size']
                )
        else:
            ds.set_subset(subset_indices)
            dl = torch.utils.data.DataLoader(
                ds,
                pin_memory= True,
                num_workers= config['dataloader']["num_workers"],
                batch_sampler= ClassBalancedSampler(len(ds),ds.image_dict, config['num_samples_per_class'])
                )
    else:
        # else init or eval loader (shuffle = false)
        dl = torch.utils.data.DataLoader(ds, **config['dataloader'])
    return dl


def make_from_clusters(C, subset_indices, config):
    """Get different dataloaders for different clusters

    Args:
        C (list): cluster labels
        subset_indices (list): original data indexs for each cluster
        config (dict): [description]

    Returns:
        list: a list of dataloders
    """    
    dataloaders = [[None] for c in range(config['nb_clusters'])]
    for c in range(config['nb_clusters']):
        dataloaders[c] = make(
            config = config, 
            type = 'train',
            subset_indices = subset_indices[C == c],
            dset_type = 'train')
        dataloaders[c].dataset.id = c
    return dataloaders


def merge(dls_non_iter):
    """For project divide_and_conquer.
       Merge a list of torch dataloaders to feed in tqmd

    Args:
        dls_non_iter (list): a list of torch dataloaders

    Yields:
        tensor: yield a batch
        int: the index of dataloader
    """    
    nb_batches_per_dl = [len(dl) for dl in dls_non_iter]
    nb_batches = max(nb_batches_per_dl)
    I = range(len(dls_non_iter))
    dls = [iter(dl) for dl in dls_non_iter]

    for j in range(nb_batches):
        for i in I:
            b = next(dls[i], None)
            if b == None:
                # initialize new dataloader in case no batches left
                dls[i] = iter(dls_non_iter[i])
                b = next(dls[i])
            yield b, dls[i]._dataset # changed .dataset to ._dataset (there was no attribute 'dataset')

