from __future__ import print_function
from __future__ import division

import torch
import logging
import numpy as np
import sklearn.cluster
import sklearn.metrics.cluster
from . import evaluation
from . import faissext
from . import data
from . import utils


def get_cluster_labels(model, data_loader, use_penultimate, nb_clusters, gpu_ids=None, backend='faiss'):
    """
    Get the clusted labels of the dataset
        Args:
            model: resnet50
            data_loader: torch data loader
            use_penultimate: if false, use the embedding layer to get embeddings
        Return:
            C: new cluster labels
            T_all: labels of original embeddings
            I_all: index of original embeddings
    """
    is_dry_run = (nb_clusters == 1)
    device = torch.device("cuda:0") if gpu_ids ==0 else torch.device("cpu") 
    if not is_dry_run:
        if not use_penultimate:
            print('Using the final layer for clustering')
        X_all, T_all, I_all = utils.predict_batchwise(
            model=model,
            dataloader=data_loader,
            device = device,
            use_penultimate=use_penultimate,
            is_dry_run=is_dry_run
        )
        perm = np.argsort(I_all)
        X_all = X_all[perm]  # embeddings
        I_all = I_all[perm]  # index
        T_all = T_all[perm]  # labels
        if backend == 'torch+sklearn':
            clustering_algorithm = sklearn.cluster.KMeans(n_clusters=nb_clusters)
            C = clustering_algorithm.fit(X_all).labels_
        else:
            C = faissext.do_clustering(
                X_all,
                num_clusters = nb_clusters,
                gpu_ids = gpu_ids,
                niter=100,
                nredo=5,
                verbose=0
            )
    else:
        T_all = np.array(data_loader.dataset.ys)
        I_all = np.array(data_loader.dataset.I)
        C = np.zeros(len(T_all), dtype=int)
    # return the clustered index of embeddings, labels, index
    return C, T_all, I_all


def make_clustered_dataloaders(model, dataloader_init, config,reassign = False, I_prev = None, C_prev = None, LOG = None, initial_C_T_I=None):
    """
    Get different dataloaders for different clusters
    """
    def correct_indices(I):
        return torch.sort(torch.LongTensor(I))[1]

    if initial_C_T_I is None:
        C, T, I = get_cluster_labels(
            model,
            dataloader_init,
            use_penultimate = True,
            nb_clusters = config['nb_clusters'],
            gpu_ids= config['cuda_device'],
            backend=config['backend']
        )
    else:
        C = np.array(initial_C_T_I['C'])
        T = np.array(initial_C_T_I['T'])
        I = np.array(initial_C_T_I['I'])

    if reassign == True:

        # get correct indices for samples by sorting them and return arg sort
        I_correct = correct_indices(I)
        I = I[I_correct]
        T = T[I_correct]
        C = C[I_correct]

        # also use the same indices of sorted samples for previous data
        I_prev_correct = correct_indices(I_prev)
        I_prev = I_prev[I_prev_correct]
        C_prev = C_prev[I_prev_correct]

        print('Reassigning clusters...')
        nmi =  sklearn.metrics.cluster.normalized_mutual_info_score(C[I],C_prev[I_prev])
        print("Calculating NMI for consecutive cluster assignments:{}".format(nmi))
        LOG.progress_saver['Train'].log('nmi',nmi,'reclustering')

        # assign s.t. least costs w.r.t. L1 norm
        C, costs = data.loader.reassign_clusters(C_prev = C_prev,C_curr = C, I_prev = I_prev, I_curr = I)
        # print('Costs before reassignment: {}'.format(costs))
        # _, costs = data.loader.reassign_clusters(C_prev = C_prev,C_curr = C, I_prev = I_prev, I_curr = I)
        # after printing out the costs now, the trace of matrix should have lower numbers than other entries in matrix
        # print('Costs after reassignment: {}'.format(costs))

    #  remove labels s.t. minimum 2 samples per class per cluster
    for c in range(config['nb_clusters']):
        for t in np.unique(T[C == c]):
            if (T[C == c] == t).sum().item() == 1:
                # assign to cluster -1 if only one sample from class
                C[(T == t) & (C == c)] = -1

    dls = data.loader.make_from_clusters(C = C, subset_indices = I, config = config)

    return dls, C, T, I
