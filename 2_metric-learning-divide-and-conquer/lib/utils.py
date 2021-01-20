from __future__ import print_function
from __future__ import division

from . import evaluation
from . import similarity
import numpy as np
import torch
from tqdm import tqdm


def predict_batchwise(model, dataloader, use_penultimate, is_dry_run=False):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # use tqdm when the dataset is large (SOProducts)
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc='predict', disable=not is_verbose):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    if not is_dry_run:
                        # move images to device of model (approximate device)
                        J = J.to(list(model.parameters())[0].device)
                        # predict model output for image
                        J = model(J, use_penultimate).data.cpu().numpy()
                        # take only subset of resulting embedding w.r.t dataset
                    else:
                        # just a placeholder not to break existing code
                        J = np.array([-1])
                for j in J:
                    A[i].append(np.asarray(j))
        result = [np.stack(A[i]) for i in range(len(A))]
    model.train()
    model.train(model_is_training) # revert to previous training state
    if is_dry_run:
        # do not return features if is_dry_run
        return [None, *result[1:]]
    else:
        return result


def evaluate(model,  config, dl_query, dl_gallery, use_penultimate, backend, LOG, log_key = 'Val',with_nmi = False):
    K = [1, 2, 4, 8]
    # calculate embeddings with model and get targets
    X_query, T_query, _ = predict_batchwise(model, dl_query, use_penultimate)
    X_gallery, T_gallery, _ = predict_batchwise(model, dl_gallery, use_penultimate)

    nb_classes = dl_query.dataset.nb_classes()
    assert nb_classes == len(set(T_query))

    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat(
        [torch.from_numpy(np.array(T_query,dtype = int)), torch.from_numpy(np.array(T_gallery, dtype = int))])
    X_eval = torch.cat(
        [torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]

    D = torch.from_numpy(D)
    # get top k labels with smallest (`largest = False`) distance
    Y = T_gallery[D.topk(k = max(K), dim = 1, largest = False)[1]]

    flag_checkpoint = False
    history_recall1 = 0
    if "recall" not in LOG.progress_saver[log_key].groups.keys():
        flag_checkpoint = True
    else: 
        history_recall1 = np.max(LOG.progress_saver[log_key].groups['recall']["recall@1"]['content'])

    scores = {}
    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, Y, k)
        recall.append(r_at_k)
        LOG.progress_saver[log_key].log("recall@"+str(k),r_at_k,group='recall')
        print("recall@{} : {:.3f}".format(k, 100 * r_at_k))
        if k==1 and r_at_k > history_recall1:
            flag_checkpoint = True
        
    scores['recall'] = recall

    ### save checkpoint #####
    if  flag_checkpoint:
        savepath = LOG.config['checkfolder']+'/checkpoint_{}.pth.tar'.format("recall@1")
        aux_store = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.save({'state_dict':model.state_dict(), 'opt':config, 'progress': LOG.progress_saver, 'aux':aux_store}, savepath)

    if with_nmi:
        # calculate NMI with kmeans clustering
        nmi = evaluation.calc_normalized_mutual_information(
            T_eval.numpy(),
            similarity.cluster_by_kmeans(
                X_eval.numpy(), nb_classes, backend=backend
            )
        )
        LOG.progress_saver[log_key].log("nmi",nmi)
        print("NMI: {:.3f}".format(nmi * 100))
        scores['nmi'] = nmi

    return scores