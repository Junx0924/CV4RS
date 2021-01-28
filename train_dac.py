from __future__ import print_function
from __future__ import division

import os
import matplotlib
import numpy as np
import torch
import time
import json
import random
from tqdm import tqdm
import argparse

import warnings
import parameters as par
from utilities import misc
from utilities import logger
import lib
from lib.clustering import make_clustered_dataloaders

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.putenv("OMP_NUM_THREADS", "8")
pj_base_path= os.path.dirname(os.path.realpath(__file__))
os.environ['TORCH_HOME'] = pj_base_path + "/pretrained_weights"

def load_dac_config(config, args):

    #### Update divide and conquer parameters ###
    config['project'] = 'dac'
    config['recluster']['mod_epoch'] = args.pop('dac_mod_epoch')
    config['finetune_epoch'] = args.pop('dac_finetune_epoch')
    config['nb_clusters'] = args.pop('dac_nb_clusters')
    if 'sub_embed_sizes' not in config.keys():
        config['sub_embed_sizes'] = [config['sz_embedding']//config['nb_clusters']]*config['nb_clusters']
        assert sum(config['sub_embed_sizes']) == config['sz_embedding']
    if config['nb_clusters'] == 1:  config['recluster']['enabled'] = False
    
    return config


def json_dumps(**kwargs):
    # __repr__ may contain `\n`, json replaces it by `\\n` + indent
    return json.dumps(**kwargs).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)



def train_batch(model, criterion, optimizer, config, batch, cluster_id, epoch):
    X = batch[0].to(config['device']) # images
    T = batch[1].to(config['device']) # class labels
    I = batch[2] # image ids

    M = model(X)

    if epoch >= config['finetune_epoch'] * 8 / 19:
        M_sub= M
        pass
    else:
        M = M.split(config['sz_embedding'] // config['nb_clusters'], dim = 1)
        M_sub = M[cluster_id]

    M_sub = torch.nn.functional.normalize(M_sub, p=2, dim=1)
    #loss = criterion[cluster_id](M_sub, T)
    loss = criterion(M_sub, T)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def get_optim(config, model):
    # to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
    #                **config['opt']['backbone'] }]
    to_optim = [{'params': model.parameters_dict['backbone'],
                    **config['opt']['backbone']}]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_dac_config(config, args)
    metrics = {}
    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new=True, log_online=config['log_online'])
   
    # reserve GPU memory for faiss if faiss-gpu used
    faiss_reserver = lib.faissext.MemoryReserver()

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    faiss_reserver.lock(config['backend'])

    
    model = lib.model.make(config)
    _ = model.to(config['device'])

    #model.embedding.weights
    start_epoch = 0
    best_epoch = -1
    best_recall = 0

    # create init and eval dataloaders; init used for creating clustered DLs
    dataloaders = {}
    dataloaders['init'] = lib.data.loader.make(config, model,'init', dset_type = 'train')
    
    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query',is_onehot= True)
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery',is_onehot= True)
    #dl_eval_train = lib.data.loader.make(config, model,'eval', dset_type = 'train',is_onehot = True)

    
    to_optim = get_optim(config, model)
    # temp = [lib.loss.select(config,to_optim,'margin','semihard') for i in range(config['nb_clusters'])]
    # criterion = [item[0] for item in temp]
    # optimizer = [torch.optim.Adam(item[1]) for item in temp]
    criterion, to_optim = lib.loss.select(config,to_optim,'margin','semihard')
    optimizer = torch.optim.Adam(to_optim)

    faiss_reserver.release()
    print("Evaluating initial model...")
    metrics[-1] = {'score': lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, True, config['backend'], LOG, 'Val')}
    best_recall = metrics[-1]['score']['recall'][0]
    dataloaders['train'], C, T, I = make_clustered_dataloaders(model,dataloaders['init'], config, reassign = False)
    faiss_reserver.lock(config['backend'])

    metrics[-1].update({'C': C, 'T': T, 'I': I})

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(start_epoch, config['nb_epochs']):
        is_best = False
        config['epoch'] = e
        metrics[e] = {}
        time_per_epoch_1 = time.time()
        losses_per_epoch = []

        if e >= config['finetune_epoch']:
            if e == config['finetune_epoch'] or e == start_epoch:
                print('Starting to finetune model...')
                config['nb_clusters'] = 1
                print("config['nb_clusters']: {})".format(config['nb_clusters']))
                faiss_reserver.release()
                dataloaders['train'], C, T, I = make_clustered_dataloaders(model, dataloaders['init'], config)
                assert len(dataloaders['train']) == 1
        elif e > 0 and config['recluster']['enabled'] and config['nb_clusters'] > 0:
            if e % config['recluster']['mod_epoch'] == 0:
                print("Reclustering dataloaders...")
                faiss_reserver.release()
                dataloaders['train'], C, T, I = make_clustered_dataloaders(model, dataloaders['init'], config, reassign = True,C_prev = C, I_prev = I, LOG = LOG)
                faiss_reserver.lock(config['backend'])
                metrics[e].update({'C': C, 'T': T, 'I': I})

        # merge dataloaders (created from clusters) into one dataloader
        mdl = lib.data.loader.merge(dataloaders['train'])

        # calculate number of batches for tqdm
        max_len_dataloaders = max([len(dl) for dl in dataloaders['train']])
        num_batches_approx = max_len_dataloaders * len(dataloaders['train'])
        
        _ = model.train()
        for batch, dset in tqdm(mdl,total = num_batches_approx,disable = num_batches_approx < 100,desc = 'Train epoch {}.'.format(e)):
            loss = train_batch(model, criterion, optimizer, config, batch, dset.id, e)
            losses_per_epoch.append(loss)

        time_per_epoch_2 = time.time()
        current_loss = np.mean(losses_per_epoch)
        LOG.progress_saver['Train'].log('epochs', e)
        LOG.progress_saver['Train'].log('Train_loss', current_loss)
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))
        faiss_reserver.release()

        # evaluate
        _ = model.eval()
        tic = time.time()
        metrics[e].update({
            'score': lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'], LOG, 'Val'),
            'loss': {'train': current_loss}
        })

        # evaluate the distance among inter and intra class
        lib.utils.DistanceMeasure(model,config,dl_query,LOG,'Val')

        LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
        LOG.update(all=True)
        print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
        faiss_reserver.lock(config['backend'])

        recall_curr = metrics[e]['score']['recall'][0] # take R@1
        if recall_curr > best_recall:
            best_recall = recall_curr
            best_epoch = e
            is_best = True
            print('Best epoch!')

        model.current_epoch = e
    t2 = time.time()
    print( "Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
    print("Best recall@1 = {} at epoch {}.".format(best_recall, best_epoch))


if __name__ == '__main__':
    main()
    