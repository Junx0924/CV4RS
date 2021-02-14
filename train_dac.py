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
    config['num_samples_per_class'] = args.pop('num_samples_per_class')
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
    X = batch[0] # images
    T = batch[1] # class labels
    I = batch[2] # image ids

    M = model(X.to(config['device']))
    T_list = lib.utils.classBalancedSamper(T,config['num_samples_per_class'])
    new_I = T_list[:,0]
    M = M[new_I]
    T = torch.tensor(T_list[:,1])
    T = T.to(config['device'])

    if epoch >= config['finetune_epoch'] * 8 / 19:
        M_sub= M
        pass
    else:
        M = M.split(config['sz_embedding'] // config['nb_clusters'], dim = 1)
        M_sub = M[cluster_id]

    M_sub = torch.nn.functional.normalize(M_sub, p=2, dim=1)
    loss = criterion(M_sub, T)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def get_optim(config, model):
    to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
                   **config['opt']['backbone'] }]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_dac_config(config, args)
    metrics = {}
   
    # reserve GPU memory for faiss if faiss-gpu used
    faiss_reserver = lib.faissext.MemoryReserver()

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    faiss_reserver.lock(config['backend'])

    model = lib.multifeature_resnet50.Network(config)
    _  = model.to(config['device'])

    # create init and eval dataloaders; init used for creating clustered DLs
    dataloaders = {}
    dataloaders['init'] = lib.data.loader.make(config, model,'init', dset_type = 'train')
    
    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery')
    # update num_classes
    ds_name = config['dataset_selected']
    num_classes= dl_query.dataset.nb_classes()
    config['dataset'][ds_name]["classes"] = num_classes
    config['dataloader']['batch_size'] = num_classes* config['num_samples_per_class']

    
    to_optim = get_optim(config, model)
    criterion, to_optim = lib.loss.select(config,to_optim,'margin','semihard')
    optimizer = torch.optim.Adam(to_optim)
    # As optimizer, Adam with standard parameters is used.
    optimizer = torch.optim.Adam(to_optim)
    if config['scheduler']=='exp':
        scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler']=='step':
        scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['tau'], gamma=config['gamma'])
    elif config['scheduler']=='none':
        print('No scheduling used!')
    else:
        raise Exception('No scheduling option for input: {}'.format(config['scheduler']))

    faiss_reserver.release()
    dataloaders['train'], C, T, I = make_clustered_dataloaders(model,dataloaders['init'], config, reassign = False)
    faiss_reserver.lock(config['backend'])

    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val', 'Grad']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new=True, log_online=config['log_online'])
    config['checkfolder'] = LOG.config['checkfolder']
    
    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(0, config['nb_epochs']):
        if config['scheduler']!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_last_lr())))
        config['epoch'] = e # for wandb
        time_per_epoch_1 = time.time()
        losses_per_epoch = []

        if e == config['finetune_epoch'] :
            print('Starting to finetune model...')
            config['nb_clusters'] = 1
            print("config['nb_clusters']: {})".format(config['nb_clusters']))
            faiss_reserver.release()
            dataloaders['train'], C, T, I = make_clustered_dataloaders(model, dataloaders['init'], config)
            assert len(dataloaders['train']) == 1
        if e > 0 and config['recluster']['enabled'] and config['nb_clusters'] > 0 and e % config['recluster']['mod_epoch'] == 0:
            print("Reclustering dataloaders...")
            faiss_reserver.release()
            dataloaders['train'], C, T, I = make_clustered_dataloaders(model, dataloaders['init'], config, reassign = True,C_prev = C, I_prev = I, LOG = LOG)
            faiss_reserver.lock(config['backend'])

        # merge dataloaders (created from clusters) into one dataloader
        mdl = lib.data.loader.merge(dataloaders['train'])

        # calculate number of batches for tqdm
        max_len_dataloaders = max([len(dl) for dl in dataloaders['train']])
        num_batches_approx = max_len_dataloaders * len(dataloaders['train'])
        
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
        if e% config['eval_epoch'] ==0:
            _ = model.eval()
            tic = time.time()
            lib.utils.evaluate_standard(model, config, dl_query, False, config['backend'], LOG, 'Val') 
            LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
            print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
            _ = model.train()
        LOG.update(all=True)
        faiss_reserver.lock(config['backend'])
        model.current_epoch = e
        ### Learning Rate Scheduling Step
        if config['scheduler'] != 'none':  scheduler.step()
        
    full_training_time = time.time()-t1
    print('Training Time: {} min.\n'.format(np.round(full_training_time/60,2))) 
    lib.utils.eval_final_model(model,config,dl_query,dl_gallery,config['checkfolder'])   

if __name__ == '__main__':
    main()
    