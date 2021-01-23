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

def load_config(config_name):
    ################### INPUT ARGUMENTS ###################
    parser = argparse.ArgumentParser()
    parser = par.basic_training_parameters(parser)
    parser = par.setup_parameters(parser)
    parser = par.wandb_parameters(parser)
    parser = par.BIER(parser)
    ##### Read in parameters
    args = vars(parser.parse_args())

    ##### Read config.json
    config_name = pj_base_path +'/config.json'
    with open(config_name, 'r') as f: config = json.load(f)
   
    #### Update config.json from INPUT ARGUMENTS ###########
    config['pj_base_path'] = pj_base_path
    config['pretrained_weights_file'] = pj_base_path + '/' + config['pretrained_weights_file']
    config['dataloader']['batch_size'] = args.pop('batch_size')
    config['dataloader']['num_workers'] = args.pop('num_workers')
    config['opt']['backbone']['lr'] = args.pop('backbone_lr')
    config['opt']['backbone']['weight_decay'] = args.pop('backbone_wd')
    config['opt']['embedding']['lr'] =args.pop('embedding_lr')
    config['opt']['embedding']['weight_decay'] =args.pop('embedding_wd')
    dataset_name =  args.pop('dataset_name')
    config['dataset_selected'] = dataset_name
    config['dataset'][dataset_name]['root'] = args.pop('source_path') + '/' + dataset_name
    config['random_seed'] = args.pop('random_seed')
    config['log_online'] = args.pop('log_online')
    config['frozen'] = args.pop('frozen')
    config['log']['save_path'] = args.pop('save_path')
    config['log']['save_name'] = dataset_name +'_s{}'.format(config['random_seed'])
   
   #### Bier parameter 
    config['lambda_weight'] = args.pop('lambda_weight')
    config['sub_embed_sizes'] = args.pop('sub_embed_sizes')
    config['hidden_adversarial_size'] = args.pop('hidden_adversarial_size')
    config['shrinkage'] = args.pop('shrinkage')
    config['lambda_div']= args.pop('lambda_div')
    config['eta_style'] = args.pop('eta_style')

    if config['log_online']:
        config['wandb']['wandb_key'] = args.pop('wandb_key')
        config['wandb']['project'] =args.pop('project')
        config['wandb']['group'] =args.pop('group')
        # update save_name
        config['log']['save_name'] =  config['wandb']['group']+'_s{}'.format(config['random_seed'])
        import wandb
        os.environ['WANDB_API_KEY'] = config['wandb']['wandb_key']
        os.environ["WANDB_MODE"] = "dryrun" # for wandb logging on HPC
        _ = os.system('wandb login --relogin {}'.format(config['wandb']['wandb_key']))
        wandb.init(project=config['wandb']['project'], group=config['wandb']['group'], name=config['log']['save_name'], dir=config['log']['save_path'])
        wandb.config.update(config)
    # update save_path
    config['log']['save_path'] = config['log']['save_path']+ '/' + dataset_name
        
    if config['nb_clusters'] == 1:  config['recluster']['enabled'] = False
    
    for k in args:
        if k in config:
            config[k] = args[k]

    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                if type(config[k]) is str:
                    # if python types, then evaluate str expressions
                    if config[k][:5] in ['range', 'float']:
                        config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)   
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


def evaluate(model,LOG, backend='faiss-gpu', config = None, log_key= 'Val'):
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery')
    score = lib.utils.evaluate(model,  config, dl_query, dl_gallery, False, backend, LOG, log_key)
    return score


def train_batch(model, opt, config, batch, LOG, log_key='Train'):
    if torch.cuda.is_available():
        X = batch[0].cuda(non_blocking=True) # images
        T = batch[1].cuda(non_blocking=True) # class labels
    else:
        X = batch[0]
        T = batch[1]
    I = batch[2] # image ids

    opt.zero_grad()
    M = model(X)
    loss = 0.0
    binominal_loss = get_criterion(config, 'boosted_binominal')(M,T)
    LOG.progress_saver[log_key].log('boosted_binominal_loss',binominal_loss.item())
    loss = loss + binominal_loss
    if config['lambda_div'] > 0.0:
        adv_loss = get_criterion(config,"adversarial")(M)
        LOG.progress_saver[log_key].log('adversarial',adv_loss.item())
        weight_loss = 0.0
        embedding_weights = model.embedding.weight.data
        for W in embedding_weights:
            weight_loss += torch.mean(torch.square(torch.sum(W * W, axis=1) - 1))
        LOG.progress_saver[log_key].log('weight_loss',weight_loss.item())
        loss = loss + (adv_loss + weight_loss*config['lambda_weight'])*config['lambda_div']
    
    loss.backward()
    opt.step()
    return loss.item()


def get_criterion(config, loss_name):
    sub_embed_sizes = config['sub_embed_sizes'] 
    print('Create {} loss'.format(loss_name))
    if "adversarial" in loss_name:
        criterion = lib.loss.adversarial(sub_embed_sizes,config['hidden_adversarial_size'],config['lambda_weight'])
    if 'binominal' in loss_name:
        criterion = lib.loss.boosted_binominal(sub_embed_sizes,config['shrinkage'],config['lambda_div'])
    if torch.cuda.is_available(): criterion = criterion.cuda()
    return criterion


def get_optimizer(config, model):
    opt = torch.optim.Adam([
        {
            'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
            **config['opt']['backbone']
        },
        {
            'params': model.parameters_dict['embedding'],
            **config['opt']['embedding']
        }
    ])
    return opt


def main():
    config_name = pj_base_path + '/config.json'
    config = load_config(config_name)
    metrics = {}
    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new=True, log_online=config['log_online'])
   
    # reserve GPU memory for faiss if faiss-gpu used
    faiss_reserver = lib.faissext.MemoryReserver()

    if torch.cuda.is_available():
        torch.cuda.set_device(config['cuda_device'])

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    faiss_reserver.lock(config['backend'])

    if torch.cuda.is_available():
        model = lib.model.make(config).cuda()
    else:
        model = lib.model.make(config)

    #model.embedding.weights
    start_epoch = 0
    best_epoch = -1
    best_recall = 0

    # create init and eval dataloaders; init used for creating clustered DLs
    dataloaders = {}
    dataloaders['train'] = lib.data.loader.make(config, model,'train', dset_type = 'train')
        
    opt = get_optimizer(config, model)

    faiss_reserver.release()
    # print("Evaluating initial model...")
    # metrics[-1] = {'score': evaluate(model, LOG, backend = config['backend'],config = config, log_key ='Val')}
    # best_recall = metrics[-1]['score']['recall'][0]

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(start_epoch, config['nb_epochs']):
        is_best = False
        config['epoch'] = e
        metrics[e] = {}
        time_per_epoch_1 = time.time()
        losses_per_epoch = []

        for batch in tqdm(dataloaders['train'],desc = 'Train epoch {}.'.format(e)):
            loss = train_batch(model, opt, config, batch, LOG, log_key='Train')
            losses_per_epoch.append(loss)

        time_per_epoch_2 = time.time()
        current_loss = np.mean(losses_per_epoch)
        LOG.progress_saver['Train'].log('epochs', e)
        LOG.progress_saver['Train'].log('Train_loss', current_loss)
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))
        faiss_reserver.release()

        # evaluate
        tic = time.time()
        metrics[e].update({
            'score': evaluate(model,LOG,backend=config['backend'],config = config,log_key='Val'),
            'loss': {'train': current_loss}
        })
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
    