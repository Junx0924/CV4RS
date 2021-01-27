from __future__ import print_function
from __future__ import division

import os
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import json
import random
from tqdm import tqdm
import argparse
import itertools

import warnings
import parameters as par
from utilities import misc
from utilities import logger
import lib

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
    parser = par.diva(parser)
    ##### Read in parameters
    args = vars(parser.parse_args())

    ##### Read config.json
    config_name = pj_base_path +'/config.json'
    with open(config_name, 'r') as f: config = json.load(f)
   
    #### Update config.json from INPUT ARGUMENTS ###########
    config['sz_embedding'] = args.pop('sz_embedding')
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
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Update wandb config
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

    #### Update Diva parameter 
    config['diva_features'] = args.pop('diva_features')
    if 'sub_embed_sizes' not in config.keys():
        num_feature =  len(config['diva_features'])
        config['sub_embed_sizes'] =[config['sz_embedding'] //num_feature]*num_feature
        assert sum(config['sub_embed_sizes']) == config['sz_embedding']

    config['hidden_adversarial_size'] = args.pop('hidden_adversarial_size')

    config['loss_weight']={'selfsimilarity':args.pop('diva_alpha_ssl'),
                    'intra':args.pop('diva_alpha_intra'),
                    'shared':args.pop('diva_alpha_shared'),
                    'discriminative': 1,
                    'separation':1}
    if len(config['diva_features']) >1:
        config['evaluation_weight'] = args.pop('evaluation_weight')
        assert len(config['diva_features']) == len(config['evaluation_weight'])

    if 'selfsimilarity' in config['diva_features']:
       config['include_aux_augmentations'] = True 
    else:
        config['include_aux_augmentations'] = False 

    if 'shared' in config['diva_features'] and len(config['diva_features'])==4:
        config['diva_adversarial_weight'] =args.pop('diva_adversarial_weight')
        config['diva_decorrelations'] = ['selfsimilarity-discriminative', 'shared-discriminative', 'intra-discriminative']
        assert len(config['diva_decorrelations']) == len(config['diva_adversarial_weight'])
    
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



def train_batch(model,criterion_dict, opt, config, batch,selfsim_model=None):
    if len(batch) ==5:
        X = batch[0] # images
        T = batch[1] # class labels
        I = batch[2] # image ids
        X_aux = batch[3] # augumented images
        T_aux = batch[4] # psudo labels
    else:
        X = batch[0] # images
        T = batch[1]# class labels
        I = batch[2] # image ids
    
    fvecs = model(X.to(config['device']))
    sub_dim = config['sz_embedding'] // len(config['diva_features'])
    fvecs = fvecs.split(sub_dim, dim =1)
    # l2 normalize feature
    features = {}
    for name, fvec in zip(config['diva_features'],fvecs):
        features[name] = F.normalize(fvec,p =2,dim =1)
    
    if 'selfsimilarity' in criterion_dict.keys():
        with torch.no_grad():
            ### Use shuffleBN to avoid information bleeding making samples interdependent.
            forward_shuffle, backward_reorder = criterion_dict['selfsimilarity'].shuffleBN(len(features['selfsimilarity']))
            temp              = selfsim_model(X_aux[forward_shuffle].to(config['device']))
            temp = temp.split(sub_dim, dim =1)
            ind = config['diva_features'].index('selfsimilarity')
            selfsim_key_features = temp[ind][backward_reorder]
    
    # record different losses
    losses ={} 
    loss_weight = config['loss_weight']
    total_loss =0.0
    for key in criterion_dict.keys():
        if key =='selfsimilarity':
            temp_loss = criterion_dict[key](features[key],selfsim_key_features)
        elif key =='separation':
            temp_loss =0.0
            for direction, weight in zip(config['diva_decorrelations'], config['diva_adversarial_weight']):
                source, target = direction.split('-')
                temp_loss += weight* criterion_dict[key](features[source],features[target])
        else:
            temp_loss =  criterion_dict[key](features[key],T)
        
        losses[key] = temp_loss.item()
        total_loss += loss_weight[key]*temp_loss
    losses['total'] = total_loss.item()
        
    opt.zero_grad()
    total_loss.backward()
    ### Update network weights!
    opt.step()

    if 'selfsimilarity' in criterion_dict:
        ### Update Key Network
        for model_par, key_model_par in zip(model.parameters(), selfsim_model.parameters()):
            momentum = criterion_dict['selfsimilarity'].momentum
            key_model_par.data.copy_(key_model_par.data*momentum + model_par.data*(1-momentum))
        ###
        criterion_dict['selfsimilarity'].update_memory_queue(selfsim_key_features)

    return losses


def get_criterion(config):
    criterion_dict ={}
    for name in config['diva_features']:
        criterion_dict[name] = lib.loss.select(config,name)
    if len(config['diva_features']) >1:
        criterion_dict['separation'] =  lib.loss.select(config,'separation')
    return criterion_dict


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

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    faiss_reserver.lock(config['backend'])

    # get model
    # model = lib.model.make(config)
    # _  = model.to(config['device'])
    # if 'selfsimilarity' in config['diva_features']:
    #     selfsim_model = lib.model.make(config)
    #     _  = selfsim_model.to(config['device'])
    model = lib.multifeature_resnet50.Network(config)
    _  = model.to(config['device'])
    if 'selfsimilarity' in config['diva_features']:
        selfsim_model = lib.multifeature_resnet50.Network(config)
        _  = selfsim_model.to(config['device'])

    
    start_epoch = 0
    best_epoch = -1
    best_recall = 0
 
    # create train dataset
    dataloaders = {}
    flag_aux =config['include_aux_augmentations']
    dataloaders['train'] = lib.data.loader.make(config, model,'train', dset_type = 'train',include_aux_augmentations=flag_aux)
    
    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query',is_onehot= True)
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery',is_onehot= True)

    # define loss function for each feature
    criterion_dict = get_criterion(config)
    if 'selfsimilarity' in criterion_dict.keys():
        dataloaders['init'] = lib.data.loader.make(config, model,'init', dset_type = 'train',is_onehot=True)
        criterion_dict['selfsimilarity'].create_memory_queue(selfsim_model, dataloaders['init'], config['device'], opt_key='selfsimilarity') 

    opt = get_optimizer(config, model)

    faiss_reserver.release()
    print("Evaluating initial model...")
    metrics[-1] = {'score': lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, True, config['backend'], LOG, init_eval= True)}
    best_recall = metrics[-1]['score']['recall'][0]

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(start_epoch, config['nb_epochs']):
        is_best = False
        config['epoch'] = e
        metrics[e] = {}
        time_per_epoch_1 = time.time()

        loss_collect ={}
        for batch in tqdm(dataloaders['train'],desc = 'Train epoch {}.'.format(e)):
            if 'selfsimilarity' in criterion_dict.keys():
                losses = train_batch(model, criterion_dict,opt, config, batch,selfsim_model=selfsim_model)
            else:
                losses = train_batch(model, criterion_dict,opt, config, batch)
            for key in losses:
                if key not in loss_collect.keys():
                    loss_collect[key] = []
                loss_collect[key].append(losses[key])

        time_per_epoch_2 = time.time()
        current_loss = np.mean(loss_collect['total'])
        LOG.progress_saver['Train'].log('epochs', e)
        for key in loss_collect.keys():
            LOG.progress_saver['Train'].log(key +'_loss',np.mean(loss_collect[key]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))
        faiss_reserver.release()

        # evaluate
        tic = time.time()
        metrics[e].update({
            'score': lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'], LOG),
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
    