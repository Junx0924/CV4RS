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

def load_diva_config(config,args):
    #### Update Diva parameter  ###########
    config['project'] = 'diva'
    config['diva_features'] = args.pop('diva_features')
    if 'sub_embed_sizes' not in config.keys():
        num_feature =  len(config['diva_features'])
        config['sub_embed_sizes'] =[config['sz_embedding'] //num_feature]*num_feature
        config['dive_feature_size'] = { feature:int(size) for feature, size in zip(config['diva_features'],config['sub_embed_sizes'])}
        assert sum(config['sub_embed_sizes']) == config['sz_embedding']

    config['hidden_adversarial_size'] = args.pop('diva_hidden_adversarial_size')

    config['criterion']={'selfsimilarity':{'weight':args.pop('diva_alpha_ssl'),'loss':'fast_moco','batchminner':None},
                         'intra':         {'weight':args.pop('diva_alpha_intra'),'loss':'margin', 'batchminner':'intra_random'},
                         'shared':        {'weight':args.pop('diva_alpha_shared'),'loss': 'margin','batchminner':'random_distance'},
                         'discriminative':{'weight':1,'loss': 'margin','batchminner':'distance'},
                         'decorrelation':    {'weight':1,'loss':'adversarial','batchminner':None}
                        }
    
    # config the decorrelation between features
    if 'shared' in config['diva_features'] and len(config['diva_features'])==4:
        decorrelation =  ['selfsimilarity-discriminative', 'shared-discriminative', 'intra-discriminative']
        weights = args.pop('diva_adversarial_weight')

        assert len(weights) == len(decorrelation)
        config['decorrelation']= {}
        for item, weight in zip(decorrelation, weights) :
            feature1, feature2 = item.split('-')
            dim1, dim2 = config['dive_feature_size'][feature1], config['dive_feature_size'][feature2]
            config['decorrelation'][item] = {'dim':str(dim1)+'-'+str(dim2), 'weight':weight}
    
    if len(config['diva_features']) >1:
        config['evaluation_weight'] = args.pop('diva_evaluation_weight')
        assert len(config['diva_features']) == len(config['evaluation_weight'])

    config['include_aux_augmentations'] = True if 'selfsimilarity' in config['diva_features'] else False
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

def train_batch(model,criterion_dict, optimizer, config, batch,LOG=None, log_key ='',selfsim_model=None):
    if len(batch) ==4:
        X = batch[0] # images
        T = batch[1] # class labels
        I = batch[2] # image ids
        X_aux = batch[3] # augumented images
    else:
        X = batch[0] # images
        T = batch[1]# class labels
        I = batch[2] # image ids

    fvecs = model(X.to(config['device']))
    T_list = lib.utils.classBalancedSamper(T,config['num_samples_per_class'])
    new_I = T_list[:,0]
    T = T_list[:,1]
    fvecs = fvecs[new_I]
    if len(batch) ==4:
        X_aux = X_aux[new_I]
    
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
    total_loss =0.0
    for key in criterion_dict.keys():
        if key =='selfsimilarity':
            temp_loss = criterion_dict[key](features[key],selfsim_key_features)
        elif key =='decorrelation':
            temp_loss = criterion_dict[key](features)
        else:
            temp_loss = criterion_dict[key](features[key],T)
        
        losses[key] = temp_loss.item()
        weight = config['criterion'][key]['weight']
        total_loss += weight*temp_loss
    losses['Train'] = total_loss.item()
        
    optimizer.zero_grad()
    total_loss.backward()

    # log the gradient of each layer
    #lib.utils.GradientMeasure(model,LOG,log_key)
    ### Update network weights!
    optimizer.step()

    if 'selfsimilarity' in criterion_dict:
        ### Update Key Network
        for model_par, key_model_par in zip(model.parameters(), selfsim_model.parameters()):
            momentum = criterion_dict['selfsimilarity'].momentum
            key_model_par.data.copy_(key_model_par.data*momentum + model_par.data*(1-momentum))
        ###
        criterion_dict['selfsimilarity'].update_memory_queue(selfsim_key_features)

    return losses


def get_criterion(config, to_optim):
    criterion_dict ={}
    for feature in config['diva_features']:
        loss_name = config['criterion'][feature]['loss']
        batchminner = config['criterion'][feature]['batchminner']
        criterion_dict[feature], to_optim = lib.loss.select(config,to_optim,loss_name,batchminner)
    if 'decorrelation' in config.keys():
        loss_name = config['criterion']['decorrelation']['loss']
        batchminner = config['criterion']['decorrelation']['batchminner']
        criterion_dict['decorrelation'], to_optim = lib.loss.select(config,to_optim,loss_name,batchminner)

    return criterion_dict,to_optim

def get_optim(config, model):
    to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
                   **config['opt']['backbone'] }]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_diva_config(config, args)
    metrics = {}
    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val', 'Grad']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new=True, log_online=config['log_online'])
   
    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # get model
    model = lib.multifeature_resnet50.Network(config)
    _  = model.to(config['device'])
    if 'selfsimilarity' in config['diva_features']:
        selfsim_model = lib.multifeature_resnet50.Network(config)
        _  = selfsim_model.to(config['device'])

    # create train dataset
    flag_aux =config['include_aux_augmentations']
    dl_train = lib.data.loader.make(config, model,'train', dset_type = 'train',include_aux_augmentations=flag_aux)
    # update num_classes
    ds_name = config['dataset_selected']
    num_classes= dl_train.dataset.nb_classes()
    config['dataset'][ds_name]["classes"] = num_classes
    config['dataloader']['batch_size'] = num_classes* config['num_samples_per_class']
    
    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery')
    print("Evaluate initial model") 
    lib.utils.evaluate_standard(model, config, dl_query, False, config['backend'], LOG, 'Val',is_init=True) 
    
    # define loss function for each feature
    to_optim = get_optim(config, model)
    criterion_dict, to_optim = get_criterion(config, to_optim)
    
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

    if 'selfsimilarity' in criterion_dict.keys():
        dl_init = lib.data.loader.make(config, model,'init', dset_type = 'train')
        criterion_dict['selfsimilarity'].create_memory_queue(selfsim_model, dl_init, config['device'], opt_key='selfsimilarity') 

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(0, config['nb_epochs']):
        config['epoch'] = e # for wandb
        if config['scheduler']!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_last_lr())))
       
        time_per_epoch_1 = time.time()
        loss_collect ={}
        _ = model.train()
        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(e)):
            if 'selfsimilarity' in criterion_dict.keys():
                losses = train_batch(model, criterion_dict,optimizer, config, batch, LOG, 'Grad', selfsim_model)
            else:
                losses = train_batch(model, criterion_dict,optimizer, config, batch, LOG, 'Grad')
            for key in losses:
                if key not in loss_collect.keys():
                    loss_collect[key] = []
                loss_collect[key].append(losses[key])

        time_per_epoch_2 = time.time()
        current_loss = np.mean(loss_collect['Train'])
        LOG.progress_saver['Train'].log('epochs', e)
        for key in loss_collect.keys():
            LOG.progress_saver['Train'].log(key +'_loss',np.mean(loss_collect[key]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))

        # evaluate
        if e % config['eval_epoch'] ==0:
            _ = model.eval()
            tic = time.time()
            lib.utils.evaluate_standard(model, config, dl_query, False, config['backend'], LOG, 'Val') 
            LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
            _ = model.train()
        LOG.update(all=True)

        ### Learning Rate Scheduling Step
        if config['scheduler'] != 'none':  scheduler.step()

    ### CREATE A SUMMARY TEXT FILE
    summary_text = ''
    full_training_time = time.time()-t1
    summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))
    summary_text += '---------------\n'+ config['project'] + ' Retrieve performance\n'
    # load checkpoint file
    # check retrieval performance
    checkpoint = torch.load(LOG.config['checkfolder']+"/checkpoint_recall@1.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    retrieve_score = lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend']) 
    for key in retrieve_score: 
        summary_text += str(key)+": " + str(retrieve_score[key])+ '\n'
    with open(LOG.config['checkfolder']+'/training_summary.txt','w') as summary_file:
        summary_file.write(summary_text)
    # apply tsne to embeddings from query dataset
    X, T, _ = lib.utils.predict_batchwise(model, dl_query, config['device'], False, desc='Extraction Eval Features') 
    lib.utils.apply_tsne(X,T, dl_query.dataset.conversion, LOG.config['checkfolder']+'/'+config['project']+'_tsne.png')     

if __name__ == '__main__':
    main()
    