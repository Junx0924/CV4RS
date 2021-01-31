from __future__ import print_function
from __future__ import division

import os
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
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
from lib.clustering import make_clustered_dataloaders
from torch.autograd import Variable

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.putenv("OMP_NUM_THREADS", "8")
pj_base_path= os.path.dirname(os.path.realpath(__file__))
os.environ['TORCH_HOME'] = pj_base_path + "/pretrained_weights"

def load_bier_config(config, args):
    
    #### UPdate Bier parameter 
    config['project'] = 'bier'
    config['lambda_weight'] = args.pop('bier_lambda_weight')
    config['lambda_div']= args.pop('bier_lambda_div')
    config['sub_embed_sizes'] = args.pop('bier_sub_embed_sizes')
    assert sum(config['sub_embed_sizes']) == config['sz_embedding']
    
    # config the decorrelation between features
    decorrelation = list(itertools.combinations(config['sub_embed_sizes'],2))
    config['decorrelation'] = {}
    for item in decorrelation:
        direction_name = str(item[0])+ '-' + str(item[1])
        config['decorrelation'][direction_name] = {'dim':str(item[0])+ '-' + str(item[1]),'weight':config['lambda_weight']}

    config['hidden_adversarial_size'] = args.pop('bier_hidden_adversarial_size')
    config['hard_mining'] = args.pop('bier_hard_mining')
    config['alpha'] = args.pop('bier_alpha')
    config['beta'] = args.pop('bier_beta')
    config['margin'] = args.pop('bier_margin')
    
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



def train_batch(model, criterion_dict,opt, config, batch,LOG=None, log_key =''):
    
    X = batch[0] # images
    T = batch[1] # class labels
    I = batch[2] # image ids

    feature = model(X.to(config['device']))
    
    if len(T.size())==2: 
        T_list = [ np.where(t==1)[0] for t in T] 
        T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
        feature = feature[T_list[:,0]]
        T = torch.tensor(T_list[:,1])

    n = len(feature)
    total_loss =0.0
    bin_loss =0.0
    # l2 normalize feature
    normed_fvecs = {}
    sub_dim = config['sub_embed_sizes']
    for i in range(len(sub_dim)):
        start = int(sum(sub_dim[:i]))
        stop = int(start + sub_dim[i])
        fvecs = F.normalize(feature[:, start:stop],p =2,dim =1)
        normed_fvecs[str(sub_dim[i])] = fvecs
    
    # create similarity matrix for each sublearner
    sim_mats =[torch.zeros(n,n).to(config['device'])]
    for fvecs in normed_fvecs.values():
        temp_mat = torch.matmul(fvecs, fvecs.t())
        sim_mats.append(temp_mat)
    
    # init boosting_weights for each label pair
    boosting_weights = torch.ones(n*n).to(config['device'])
    # Pairwise labels
    T = T.to(config['device'])
    a = torch.cat(n*[torch.unsqueeze(T, 0)])
    b = torch.transpose(a, 0, 1)
    pairs = torch.flatten(a==b)*1.0
    W = torch.flatten(1.0 - torch.eye(n)).to(config['device'])
    # initial weight for each label pair (not include itself)
    W = W * pairs / torch.sum(pairs) + W * (1.0 - pairs) / torch.sum(1.0 - pairs)

    # apply Online gradient boosting algorithm
    for i in range(1,len(sim_mats)):
        nu = 2.0/(i + 1.0 )
        sim_mat = (1.0-nu)*sim_mats[i-1] + nu*sim_mats[i]
        criterion = criterion_dict['binominal']
        temp_loss, temp_grad = criterion(sim_mat,T)
        bin_loss  = bin_loss + torch.sum(temp_loss*boosting_weights*W)/len(sub_dim)
        # update boosting_weights by the negative loss gradient of previous learner
        boosting_weights = -1.0* temp_grad

    adv_loss = 0.0
    if config['lambda_div'] > 0.0:
        weight_loss =0.0
        adv_loss, adv_weight_loss = criterion_dict['adversarial'](normed_fvecs)
        weight_loss += adv_weight_loss

        for item in model.last_linear.values():
            W = item.weight.data
            weight_loss += torch.mean((torch.sum(W * W, axis=1) - 1)**2)
    
    total_loss = bin_loss + (adv_loss + weight_loss) * config['lambda_div']
    opt.zero_grad()
    total_loss.backward()
    # log the gradient of each layer
    #lib.utils.GradientMeasure(model,LOG,log_key)
    ### Update network weights!
    opt.step()
    return total_loss.item(),bin_loss.item(),adv_loss.item(), weight_loss.item()


def get_optim(config, model):
    to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
                   **config['opt']['backbone'] }]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_bier_config(config, args)
    metrics = {}
    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val','Grad']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new=True, log_online=config['log_online'])
   
    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = lib.multifeature_resnet50.Network(config)
    _  = model.to(config['device'])

    to_optim = get_optim(config, model)
    criterion_dict ={} 
    criterion_dict['binominal'],to_optim = lib.loss.select(config,to_optim,'binominal')
    criterion_dict['adversarial'],to_optim = lib.loss.select(config,to_optim,'adversarial')
    optimizer = torch.optim.Adam(to_optim)
    if config['scheduler']=='exp':
        scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler']=='step':
        scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['tau'], gamma=config['gamma'])
    elif config['scheduler']=='none':
        print('No scheduling used!')
    else:
        raise Exception('No scheduling option for input: {}'.format(config['scheduler']))

    # create init and eval dataloaders; init used for creating clustered DLs
    dl_train  = lib.data.loader.make(config, model,'train', dset_type = 'train',is_multihot= True)

    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query',is_multihot= True)
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery',is_multihot= True)  
    dl_eval_train  = lib.data.loader.make(config, model,'eval', dset_type = 'train',is_multihot= True)  

    print("Evaluating initial model...")
    lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, True, config['backend'], LOG, 'Val')

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(0, config['nb_epochs']):
        config['epoch'] = e # for wandb
        if config['scheduler']!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_last_lr())))
        
        time_per_epoch_1 = time.time()
        losses = []

        _ = model.train()
        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(e)):
            total_loss, bin_loss, adv_loss, weight_loss= train_batch(model, criterion_dict, optimizer, config, batch, LOG,'Grad')
            losses.append([total_loss, bin_loss, adv_loss, weight_loss])

        time_per_epoch_2 = time.time()
        losses = np.array(losses)
        current_loss = np.mean(losses[:,0])
        LOG.progress_saver['Train'].log('epochs', e)
        LOG.progress_saver['Train'].log('Train_loss', current_loss)
        LOG.progress_saver['Train'].log('bin_loss', np.mean(losses[:,1]))
        LOG.progress_saver['Train'].log('adv_loss', np.mean(losses[:,2]))
        LOG.progress_saver['Train'].log('weight_loss', np.mean(losses[:,3]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))

        # evaluate
        _ = model.eval()
        tic = time.time()
        lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'], LOG, 'Val')
        # evaluate the distance among inter and intra class
        lib.utils.DistanceMeasure(model,config,dl_eval_train,LOG,'Val')

        LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
        LOG.update(all=True)
        print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
        ### Learning Rate Scheduling Step
        if config['scheduler'] != 'none':  scheduler.step()

    t2 = time.time()
    print( "Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))


if __name__ == '__main__':
    main()
    