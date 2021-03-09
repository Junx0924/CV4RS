from __future__ import print_function
from __future__ import division

import os
import numpy as np
import torch
import pickle as pkl
import time
import json
import random
from tqdm import tqdm
import itertools
import torch.nn.functional as F

import warnings
import parameters as par
from utilities import logger
import lib


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.putenv("OMP_NUM_THREADS", "8")
 

def load_bier_config(config, args):
    #### UPdate Bier parameter 
    config['project'] = 'Bier'
    config['lambda_weight'] = args.pop('bier_lambda_weight')
    config['lambda_div']= args.pop('bier_lambda_div')
    config['adversarial_weight']= args.pop('bier_adversarial_weight')
    config['sub_embed_sizes'] = args.pop('bier_sub_embed_sizes')
    assert sum(config['sub_embed_sizes']) == config['sz_embedding']
    
    # config the decorrelation between features
    decorrelation = list(itertools.combinations(config['sub_embed_sizes'],2))
    config['decorrelation'] = {}
    for item,weight in zip(decorrelation,config['adversarial_weight']):
        direction_name = str(item[0])+ '-' + str(item[1])
        config['decorrelation'][direction_name] = {'dim':str(item[0])+ '-' + str(item[1]),'weight':weight}

    config['hidden_adversarial_size'] = args.pop('bier_hidden_adversarial_size')
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



def train_batch(model, criterion_dict,opt, config, batch,LOG=None, log_key =''):
    X = batch[0] # images
    T = batch[1] # class labels
    I = batch[2] # image ids

    feature = model(X.to(config['device']))
    T_list = lib.utils.classBalancedSamper(T,config['num_samples_per_class'])
    new_I = T_list[:,0]
    feature = feature[new_I]
    T = torch.tensor(T_list[:,1]).to(config['device'])

    # l2 normalize feature
    normed_fvecs = {}
    sub_dim = config['sub_embed_sizes']
    for i in range(len(sub_dim)):
        start = int(sum(sub_dim[:i]))
        stop = int(start + sub_dim[i])
        fvec = F.normalize(feature[:, start:stop],p =2,dim =1)
        normed_fvecs[str(sub_dim[i])]= fvec

    loss = {}
    bin_loss = criterion_dict['binominal'](normed_fvecs,T)
    loss['binominal'] = bin_loss.item()
    adv_loss, adv_weight_loss,embed_weight_loss = 0.0, 0.0,0.0
    if config['lambda_div'] > 0.0:
        adv_loss = criterion_dict['adversarial'](normed_fvecs)
        loss['adversarial'] = adv_loss.item()
        for regressor in criterion_dict['adversarial'].regressors.values():
            for i in range(len(regressor)):
                # relu layer has no weights and bias
                if i !=1:
                    W_hat = regressor[i].weight
                    B_hat = regressor[i].bias
                    adv_weight_loss += torch.mean((torch.sum(W_hat * W_hat, axis=1) - 1)**2) + torch.max(torch.tensor([0.0,torch.sum(B_hat * B_hat) - 1.0])) 
        adv_weight_loss = adv_weight_loss / len(criterion_dict['adversarial'].regressors)
        loss['adv_weight'] = adv_weight_loss.item()
        for item in model.last_linear.values():
            W = item.weight
            embed_weight_loss += torch.mean((torch.sum(W * W, axis=1) - 1)**2)
        embed_weight_loss = embed_weight_loss / len(model.last_linear)
        loss['embed_weight'] = embed_weight_loss.item()
    
    total_loss = bin_loss + config['lambda_div']*(adv_loss + config['lambda_weight']* (adv_weight_loss + embed_weight_loss))
    loss['Train'] = total_loss.item()
    opt.zero_grad()
    total_loss.backward()
    
    ### Update network weights!
    opt.step()
    return loss


def get_optim(config, model):
    to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
                   **config['opt']['backbone'] }]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_bier_config(config, args)
    start_new = True
    if os.path.exists(config['load_from_checkpoint']):
        start_new = False
        checkfolder = config['load_from_checkpoint']
        checkpoint = torch.load(checkfolder +"/checkpoint_recall@1.pth.tar")
        config['checkfolder'] = checkfolder
        with open(checkfolder +"/hypa.pkl","rb") as f:
            config_old = pkl.load(f)
            config['wandb']['wandb_id'] = config_old['wandb']['wandb_id']

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = lib.multifeature_resnet50.Network(config)
    if not start_new:
        model.load_state_dict(checkpoint['state_dict']) 
    if len(config['gpu_ids'])>1:
        model = torch.nn.DataParallel(model) 
    _  = model.to(config['device'])

    # define loss function (criterion)
    to_optim = get_optim(config, model)
    config['class_specific_beta'] =False
    criterion_dict ={} 
    criterion_dict['binominal'],to_optim = lib.loss.select(config,to_optim,'binominal')
    criterion_dict['adversarial'],to_optim = lib.loss.select(config,to_optim,'adversarial')
    optimizer = torch.optim.Adam(to_optim)
    
    # create dataset
    dl_train  = lib.data.loader.make(config ,'train', dset_type = 'train')
    dl_val = lib.data.loader.make(config ,'eval', dset_type = 'val') 
    # update num_classes
    ds_name = config['dataset_selected']
    num_classes= dl_train.dataset.nb_classes()
    config['dataset'][ds_name]["classes"] = num_classes
    config['dataloader']['batch_size'] = num_classes* config['num_samples_per_class']

    # config learning scheduler
    if not start_new:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if config['scheduler']=='exp':
        scheduler    = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler']=='step':
        scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['tau'], gamma=config['gamma'])
    elif config['scheduler']=='none':
        print('No scheduling used!')
    else:
        raise Exception('No scheduling option for input: {}'.format(config['scheduler']))
   
    #################### CREATE LOGGING FILES ###############
    if config['log_online']: lib.utils.start_wandb(config)
    sub_loggers = ['Train', 'Val', 'Grad']
    LOG = logger.LOGGER(config, sub_loggers=sub_loggers, start_new= start_new, log_online=config['log_online'])
    config['checkfolder'] = LOG.config['checkfolder']
    start_epoch =0
    if not start_new:
        LOG.progress_saver= checkpoint['progress']
        start_epoch = checkpoint['epoch'] + 1
    
    ## optional, check the image distribution for train dataset
    #lib.utils.plot_dataset_stat(dl_train.dataset,save_path= config['checkfolder'], dset_type = 'train')
    #################### START TRAINING ###############
    history_recall = 0
    if LOG !=None and "recall" in LOG.progress_saver["Val"].groups.keys():
        history_recall = np.max(LOG.progress_saver["Val"].groups['recall']["recall@1"]['content'])
    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for e in range(start_epoch, config['nb_epochs']):
        config['epoch'] = e # for wandb
        if config['scheduler']!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_last_lr())))
        
        time_per_epoch_1 = time.time()
        if config['lambda_div'] > 0.0:
            losses ={key:[] for key in ['Train','adv_weight','embed_weight','binominal','adversarial']}
        else:
            losses ={key:[] for key in ['Train','binominal']}

        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(e)):
            loss= train_batch(model, criterion_dict, optimizer, config, batch, LOG,'Grad')
            [losses[key].append(loss[key]) for key in losses.keys()]

        time_per_epoch_2 = time.time()
        current_loss = np.mean(losses['Train'])
        for key in losses.keys():
            LOG.progress_saver['Train'].log(key+'_loss', np.mean(losses[key]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(e,current_loss,time_per_epoch_2 - time_per_epoch_1))

        # evaluate
        if e % config['eval_epoch'] ==0:
            _ = model.eval()
            tic = time.time()
            scores =lib.utils.evaluate_standard(model, config, dl_val, False, LOG, 'Val') 
            if scores['recall@1'] >history_recall:
                ### save checkpoint #####
                history_recall = scores['recall@1']
                print("Best epoch! save to checkpoint")
                savepath = config['checkfolder']+'/checkpoint_{}.pth.tar'.format("recall@1")
                torch.save({'state_dict':model.state_dict(), 'epoch':e, 'progress': LOG.progress_saver, 'optimizer':optimizer.state_dict()}, savepath)
            LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
            _ = model.train()
        
        LOG.update(all=True)
        ### Learning Rate Scheduling Step
        if config['scheduler'] != 'none':  scheduler.step()

    full_training_time = time.time()-t1
    print('Training Time: {} min.\n'.format(np.round(full_training_time/60,2))) 

if __name__ == '__main__':
    main()
    
