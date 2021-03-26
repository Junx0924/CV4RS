from __future__ import print_function
from __future__ import division

import os
import numpy as np
import torch
import pickle as pkl
import torch.nn.functional as F
import time
import random
from tqdm import tqdm

import warnings
import parameters as par
from utilities import logger
import lib

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.putenv("OMP_NUM_THREADS", "8")
 

def train_batch(model,criterion, optimizer, config, batch):
    X = batch[0] # images
    T = batch[1]# class labels
    I = batch[2] # image ids
    M = model(X.to(config['device']))
    if config['batch_minner'] != 'multiLabelSemihard':
        # convert multi-hot labes to category labels
        T_list = lib.utils.classBalancedSamper(T,config['num_samples_per_class'])
        new_I = T_list[:,0]
        M = M[new_I]
        T = torch.tensor(T_list[:,1])
    else:
        T = torch.from_numpy(np.array(T, dtype=np.float32)).to(config['device'])
    loss = criterion(M, T)
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
    config['project'] = 'Baseline'
    config['sub_embed_sizes'] = [config['sz_embedding']]
    config['batch_minner'] = args.pop('baseline_batch_minner')
    config['loss_name'] = args.pop('baseline_loss')
    start_new = True
    if os.path.exists(config['load_from_checkpoint']):
        start_new = False
        checkfolder = config['load_from_checkpoint']
        checkpoint = torch.load(checkfolder +"/checkpoint_recall@1.pth.tar")
        with open(checkfolder +"/hypa.pkl","rb") as f:
            config_old = pkl.load(f)
            config['wandb']['wandb_id'] = config_old['wandb']['wandb_id']
        
    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # get model
    model = lib.multifeature_resnet50.Network(config)
    if not start_new:
        model.load_state_dict(checkpoint['state_dict'])
    if len(config['gpu_ids'])>1:
        model = torch.nn.DataParallel(model)  
    _  = model.to(config['device'])
    
    # create dataset
    dl_train = lib.data.loader.make(config, 'train', dset_type = 'train')
    dl_val = lib.data.loader.make(config,  'eval', dset_type = 'val')
    # update num_classes
    ds_name = config['dataset_selected']
    num_classes= dl_train.dataset.nb_classes()
    config['dataset'][ds_name]["classes"] = num_classes
    config['dataloader']['batch_size'] = num_classes* config['num_samples_per_class']
   
    # config loss function and optimizer
    to_optim = get_optim(config, model)
    criterion, to_optim = lib.loss.select(config,to_optim,config['loss_name'], config['batch_minner'])
    optimizer = torch.optim.Adam(to_optim)
    if not start_new:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # config learning scheduler
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
        losses_per_epoch = []
        _ = model.train()
        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(e)):
            loss = train_batch(model, criterion,optimizer, config, batch)
            losses_per_epoch.append(loss)

        time_per_epoch_2 = time.time()
        current_loss = np.mean(losses_per_epoch)
        LOG.progress_saver['Train'].log('epochs', e)
        LOG.progress_saver['Train'].log('Train_loss', current_loss)
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
    
