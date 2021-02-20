from __future__ import print_function
from __future__ import division

import os
import numpy as np
import torch
import torch.nn.functional as F
import time
import json
import random
from tqdm import tqdm

import warnings
import parameters as par
from utilities import logger
import lib
from lib.LinearAverage import LinearAverage
import pickle as pkl

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.putenv("OMP_NUM_THREADS", "8")
pj_base_path= os.path.dirname(os.path.realpath(__file__))
os.environ['TORCH_HOME'] = pj_base_path + "/pretrained_weights"

def load_sndl_config(config, args):
    #### UPdate sndl parameter 
    config['project'] = 'sndl'
    config['margin'] = args.pop('sndl_margin')
    config['temperature']= args.pop('sndl_temperature')
    config['memory_momentum'] = args.pop('sndl_memory_momentum')
    config['sub_embed_sizes'] = [config['sz_embedding']]
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

def train_batch(model, lemniscate,criterion_dict, optimizer, config, batch,LOG=None, log_key =''):
    X = batch[0].to(config['device']) # images
    T = batch[1]# image labels,multi-hot
    I = batch[2].to(config['device'])# image index

    assert len(T.size()) ==2
    loss = {}
    feature = model(X)
    # caculate the similarity between feature and the memory bank
    embed_sim = lemniscate(feature, I)
    # from the index to get the label similarity mat
    # caculate loss by aligning the similarity mat from features and labels
    nca = criterion_dict['nca'](embed_sim, I)
    loss['nca'] = nca.item()
    bce = criterion_dict['bce'](feature,T.to(config['device']))
    loss['bce'] = bce.item()
    total_loss = nca  + bce
    loss['Train'] = total_loss.item()
    optimizer.zero_grad()
    total_loss.backward()
    # log the gradient of each layer
    #lib.utils.GradientMeasure(model,LOG,log_key)
    ### Update network weights!
    optimizer.step()
    return loss


def get_optim(config, model):
    to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
                   **config['opt']['backbone'] }]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_sndl_config(config, args)
    start_new = True
    if os.path.exists(config['load_from_checkpoint']):
        start_new = False
        checkfolder = config['load_from_checkpoint']
        checkpoint = torch.load(checkfolder +"/checkpoint_recall@1.pth.tar")
        with open(checkfolder +"/hypa.pkl","rb") as f:
            config_new = pkl.load(f)
        # update the absolute path
        config_new['checkfolder'] = checkfolder
        ds_selected = config['dataset_selected']
        config_new['dataset'][ds_selected]['root'] = config['dataset'][ds_selected]['root']
        config_new['pretrained_weights_file'] =config['pretrained_weights_file']
        config_new['pj_base_path'] = config['pj_base_path']
        config_new['log']['save_path'] = config['log']['save_path']
        config = config_new

    # set random seed for all gpus
    seed = config['random_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = lib.multifeature_resnet50.Network(config)
    if not start_new:
        model.load_state_dict(checkpoint['state_dict'])  
    _  = model.to(config['device'])

    # create dataset
    dl_train = lib.data.loader.make(config, model,'train', dset_type = 'train')
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
    # update num_classes
    ds_name = config['dataset_selected']
    num_classes= dl_train.dataset.nb_classes()
    config['dataset'][ds_name]["classes"] = num_classes

    # define lemniscate and loss function (criterion)
    N = len(dl_train.dataset)
    lemniscate = LinearAverage(config['sz_embedding'], N, config['temperature'], config['memory_momentum']).cuda()
    to_optim = get_optim(config, model)
    criterion_dict ={}
    criterion_dict['nca'],to_optim  = lib.loss.select(config,to_optim,loss_name='nca', multi_hot =torch.Tensor(dl_train.dataset.ys))
    criterion_dict['bce'],to_optim  = lib.loss.select(config,to_optim,loss_name='bce')
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
    
     ## optional, check the image distribution for train dataset
    lib.utils.check_image_label(dl_train.dataset,save_path= config['checkfolder'], dset_type = 'train')
    #################### START TRAINING ###############
    history_recall = 0
    if LOG !=None and "recall" in LOG.progress_saver["Val"].groups.keys():
        history_recall = np.max(LOG.progress_saver["Val"].groups['recall']["recall@1"]['content'])
    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for epoch in range(start_epoch, config['nb_epochs']):
        config['epoch'] = epoch # for wandb
        if config['scheduler']!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_last_lr())))
        if epoch >= 80:
            lemniscate.params[1] = 0.8
        if epoch >= 120:
            lemniscate.params[1] = 0.9

        time_per_epoch_1 = time.time()
        losses ={key:[] for key in criterion_dict.keys()}
        losses ['Train']=[]

        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(epoch)):
            loss= train_batch(model, lemniscate,criterion_dict, optimizer, config, batch)
            [losses[key].append(loss[key]) for key in losses.keys()]

        time_per_epoch_2 = time.time()
        current_loss = np.mean(losses['Train'])
        for key in losses.keys():
            LOG.progress_saver['Train'].log(key+'_loss', np.mean(losses[key]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(epoch,current_loss,time_per_epoch_2 - time_per_epoch_1))

        # evaluate
        if epoch % config['eval_epoch'] ==0:
            _ = model.eval()
            tic = time.time()
            scores =lib.utils.evaluate_standard(model, config, dl_query, False, config['backend'], LOG, 'Val') 
            print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
            LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
            _ = model.train()

            if scores['recall@1'] >history_recall:
                ### save checkpoint #####
                print("Best epoch! save to checkpoint")
                savepath = config['checkfolder']+'/checkpoint_{}.pth.tar'.format("recall@1")
                torch.save({'state_dict':model.state_dict(), 'epoch':epoch, 'progress': LOG.progress_saver, 'optimizer':optimizer.state_dict()}, savepath)
        
        LOG.update(all=True)
        ### Learning Rate Scheduling Step
        if config['scheduler'] != 'none':  scheduler.step()

    full_training_time = time.time()-t1
    print('Training Time: {} min.\n'.format(np.round(full_training_time/60,2))) 
    
if __name__ == '__main__':
    main()
    