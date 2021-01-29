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
from lib.LinearAverage import LinearAverage
from lib.clustering import make_clustered_dataloaders
from torch.autograd import Variable

warnings.simplefilter("ignore", category=PendingDeprecationWarning)
os.putenv("OMP_NUM_THREADS", "8")
pj_base_path= os.path.dirname(os.path.realpath(__file__))
os.environ['TORCH_HOME'] = pj_base_path + "/pretrained_weights"

def load_snca_config(config, args):
    #### UPdate snca parameter 
    config['project'] = 'snca'
    config['margin'] = args.pop('snca_margin')
    config['temperature']= args.pop('snca_temperature')
    config['memory_momentum'] = args.pop('snca_memory_momentum')
    config['momentum'] = args.pop('snca_momentum')
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
    T = batch[1].to(config['device']) # image labels, onehot
    I = batch[2].to(config['device']) # image index

    assert isinstance(T[0],list) ==True
    total_loss = 0.0
    X_var = torch.autograd.Variable(X)
    T_var = torch.autograd.Variable(T)
    I_var = torch.autograd.Variable(I)
    #I_var = torch.autograd.Variable(I)
    feature = model(X_var)
    # caculate the similarity between feature and the memory bank
    output = lemniscate(feature, I_var)
    # from the index_var to get the label similarity mat
    # caculate loss by aligning the similarity mat from features and labels
    nca_loss = criterion_dict['nca'](output, I_var)
    bce_loss = criterion_dict['bce'](feature,T_var)

    total_loss = nca_loss + bce_loss
    optimizer.zero_grad()
    total_loss.backward()
    # log the gradient of each layer
    #lib.utils.GradientMeasure(model,LOG,log_key)
    ### Update network weights!
    optimizer.step()
    return total_loss.item(), nca_loss.item(),bce_loss.item()


def get_optim(config, model):
    # to_optim = [{'params': filter(lambda p: p.requires_grad, model.parameters_dict['backbone']),
    #                **config['opt']['backbone'] }]
    to_optim = [{'params': model.parameters_dict['backbone'],
                    **config['opt']['backbone']}]
    to_optim += [{'params': model.parameters_dict['embedding'],**config['opt']['embedding']}]
    return to_optim


def main():
    config, args = par.load_common_config()
    config = load_snca_config(config, args)
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

    model = lib.model.make(config)
    _ = model.to(config['device'])

    # create init and eval dataloaders; init used for creating clustered DLs
    dl_train = lib.data.loader.make(config, model,'train', dset_type = 'train', is_multihot=True)

    # define lemniscate and loss function (criterion)
    N = len(dl_train.dataset)
    lemniscate = LinearAverage(config['sz_embedding'], N, config['temperature'], config['memory_momentum']).cuda()
    
    to_optim = get_optim(config, model)
    criterion_dict ={}
    criterion_dict['nca'],to_optim  = lib.loss.select(config,to_optim,loss_name='nca', onehot_labels =torch.Tensor(dl_train.dataset.ys))
    criterion_dict['bce'],to_optim  = lib.loss.select(config,to_optim,loss_name='bce')

    optimizer = torch.optim.SGD(to_optim,momentum = config['momentum'], nesterov=True)

    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query',is_multihot= True)
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery',is_multihot= True)  
    dl_eval_train = lib.data.loader.make(config, model,'eval', dset_type = 'train',is_multihot= True)  

    print("Evaluating initial model...")
    lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, True, config['backend'], LOG, 'Val')

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for epoch in range(0, config['nb_epochs']):
        config['epoch'] = epoch # for wandb
        if epoch >= 80:
            lemniscate.params[1] = 0.8
        if epoch >= 120:
            lemniscate.params[1] = 0.9

        time_per_epoch_1 = time.time()
        losses = []

        _ = model.train()
        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(epoch)):
            total_loss, nca_loss, bce_loss= train_batch(model, lemniscate,criterion_dict, optimizer, config, batch)
            losses.append([total_loss, nca_loss, bce_loss])

        time_per_epoch_2 = time.time()
        losses = np.array(losses)
        current_loss = np.mean(losses[:,0])
        LOG.progress_saver['Train'].log('epochs', epoch)
        LOG.progress_saver['Train'].log('total_loss', current_loss)
        LOG.progress_saver['Train'].log('nca_loss', np.mean(losses[:,1]))
        LOG.progress_saver['Train'].log('bce_loss', np.mean(losses[:,2]))
        LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
        print("\nEpoch: {}, loss: {}, time (seconds): {:.2f}.".format(epoch,current_loss,time_per_epoch_2 - time_per_epoch_1))

        # evaluate
        _ = model.eval()
        tic = time.time()
        lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'], LOG, 'Val') 
          
        # evaluate the distance among inter and intra class
        lib.utils.DistanceMeasure(model,config,dl_eval_train,LOG,'Val')
        LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
        LOG.update(all=True)
        print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
      
    t2 = time.time()
    print( "Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))


if __name__ == '__main__':
    main()
    