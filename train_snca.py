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
    T = batch[1]# image labels,multi-hot
    I = batch[2].to(config['device'])# image index

    assert len(T.size()) ==2
    loss = {}
    feature = model(X)
    # caculate the similarity between feature and the memory bank
    output = lemniscate(feature, I)
    # from the index to get the label similarity mat
    # caculate loss by aligning the similarity mat from features and labels
    loss['nca'] = criterion_dict['nca'](output, I)
    loss['bce'] = criterion_dict['bce'](feature,T.to(config['device']))
    loss['Train'] = loss['nca']  + loss['bce']
    optimizer.zero_grad()
    loss['Train'].backward()
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

    model = lib.multifeature_resnet50.Network(config)
    _  = model.to(config['device'])

    # create init and eval dataloaders; init used for creating clustered DLs
    dl_train = lib.data.loader.make(config, model,'train', dset_type = 'train')

    # define lemniscate and loss function (criterion)
    N = len(dl_train.dataset)
    lemniscate = LinearAverage(config['sz_embedding'], N, config['temperature'], config['memory_momentum']).cuda()
    
    to_optim = get_optim(config, model)
    criterion_dict ={}
    criterion_dict['nca'],to_optim  = lib.loss.select(config,to_optim,loss_name='nca', onehot_labels =torch.Tensor(dl_train.dataset.ys))
    criterion_dict['bce'],to_optim  = lib.loss.select(config,to_optim,loss_name='bce')
    #criterion_dict['margin'], to_optim = lib.loss.select(config,to_optim,'margin','semihard')

    optimizer = torch.optim.SGD(to_optim,momentum = config['momentum'], nesterov=True)

    # create query and gallery dataset for evaluation
    dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
    dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery')  

    print("Training for {} epochs.".format(config['nb_epochs']))
    t1 = time.time()

    for epoch in range(0, config['nb_epochs']):
        config['epoch'] = epoch # for wandb
        if epoch >= 80:
            lemniscate.params[1] = 0.8
        if epoch >= 120:
            lemniscate.params[1] = 0.9

        time_per_epoch_1 = time.time()
        losses ={}
        losses ={key:[] for key in criterion_dict.keys()}
        losses ['Train']=[]

        for batch in tqdm(dl_train,desc = 'Train epoch {}.'.format(epoch)):
            loss= train_batch(model, lemniscate,criterion_dict, optimizer, config, batch)
            [losses[key].append(loss[key].item()) for key in losses.keys()]

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
            checkpoint = lib.utils.evaluate_standard(model, config, dl_query, False, config['backend'], LOG, 'Val') 
            if checkpoint: 
                # check retrieval performance
                lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'], LOG, 'Val')  
            #lib.utils.DistanceMeasure(model,config,dl_eval_train,LOG,'Val')
            LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
            _ = model.train()
        LOG.update(all=True)
    t2 = time.time()
    print( "Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))


if __name__ == '__main__':
    main()
    