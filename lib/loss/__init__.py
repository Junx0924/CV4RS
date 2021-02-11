from .margin_loss import MarginLoss
from .adversarial_loss import Adversarial
from .binominal_loss import BinomialLoss
from .fast_moco import Fast_moco
from .nca import NCACrossEntropy
from .bcelogitloss import BCELogitLoss

from .batchminner.semihard import Semihard
from .batchminner.random_distance import Random_distance
from .batchminner.intra_random import Intra_random
from .batchminner.distance import Distance

def select(config,to_optim,loss_name="",minner_name= "",onehot_labels = None):
    batch_minner = select_batchminner(minner_name)
    ds_name = config['dataset_selected']
    num_classes = int(config['dataset'][ds_name]["classes"])
    if 'margin' in loss_name:
        criterion = MarginLoss(num_classes,batchminner = batch_minner)
        to_optim    += [{'params':criterion.parameters(), 'lr':criterion.beta_lr}]
    elif 'moco' in loss_name: 
        criterion = Fast_moco(config) 
    elif  'adversarial' in loss_name : 
        criterion = Adversarial(config['hidden_adversarial_size'],config['decorrelation']) 
        to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]
    elif 'binominal' in loss_name : 
        criterion = BinomialLoss()
    elif 'nca' in loss_name:
        criterion = NCACrossEntropy(onehot_labels,config['margin'] / config['temperature'])
    elif 'bce' in loss_name:
        criterion = BCELogitLoss(config['sz_embedding'],num_classes)
        to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]
    else:
        raise Exception('Loss {} not available!'.format(loss_name))
    return criterion.to(config['device']), to_optim
                    
def select_batchminner(name=""):
    if name =='semihard' :
        batchminner = Semihard()
    elif name =='distance' :
        batchminner = Distance()
    elif name =='intra_random' :
        batchminner = Intra_random()
    elif name =='random_distance' :
        batchminner = Random_distance()
    else:
        batchminner = None
    return batchminner