from .margin_loss import MarginLoss
from .adversarial_loss import Adversarial
from .binominal_loss import BinomialLoss
from .fast_moco import Fast_moco

from .batchminner.semihard import Semihard
from .batchminner.random_distance import Random_distance
from .batchminner.intra_random import Intra_random
from .batchminner.distance import Distance

def select(config,to_optim,loss_name="",minner_name= ""):
    batch_minner = select_batchminner(minner_name)
    if 'margin' in loss_name:
        ds_name = config['dataset_selected']
        nb_classes = int(config['transform_parameters'][ds_name]["classes"])
        criterion = MarginLoss(nb_classes,batchminner = batch_minner)
        to_optim    += [{'params':criterion.parameters(), 'lr':criterion.beta_lr}]
    elif 'moco' in loss_name: 
        criterion = Fast_moco(config) 
    elif  'adversarial' in loss_name : 
        criterion = Adversarial(config['hidden_adversarial_size'],config['decorrelation']) 
        if config['project'] =='bier':
            criterion =Adversarial(config['hidden_adversarial_size'],config['decorrelation'],need_weight_loss= True) 
        to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]
    elif 'binominal' in loss_name : 
        criterion = BinomialLoss(config['alpha'],config['beta'], config['margin'])
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