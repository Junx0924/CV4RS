from .margin_loss import MarginLoss
from .adversarial_loss import Adversarial
from .binominal_loss import BinomialLoss
from .fast_moco import Fast_moco
from .batchminner.semihard import Semihard
from .batchminner.random_distance import random_distance
from .batchminner.intra_random import intra_random
from .batchminner.distance import distance

def select(config,loss_name=''):
    ds_name = config['dataset_selected']
    nb_classes = int(config['transform_parameters'][ds_name]["classes"])
   
    if loss_name =='discriminative': 
        criterion = MarginLoss(nb_classes,batchminer= distance())
    elif loss_name =='selfsimilarity': 
        criterion = Fast_moco(config) 
    elif loss_name =='separation': 
        criterion =Adversarial(config['hidden_adversarial_size']) 
    elif loss_name == 'adversarial':
        criterion =Adversarial(config['hidden_adversarial_size'],need_weight_loss= True) 
    elif loss_name =='intra': 
        criterion = MarginLoss(nb_classes,batchminer=  intra_random())  
    elif loss_name =='shared':
        criterion = MarginLoss(nb_classes,batchminer=  random_distance())
    elif loss_name =='binominal':
        criterion = BinomialLoss(config['alpha'],config['beta'], config['margin'])
    else:
        criterion = MarginLoss(nb_classes,batchminer= Semihard())
    return criterion.to(config['device'])
                    
