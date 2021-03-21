from __future__ import print_function

import argparse
import os
import json
import torch

def setup_parameters(parser):
     ##### Setup Parameters
    parser.add_argument('--dataset_name',      default='MLRSNet',   type=str,choices=['BigEarthNet', 'MLRSNet'], required = True,  help='Dataset to use. This version support BigEarthNet and MLRSNet with train/val/test split 40%/10%/50%')
    parser.add_argument('--source_path',  default="../Dataset",   type=str,  required = True, help='Path to training data.')
    parser.add_argument('--save_path',    default="../Training_result", type=str,  required = True, help='Where to save everything.')
    return parser

def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--frozen',   action='store_true',help='Flag. If set,frozen the backbone')
    parser.add_argument('--gpu_ids', default=[0], nargs='+', type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--random-seed', default = 0, type = int)
    parser.add_argument('--nb_epochs', default=120, type = int)
    parser.add_argument('--sz_embedding', default=512, type=int, help='the dimension of final embedding')
    parser.add_argument('--backbone-wd', default=1e-4, type=float, help='weight decay for backbone')
    parser.add_argument('--backbone-lr', default=1e-5, type=float, help ='learning rate for backbone')
    parser.add_argument('--embedding-wd', default=1e-4, type=float, help='weight decay for embedding layer')
    parser.add_argument('--embedding-lr', default=1e-5, type=float, help='learning rate for embedding layer')
    parser.add_argument('--backend',      default='faiss-gpu',choices=('faiss', 'faiss-gpu'))
    parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
    parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
    parser.add_argument('--tau',               default=[55],nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')
    parser.add_argument('--use_npmem',   action='store_true',help='Flag. If set, create npmem file and read data from npmem during training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_samples_per_class',type=int, default=2, help='number of images sampled for each class')
    parser.add_argument('--savename', default="", type = str, help='the folder name to save the training results')
    parser.add_argument('--eval_epoch',type=int, default=10, help='epoch interval for validation')
    parser.add_argument('--eval_metric',  default=['recall','map'], nargs='+', type=str, help='metric for evaluate performance of multi-label learning')
    parser.add_argument('--load_from_checkpoint', default="", type=str,  help='the checkpoint folder to resume the training')
    parser.add_argument('--unfrozen_layers', default=[-1], nargs='+', type=int, help='layer index for unfrozen, select from 0 to 7')
    return parser

def divide_and_conquer(parser):
    ### for Method Divide and conquer
    parser.add_argument('--dac_mod_epoch', default=10, type = int, help = 'the steps for reclustering train dataset')
    parser.add_argument('--dac_nb_clusters', default=8, type = int, help='the number of learners')
    parser.add_argument('--dac_finetune_epoch', default=110, type = int, help='after the finetune epoch, there is no reclustering when training')
    parser.add_argument('--dac_class_specific_beta',   action='store_true',help='Flag. If set, the margin beta for margin loss will be trainable and class-specific')
    return parser

def bier(parser):
    ## for Method Boosting Independent Embeddings (BIER)
    parser.add_argument('--bier_lambda_weight', type=float, default=1e5, help='the regularization parameter for the weights of embedding layer and adversarial layer')
    parser.add_argument('--bier_adversarial_weight', default=[1e5,1e5,1e5], nargs='+', type=float, help= 'Weights for adversarial Separation of embeddings.')
    parser.add_argument('--bier_lambda_div', type=float, default=5e-5, help ='regularization parameter')
    parser.add_argument('--bier_sub_embed_sizes', default=[96,160,256], nargs='+',type=int, help= 'the dimension of features')
    parser.add_argument('--bier_hidden_adversarial_size',type=int, default=512, help='the hidden dimension for adversarial loss')
    ## Binomial loss
    parser.add_argument('--bier_beta_trainable',   action='store_true',help='Flag. If set, the margin beta for binomial loss will be trainable')
    return parser 

def diva(parser):
    ## for Method Diverse Visual Feature Aggregation (Diva)
    parser.add_argument('--diva_features',   default=['discriminative', 'selfsimilarity', 'shared', 'intra'], nargs='+', type=str,   help='Type of features to learn, this version support: [discriminative, selfsimilarity, shared, intra]')
    parser.add_argument('--diva_alpha_ssl',      default=0.3,  type=float, help='weight for selfsimilarity feature')
    parser.add_argument('--diva_alpha_shared',   default=0.3,  type=float, help='weight for Class-shared feature') 
    parser.add_argument('--diva_alpha_intra',    default=0.3,  type=float, help='weight for Intra-class feature') 
    parser.add_argument('--diva_evaluation_weight', nargs='+', default=[0.5,1,1,1], type=float, help='to compute evaluation metrics on weighted (normalized) combinations')
    
    ## Margin loss
    parser.add_argument('--diva_class_specific_beta',   action='store_true',help='Flag. If set, the margin beta for margin loss will be trainable and class-specific')
    
    ### (Fast) Momentum Contrast Loss for learning the selfsimiarility feature
    parser.add_argument('--diva_moco_momentum',      default=0.9, type=float, help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--diva_moco_temperature',   default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--diva_moco_n_key_batches', default=30,  type=int, help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--diva_moco_lower_cutoff',  default=0.5,  type=float, help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--diva_moco_upper_cutoff',  default=1.4,  type=float, help='Upper cutoff on distances - values above are IGNORED.')

    ### Adversarial loss for decorrelating features
    parser.add_argument('--diva_hidden_adversarial_size',type=int, default=512, help='the hidden dimension for adversarial loss')
    parser.add_argument('--diva_adversarial_weight',      default=[1500,1500,1500], nargs='+', type=int, help= 'Weights for adversarial Separation of embeddings.')
    return parser 

def sndl(parser):
    ## for Method Scalable Neighborhood Component Analysis
    parser.add_argument('--sndl_margin', default=0.0, type=float,help='classification margin')
    parser.add_argument('--sndl_temperature', default=0.05, type=float,  help='temperature parameter')
    parser.add_argument('--sndl_memory_momentum',  default=0.5, type=float,  help='momentum for non-parametric updates')                
    return parser 

def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true',help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
    parser.add_argument('--wandb_key',       default='3e2d26edd7a78aa9e725fbed6f42cfdda1cbcfeb',  type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='DAC_Sample_Runs',  type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
    parser.add_argument('--group',           default='Sample_Run',  type=str,   help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                              In --savename default setting part of the savename.')
    parser.add_argument('--wandb_dry_run',     action='store_true', help='if set, wandb will not access the internet to update wandb records online')
    return parser 

def load_common_config():
    parser = argparse.ArgumentParser()
    parser = basic_training_parameters(parser)
    parser = setup_parameters(parser)
    parser = wandb_parameters(parser)
    parser = diva(parser)
    parser = divide_and_conquer(parser)
    parser = bier(parser)
    parser = sndl(parser)
    args = vars(parser.parse_args())

    ##### Read config.json
    pj_base_path= os.path.dirname(os.path.realpath(__file__))
    config_name = pj_base_path +'/config.json'
    with open(config_name, 'r') as f: config = json.load(f)
   
    #### Update config.json from INPUT ARGUMENTS ###########
    config['nb_epochs'] = args.pop('nb_epochs')
    config['sz_embedding'] = args.pop('sz_embedding')
    config['pj_base_path'] = pj_base_path
    config['dataloader']['batch_size'] = args.pop('batch_size')
    config['pretrained_weights_file'] = pj_base_path + '/' + config['pretrained_weights_file']
    config['dataloader']['num_workers'] = args.pop('num_workers')
    config['opt']['backbone']['lr'] = args.pop('backbone_lr')
    config['opt']['backbone']['weight_decay'] = args.pop('backbone_wd')
    config['opt']['embedding']['lr'] =args.pop('embedding_lr')
    config['opt']['embedding']['weight_decay'] =args.pop('embedding_wd')
    dataset_name =  args.pop('dataset_name')
    config['dataset_selected'] = dataset_name
    config['dataset_selected'] = dataset_name
    config['dataset'][dataset_name]['root'] = args.pop('source_path') + '/' + dataset_name
    config['random_seed'] = args.pop('random_seed')
    config['log_online'] = args.pop('log_online')
    config['load_from_checkpoint'] = args.pop('load_from_checkpoint')
    config['frozen'] = args.pop('frozen')
    config['log']['save_path'] = args.pop('save_path')
    savename = args.pop('savename')
    config['log']['save_name'] = savename if savename !="" else dataset_name +'_s{}'.format(config['random_seed'])
    config['scheduler'] = args.pop('scheduler')
    config['decay'] = args.pop('decay')
    config['gamma'] = args.pop('gamma')
    config['tau'] = args.pop('tau')
    config['use_npmem'] = args.pop('use_npmem')
    config['eval_epoch'] = args.pop('eval_epoch')
    config['eval_metric'] = args.pop('eval_metric')
    config['gpu_ids']= args.pop('gpu_ids')
    config['device'] = torch.device("cuda") if len(config['gpu_ids'])>0 else torch.device("cpu")
    config['unfrozen_layers'] = args.pop('unfrozen_layers')

    ### Update wandb config  ###########
    if config['log_online']:
        config['wandb']['dry_run'] = args.pop('wandb_dry_run')
        config['wandb']['wandb_key'] = args.pop('wandb_key')
        config['wandb']['project'] =args.pop('project')
        config['wandb']['group'] =args.pop('group')
        # update save_name
        config['log']['save_name'] = savename if savename !="" else config['wandb']['group']+'_s{}'.format(config['random_seed'])
        
    # update save_path
    config['log']['save_path'] = config['log']['save_path']+ '/' + dataset_name

    for k in args:
        if k in config:
            config[k] = args[k]

    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                if type(config[k]) is str:
                    # if python types, then evaluate str expressions
                    if config[k][:5] in ['range', 'float']:
                        config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)   
    return config, args

