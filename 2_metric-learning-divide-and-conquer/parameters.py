from __future__ import print_function

import argparse
def setup_parameters(parser):
     ##### Setup Parameters
    parser.add_argument('--dataset_name',      default='MLRSNet',   type=str,choices=['BigEarthNet', 'MLRSNet'], required = True,  help='Dataset to use. This version support BigEarthNet and MLRSNet with train/val/test split 40%/10%/50%')
    parser.add_argument('--source_path',  default="../Dataset",   type=str,  required = True, help='Path to training data.')
    parser.add_argument('--save_path',    default="../Training_result", type=str,  required = True, help='Where to save everything.')
    return parser

def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--frozen',   action='store_true',help='Flag. If set,for MLRSNet frozen the backbone, for BigEarthNet frozen the backbone except the first layer')
    parser.add_argument('--cuda-device', default = 0, type = int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--random-seed', default = 0, type = int)
    parser.add_argument('--nb-epochs', default=120, type = int)
    parser.add_argument('--sz_embedding', default=512, type=int, help='the dimension of final embedding')
    parser.add_argument('--backbone-wd', default=1e-4, type=float, help='weight decay for backbone')
    parser.add_argument('--backbone-lr', default=1e-5, type=float, help ='learning rate for backbone')
    parser.add_argument('--embedding-wd', default=1e-4, type=float, help='weight decay for embedding layer')
    parser.add_argument('--embedding-lr', default=1e-5, type=float, help='learning rate for embedding layer')
    parser.add_argument('--backend', default='faiss-gpu',choices=('faiss', 'faiss-gpu'))
    return parser

def divid_and_conquer(parser):
    ### for Method Divide and conquer
    parser.add_argument('--mod-epoch', default=2, type = int, help = 'the steps for reclustering train dataset')
    parser.add_argument('--nb-clusters', default=8, type = int, help='the number of learners')
    parser.add_argument('--finetune-epoch', default=110, type = int)
    return parser

def BIER(parser):
    ## for Method Boosting Independent Embeddings (BIER)
    parser.add_argument('--alpha', type=float, default=25, help ='the cost of negative paris for binominal loss')
    parser.add_argument('--beta', type=float, default=2.0, help ='the scale parameter for binominal loss')
    parser.add_argument('--margin', type=float, default=0.5, help ='the magin for binominal loss')
    parser.add_argument('--hard_mining', action='store_true')
    parser.add_argument('--lambda_weight', type=float, default=100, help='weight for decorrelation')
    parser.add_argument('--lambda_div', type=float, default=5e-5, help ='regularization parameter')
    parser.add_argument('--sub_embed_sizes', default=[96,160,256], nargs='+',type=int, help= 'the dimension of sublearners')
    parser.add_argument('--hidden_adversarial_size',type=int, default=512, help='the hidden dimension for adversarial loss')
    return parser 

def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true',help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
    parser.add_argument('--wandb_key',       default='3e2d26edd7a78aa9e725fbed6f42cfdda1cbcfeb',  type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='DAC_Sample_Runs',  type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
    parser.add_argument('--group',           default='Sample_Run',  type=str,   help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                               In --savename default setting part of the savename.')
    return parser 
