import argparse

def setup_parameters(parser):
    ##### Setup Parameters
    parser.add_argument('--dataset_name',      default='MLRSNet',   type=str,choices=['BigEarthNet', 'MLRSNet'], required = True,  help='Dataset to use. This version support BigEarthNet and MLRSNet with train/val/test split 40%/10%/50%')
    parser.add_argument('--source_path',  default="../Dataset",   type=str,  required = True, help='Path to training data.')
    parser.add_argument('--save_path',    default="../Training_result", type=str,  required = True, help='Where to save everything.')
    return parser

def basic_training_parameters(parser):
    ### General Training Parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=20)
    parser.add_argument('--eta_style', action='store_true')
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--lambda_weight', type=float, default=100.0)
    parser.add_argument('--lambda_div', type=float, default=0.0)
    parser.add_argument('--samples_per_class', type=int, default=2)

    ### Learning rate
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_anneal', type=int, default =25000)
    parser.add_argument('--embedding_lr_multiplier', type=float, default=10.0)
    parser.add_argument('--use_same_learnrate', action='store_true')
    parser.add_argument('--weight_decay', default=0.0002, type=float)

    parser.add_argument('--shrinkage', type=float, default=0.06)
    parser.add_argument('--embedding_sizes', default=[96,160,256], nargs='+',type=int)
    parser.add_argument('--regularization', type=str, default='activation',choices=['activation', 'adversarial'])
    parser.add_argument('--hidden_adversarial_size',type=int, default=512)
    parser.add_argument('--num_hidden_adversarial', type=int,default=2)
    return parser

def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true',help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
    parser.add_argument('--wandb_key',       default='3e2d26edd7a78aa9e725fbed6f42cfdda1cbcfeb',  type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='DAC_Sample_Runs',  type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
    parser.add_argument('--group',           default='Sample_Run',  type=str,   help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                               In --savename default setting part of the savename.')
    parser.add_argument('--savename', default ='', type = str)
    return parser 