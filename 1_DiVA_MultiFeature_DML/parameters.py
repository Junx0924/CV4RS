import argparse, os


#######################################
def basic_training_parameters(parser):

    ### General Training Parameters
    parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr',             default=-1,       type=float, help='Optional. If not -1, sets the learning rate for the final linear embedding layer.')
    parser.add_argument('--n_epochs',          default=150,      type=int,   help='Number of training epochs.')
    parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
    parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
    parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
    parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
    parser.add_argument('--tau',               default=[10000],  nargs='+',type=int,help='Stepsize before reducing learning rate.')


    ##### Loss-specific Settings
    parser.add_argument('--use_sgd',         action='store_true',   help='Flag, Optimization method to use. If set true, use sgd, else use adam by default')
    parser.add_argument('--loss',            default='margin',      type=str,   help='Loss function for features like discriminative, shared and intra. For supported methods, please check criteria/__init__.py')
    parser.add_argument('--loss_ssl',   default='fast_moco', type=str,   help='Loss function for self-similarity feature. For supported methods, please check criteria/__init__.py')
    parser.add_argument('--batch_mining',    default='distance',    type=str,   help='Batchminer for tuple-based losses: For supported methods, please check batchminer/__init__.py')
    parser.add_argument('--extension',       default='none',        type=str,   help='Extension Method to standard metric learning losses')

    ##### Network-related Flags
    parser.add_argument('--num_in_channels', default=3,type=int, help='input data channels')
    parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: dim=64, 128 or 512 is used in most papers., depending on the architecture.')
    parser.add_argument('--arch',         default='multifeature_resnet50_normalize',  type=str, help='This version support: multifeature_resnet50_normalize or multifeature_bninception_normalize')
    parser.add_argument('--not_pretrained',   action='store_true', help='Flag. If set, no ImageNet pretraining is used to initialize the network.')

    ##### Evaluation Parameters
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'nmi', 'f1', 'mAP_c'], type=str, help='Metrics to evaluate performance by.')
    parser.add_argument('--evaltypes',          nargs='+', default=['discriminative'], type=str, help='The network may produce multiple embeddings (ModuleDict). If the key is listed here, the entry will be evaluated on the evaluation metrics.\
                                                                                                       Note: One may use Combined_embed1_embed2_..._embedn-w1-w1-...-wn to compute evaluation metrics on weighted (normalized) combinations.')
    parser.add_argument('--storage_metrics',    nargs='+', default=['e_recall@1'], type=str, help='Improvement in these metrics on the testset trigger checkpointing.')
    parser.add_argument('--realistic_augmentation', action='store_true', help='Flag, apply preprocessing/augmentation to use on the data, with color/brightness changes, flip')
    parser.add_argument('--gpu',          default=[0], nargs='+', type=int,   help='Random seed for reproducibility.')
    return parser

def setup_parameters(parser):
    ##### Setup Parameters
    parser.add_argument('--dataset',         default='MLRSNet',   type=str,   help='Dataset to use. This version support BigEarthNet and MLRSNet with train/val/test split 40%/10%/50%')
    parser.add_argument('--source_path',  default="../Dataset",   type=str, help='Path to training data.')
    parser.add_argument('--save_path',    default="../Diva", type=str, help='Where to save everything.')
    parser.add_argument('--savename',     default='group_plus_seed',   type=str,   help='Appendix to save folder name if any special information is to be included.')
    return parser


#######################################
def diva_parameters(parser):
    ##### Multifeature Parameters
    parser.add_argument('--batch_mining_shared',      default='random_distance', type=str,       help='batchmining method for shared features, choose among [shared_neg_distance, random_distance, shared_full_distance]')
    parser.add_argument('--batch_mining_intra',       default='intra_random', type=str,          help='batchmining method for intra features, this version only support intra_random')
    parser.add_argument('--diva_features',            default=['discriminative'], nargs='+', type=str,   help='Type of features to learn, this version support: [discriminative, selfsimilarity, shared, intra]')
    parser.add_argument('--diva_decorrelations',      default=[], nargs='+', type=str, help= 'The decorrelations between features')
    parser.add_argument('--diva_rho_decorrelation',   default=[1500], nargs='+', type=float, help='Weights for adversarial Separation of embeddings.')

    ### Adversarial Separation Loss for learning the decorrelation between features
    parser.add_argument('--diva_decorrnet_dim', default=512,     type=int, help='list, which indicates the size of the sub-embedding(e.g. [96, 160, 256,512]')
    parser.add_argument('--diva_decorrnet_lr',  default=0.00001, type=float, help='')

    ### Deep Clustering
    parser.add_argument('--diva_dc_update_f', default=2,    type=int, help='')
    parser.add_argument('--diva_dc_ncluster', default=300,  type=int, help='')

    ### (Fast) Momentum Contrast Loss for learning the selfsimiarility feature
    parser.add_argument('--diva_moco_momentum',      default=0.9, type=float, help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--diva_moco_temperature',   default=0.1, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--diva_moco_n_key_batches', default=50,  type=int, help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--diva_moco_lower_cutoff',  default=0.5,  type=float, help='')
    parser.add_argument('--diva_moco_upper_cutoff',  default=1.4,  type=float, help='')

    parser.add_argument('--diva_moco_temp_lr',        default=0.0005,   type=float, help='initial learning rate')
    parser.add_argument('--diva_moco_trainable_temp', action='store_true', help='')

    ### Weights for each feature space training objective
    parser.add_argument('--diva_alpha_ssl',      default=0.3,  type=float, help='weight for selfsimilarity feature')
    parser.add_argument('--diva_alpha_shared',   default=0.3,  type=float, help='weight for Class-shared feature') 
    parser.add_argument('--diva_alpha_intra',    default=0.3,  type=float, help='weight for Intra-class features') 

    return parser


#######################################
def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true',help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
    parser.add_argument('--wandb_key',       default='3e2d26edd7a78aa9e725fbed6f42cfdda1cbcfeb',  type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='DiVA_Sample_Runs',  type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
    parser.add_argument('--group',           default='Sample_Run',  type=str,   help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                               In --savename default setting part of the savename.')

    return parser


#######################################
def loss_specific_parameters(parser):
    ### Contrastive Loss
    parser.add_argument('--loss_contrastive_pos_margin', default=0, type=float, help='positive margins for contrastive pairs.')
    parser.add_argument('--loss_contrastive_neg_margin', default=1, type=float, help='negative margins for contrastive pairs.')

    ### Triplet-based Losses
    parser.add_argument('--loss_triplet_margin',       default=0.2,         type=float, help='Margin for Triplet Loss')

    ### MarginLoss
    parser.add_argument('--loss_margin_margin',       default=0.2,          type=float, help='Triplet margin')
    parser.add_argument('--loss_margin_beta_lr',      default=0.0005,       type=float, help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta',         default=1.2,          type=float, help='Initial Class Margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu',           default=0,            type=float, help='Regularisation value on betas in Margin Loss.')
    parser.add_argument('--loss_margin_beta_constant',             action='store_true', help='Flag. If set, beta-values are left untrained.')

    ### ProxyNCA
    parser.add_argument('--loss_proxynca_lr',     default=0.0005,     type=float, help='Learning Rate for Proxies in ProxyNCALoss.')
    #NOTE: The number of proxies is determined by the number of data classes.

    ### NPair L2 Penalty
    parser.add_argument('--loss_npair_l2',     default=0.005,        type=float, help='L2 weight in NPair. Note: Set to 0.02 in paper, but multiplied with 0.25 in the implementation as well.')

    ### Angular Loss
    parser.add_argument('--loss_angular_alpha',             default=36, type=float, help='Angular margin in degrees.')
    parser.add_argument('--loss_angular_npair_ang_weight',  default=2,  type=float, help='relative weighting between angular and npair contribution.')
    parser.add_argument('--loss_angular_npair_l2',          default=0.005,  type=float, help='L2 weight on NPair (as embeddings are not normalized).')

    ### Multisimilary Loss
    parser.add_argument('--loss_multisimilarity_pos_weight', default=2,         type=float, help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight', default=40,        type=float, help='Weighting on negative similarities.')
    parser.add_argument('--loss_multisimilarity_margin',     default=0.1,       type=float, help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_thresh',     default=0.5,       type=float, help='Exponential thresholding.')

    ### Lifted Structure Loss
    parser.add_argument('--loss_lifted_neg_margin', default=1,     type=float, help='Margin placed on similarities.')
    parser.add_argument('--loss_lifted_l2',         default=0.005, type=float, help='As embeddings are not normalized, they need to be placed under penalty.')

    ### Binomial Deviance Loss
    parser.add_argument('--loss_binomial_pos_weight', default=2,         type=float, help='Weighting on positive similarities.')
    parser.add_argument('--loss_binomial_neg_weight', default=40,        type=float, help='Weighting on negative similarities.')
    parser.add_argument('--loss_binomial_margin',     default=0.1,       type=float, help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_binomial_thresh',     default=0.5,       type=float, help='Exponential thresholding.')

    ### Quadruplet Loss
    parser.add_argument('--loss_quadruplet_margin_alpha1',  default=1,   type=float, help='Quadruplet Loss requires two margins. This is the first one.')
    parser.add_argument('--loss_quadruplet_margin_alpha2',  default=0.5, type=float, help='Quadruplet Loss requires two margins. This is the second one.')

    ### Soft-Triplet Loss
    parser.add_argument('--loss_softtriplet_n_centroids',   default=10,       type=int,   help='Number of proxies per class.')
    parser.add_argument('--loss_softtriplet_margin_delta',  default=0.01,    type=float, help='Margin placed on sample-proxy similarities.')
    parser.add_argument('--loss_softtriplet_gamma',         default=0.1,     type=float, help='Weight over sample-proxies within a class.')
    parser.add_argument('--loss_softtriplet_lambda',        default=20,      type=float, help='Serves as a temperature.')
    parser.add_argument('--loss_softtriplet_reg_weight',    default=0.2,     type=float, help='Regularization weight on the number of proxies.')
    parser.add_argument('--loss_softtriplet_lr',            default=0.0005,  type=float, help='Learning Rate multiplier for proxies.')

    ### Normalized Softmax Loss
    parser.add_argument('--loss_softmax_lr',           default=0.00001,   type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_softmax_temperature',  default=0.05,   type=float, help='Temperature for NCA objective.')

    ### Histogram Loss
    parser.add_argument('--loss_histogram_nbins',  default=51, type=int, help='Number of bins for histogram discretization.')

    ### SNR Triplet (with learnable margin) Loss
    parser.add_argument('--loss_snr_margin',      default=0.2,   type=float, help='Triplet margin.')
    parser.add_argument('--loss_snr_reg_lambda',  default=0.005, type=float, help='Regularization of in-batch element sum.')
    parser.add_argument('--loss_snr_beta',        default=0,     type=float, help='Example values: 0.2')
    parser.add_argument('--loss_snr_beta_lr',     default=0.0005,type=float, help='Example values: 0.2')

    ### ArcFace Loss
    parser.add_argument('--loss_arcface_lr',             default=0.0005,  type=float, help='Learning rate on class proxies.')
    parser.add_argument('--loss_arcface_angular_margin', default=0.5,     type=float, help='Angular margin in radians.')
    parser.add_argument('--loss_arcface_feature_scale',  default=64,      type=float, help='Inverse Temperature for NCA objective.')

    return parser



#######################################
def batchmining_specific_parameters(parser):
    ### Distance-based_Sampling
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float,help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float, help='Upper cutoff on distances - values above are IGNORED.')
    return parser


#######################################
def batch_creation_parameters(parser):
    parser.add_argument('--data_sampler',              default='class_random', type=str, help='How the batch is created. Available options: See datasampler/__init__.py.')
    parser.add_argument('--samples_per_class',         default=2,              type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_batchmatch_bigbs',     default=512,         type=int, help='Size of batch to be summarized into a smaller batch. For distillation/coreset-based methods.')
    parser.add_argument('--data_batchmatch_ncomps',    default=10,         type=int, help='Number of batch candidates that are evaluated, from which the best one is chosen.')
    parser.add_argument('--data_storage_no_update',    action='store_true', help='Flag for methods that need a sample storage. If set, storage entries are NOT updated.')
    parser.add_argument('--data_d2_coreset_lambda',    default=1, type=float, help='Regularisation for D2-coreset.')
    parser.add_argument('--data_gc_coreset_lim',       default=1e-9, type=float, help='D2-coreset value limit.')
    parser.add_argument('--data_sampler_lowproj_dim',  default=-1, type=int, help='Optionally project embeddings into a lower dimension to ensure that greedy coreset works better. Only makes a difference for large embedding dims.')
    parser.add_argument('--data_sim_measure',          default='euclidean', type=str, help='Distance measure to use for batch selection.')
    parser.add_argument('--data_gc_softened',          action='store_true', help='Flag. If set, use a soft version of greedy coreset.')
    parser.add_argument('--data_idx_full_prec',        action='store_true', help='Deprecated.')
    parser.add_argument('--data_mb_mom',               default=-1, type=float, help='For memory-bank based samplers - momentum term on storage entry updates.')
    parser.add_argument('--data_mb_lr',                default=1, type=float, help='Deprecated.')

    return parser
