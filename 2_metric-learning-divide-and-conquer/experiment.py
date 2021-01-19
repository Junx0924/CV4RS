from __future__ import print_function

import argparse
import math
import matplotlib
import sys
import os
import logging

import train


if __name__ == '__main__':
    pj_base_path= os.path.dirname(os.path.realpath(__file__))
    DIYlogger = logging.getLogger()
    DIYlogger.setLevel(logging.INFO)
    _FMT_STRING = '[%(levelname)s:%(asctime)s] %(message)s'
    _DATE_FMT = '%Y-%m-%d %H:%M:%S'
    file_handler = logging.FileHandler(pj_base_path + "/log/diy_log.txt", mode='w+')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
    DIYlogger.addHandler(file_handler)
    DIYlogger.info("Before parsing arguments")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['BigEarthNet', 'MLRSNet'], required = True)
    parser.add_argument('--classes', required = True, type = int)
    parser.add_argument('--source_path',  default = '../Dataset', type =str,required = True)

    parser.add_argument('--nb-clusters', default=8, type = int)
    parser.add_argument('--nb-epochs', default=120, type = int)
    parser.add_argument('--finetune-epoch', default=110, type = int)
    parser.add_argument('--mod-epoch', default=2, type = int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--sz-embedding', default=128, type=int)
    parser.add_argument('--cuda-device', default = 0, type = int)
    parser.add_argument('--exp', default='0', type=str, help='experiment identifier')
    parser.add_argument('--dir', default='default', type=str)
    parser.add_argument('--backend', default='faiss-gpu',
        choices=('torch+sklearn', 'faiss', 'faiss-gpu')
    )
    parser.add_argument('--random-seed', default = 0, type = int)
    parser.add_argument('--backbone-wd', default=1e-4, type=float)
    parser.add_argument('--backbone-lr', default=1e-5, type=float)
    parser.add_argument('--embedding-lr', default=1e-5, type=float)
    parser.add_argument('--embedding-wd', default=1e-4, type=float)
    parser.add_argument('--verbose', action = 'store_true', default = False)
    parser.add_argument('--log_gpu_info', action = 'store_true',default = True)
    args = vars(parser.parse_args())

    DIYlogger.info("Before loading config")
    config = train.load_config(config_name = pj_base_path+'/config.json')

    config['pj_base_path'] = pj_base_path
    config['pretrained_weights_file'] = pj_base_path + '/' + config['pretrained_weights_file']
    
    config['dataloader']['batch_size'] = args.pop('batch_size')
    config['dataloader']['num_workers'] = args.pop('num_workers')
    config['recluster']['mod_epoch'] = args.pop('mod_epoch')
    config['opt']['backbone']['lr'] = args.pop('backbone_lr')
    config['opt']['backbone']['weight_decay'] = args.pop('backbone_wd')
    config['opt']['embedding']['lr'] = args.pop('embedding_lr')
    config['opt']['embedding']['weight_decay'] = args.pop('embedding_wd')
    config['log_gpu_info'] = args.pop('log_gpu_info')
    
    dataset_name = args.pop('dataset_name')
    num_classes = args.pop('classes')
    config['dataset_selected'] = dataset_name
    config['dataset'][dataset_name]['root'] = args.pop('source_path') + '/' + dataset_name
    config['dataset'][dataset_name]['train'] = range(0,num_classes)
    config['dataset'][dataset_name]['init'] = range(0,num_classes)
    config['dataset'][dataset_name]['eval'] = range(0,num_classes)
    
    DIYlogger.info("Mid loading config 1/2")
    for k in args:
        if k in config:
            config[k] = args[k]

    if config['nb_clusters'] == 1:
        config['recluster']['enabled'] = False
    DIYlogger.info("Mid loading config 2/2")
    config['log'] = {
        'name': '{}-K-{}-M-{}-exp-{}'.format(
            config['dataset_selected'],
            config['nb_clusters'],
            config['recluster']['mod_epoch'],
            args['exp']
        ),
        'path': pj_base_path + '/log/{}'.format(args['dir'])
    }
    DIYlogger.info("After loading config")
    
    os.environ['TORCH_HOME'] = pj_base_path + "/pretrained_weights"

    # tkinter not installed on this system, use non-GUI backend
    matplotlib.use('agg')
    DIYlogger.info("Before train.start(config)")
    train.start(config, DIYlogger)
