import os
import matplotlib
import numpy as np
import torch
from utilities import misc
from utilities import logger
import lib
import argparse
import matplotlib.pyplot as plt
import pickle as pkl

def setup_parameters(parser):
    parser.add_argument('--load_from_checkpoint',  default='../',  type=str,  help='the checkpoint folder from training')
    parser.add_argument('--source_path',  default='../Dataset',  type=str,  help='Path to dataset')
    return parser

parser = argparse.ArgumentParser()
parser = setup_parameters(parser)
args = vars(parser.parse_args())
checkpoint_folder =  args.pop('load_from_checkpoint')
source_path = args.pop('source_path')

# load config
with open(checkpoint_folder +"/hypa.pkl","rb") as f:
    config = pkl.load(f)
# update file path
config['checkfolder'] = checkpoint_folder
ds_selected = config['dataset_selected']
config['dataset'][ds_selected]['root'] = source_path +'/'+ds_selected

if 'result_path' not in config.keys():
    result_path = config['checkfolder'] +'/evaluation_results'
    if not os.path.exists(result_path): os.makedirs(result_path)
    config['result_path'] = result_path

checkpoint = torch.load(config['checkfolder']+"/checkpoint_recall@1.pth.tar")
# load initial model
model = lib.multifeature_resnet50.Network(config)
model.load_state_dict(checkpoint['state_dict'])
_  = model.to(config['device'])

# create dataloader for evaluation
dl_test= lib.data.loader.make(config, 'eval', dset_type = 'test')
## optional, check the image distribution for each dataset
lib.utils.check_image_label(dl_test.dataset,save_path= config['result_path'], dset_type = 'test')

print("Evaluate final model")    
#### CREATE A SUMMARY TEXT FILE
summary_text = ""
summary_text += "Evaluate final model\n"
scores = lib.utils.evaluate_standard(model, config, dl_test, False, K=[1],metrics=config['eval_metric']) 
for key in scores.keys(): 
  summary_text += "{} :{:.3f}\n".format(key, scores[key])
  with open(config['result_path']+'/evaluate_final_model.txt','w+') as summary_file:
      summary_file.write(summary_text)