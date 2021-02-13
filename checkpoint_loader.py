import os
import matplotlib
import numpy as np
import torch
from utilities import misc
from utilities import logger
import lib
import argparse
import json
import matplotlib.pyplot as plt

checkpoint_folder=""
### CREATE A SUMMARY TEXT FILE
summary_text = ""

# load config
with open(checkpoint_folder + '/Parameter_Info.json', 'rb') as f:
    config = json.load(f)

# load initial model
model = lib.multifeature_resnet50.Network(config)

# create dataloader for evaluation
dl_train = lib.data.loader.make(config, model,'eval', dset_type = 'train')
dl_query = lib.data.loader.make(config, model,'eval', dset_type = 'query')
dl_gallery = lib.data.loader.make(config, model,'eval', dset_type = 'gallery')

print("Evaluate inital model\n")
summary_text += "Evaluate inital model\n"
scores = lib.utils.evaluate_query_gallery(model, config, dl_query, dl_gallery, False, config['backend'],is_init=True, K=[1],metrics=config['eval_metric'])
for key in scores.keys(): 
    summary_text += "{} :{:.3f}\n".format(key, scores[key])
with open(LOG.config['checkfolder']+'/evaluate_inital_model.txt','w') as summary_file:
    summary_file.write(summary_text)

print("Evaluate final model\n")    
lib.utils.eval_final_model(model,config,dl_train,dl_query,dl_gallery,checkpoint_folder)