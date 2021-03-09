import os
import torch
import lib
import warnings
import argparse
import pickle as pkl

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    
# update project name
if  'diva_features' in config.keys() and len(config['diva_features']) ==1:
    config['project'] ='Baseline'
if config['project']=='bier': config['project']='Bier'
if config['project']=='dac': config['project']='D&C'
if config['project']=='diva': config['project']='Diva'

# create dataloader for evaluation
dl_val= lib.data.loader.make(config, 'eval', dset_type = 'val')
dl_test= lib.data.loader.make(config, 'eval', dset_type = 'test')
## optional, check the image distribution for val/test dataset
#lib.utils.plot_dataset_stat(dl_val.dataset,save_path= config['result_path'], dset_type = 'val')
#lib.utils.plot_dataset_stat(dl_test.dataset,save_path= config['result_path'], dset_type = 'test')

# load initial model
pj_base_path= os.path.dirname(os.path.realpath(__file__))
config['pretrained_weights_file'] = pj_base_path + '/' + config['pretrained_weights_file'].split('/')[-1]
model = lib.multifeature_resnet50.Network(config)
_  = model.to(config['device'])

# optional, plot the distance density and retrieved images of initial model on val dataset
#lib.utils.evaluate_standard(model, config, dl_val,is_init=True, is_plot=True,n_img_samples=4,n_closest=8)

# load final model
checkpoint = torch.load(config['checkfolder']+"/checkpoint_recall@1.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
# plot the distance density and retrieved images of final model on val dataset
lib.utils.evaluate_standard(model, config, dl_val,is_plot=True,n_img_samples=4,n_closest=8)

print("Evaluate final model on test dataset") 
#### CREATE A SUMMARY TEXT FILE
summary_text = ""
summary_text += "Evaluate final model on test dataset\n"
scores = lib.utils.evaluate_standard(model, config, dl_test,K=[1,2,4,8],metrics=['recall']) 
for key in scores.keys(): 
  summary_text += "{} :{:.3f}\n".format(key, scores[key])
  with open(config['result_path']+'/evaluate_final_model.txt','w+') as summary_file:
      summary_file.write(summary_text)