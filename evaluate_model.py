import os
import torch
import lib
import warnings
import argparse
import pickle as pkl

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def setup_parameters(parser):
    parser.add_argument('--load_from_checkpoint',  default='../', required = True, type=str,  help='the checkpoint folder from training')
    parser.add_argument('--save_path',  default='../',  type=str, required = True, help='the path to save the results')
    parser.add_argument('--source_path',  default='../Dataset',  required = True,type=str,  help='Path to dataset')
    parser.add_argument('--dataset_type',  default='test',  type=str, choices=['val','test','train'],  help='which data spilt to evaluate')
    parser.add_argument('--is_evaluate_initial',   action='store_true',help='Flag. If set, the initial model (epoch 0) will be evaluated')
    parser.add_argument('--is_plot_dist',   action='store_true',help='Flag. If set, it will plot the distance density of inter and intra group')
    return parser

parser = argparse.ArgumentParser()
parser = setup_parameters(parser)
args = vars(parser.parse_args())
checkpoint_folder =  args.pop('load_from_checkpoint')
source_path = args.pop('source_path')
dset_type = args.pop('dataset_type')
is_evaluate_initial = args.pop('is_evaluate_initial')
is_plot_dist = args.pop('is_plot_dist')
result_path = args.pop('save_path')

# load config
with open(checkpoint_folder +"/hypa.pkl","rb") as f:
    config = pkl.load(f)
# update file path
config['checkfolder'] = checkpoint_folder
ds_selected = config['dataset_selected']
config['dataset'][ds_selected]['root'] = source_path +'/'+ds_selected

result_path = config['checkfolder'] +'/evaluation_results'
if not os.path.exists(result_path): os.makedirs(result_path)
config['result_path'] = result_path
    
# create dataloader for evaluation
dl= lib.data.loader.make(config, 'eval', dset_type = dset_type)

# load initial model
pj_base_path= os.path.dirname(os.path.realpath(__file__))
config['pretrained_weights_file'] = pj_base_path + '/' + config['pretrained_weights_file'].split('/')[-1]
model = lib.multifeature_resnet50.Network(config)
_  = model.to(config['device'])
config['epoch'] = 0
### CREATE A SUMMARY TEXT FILE
summary_text, floder_name = "", ""
if is_evaluate_initial:
    summary_text = config['project']+ ": evaluate inital model on "+ dset_type +" dataset\n" 
    floder_name = "/init_" + dset_type
else:
    summary_text = config['project']+ ": evaluate best model on "+ dset_type +" dataset\n"
    floder_name = "/final_" + dset_type
    # load best model
    checkpoint = torch.load(config['checkfolder']+"/checkpoint_recall@1.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    config['epoch'] = checkpoint['epoch'] +1
print(summary_text) 
scores =lib.utils.evaluate_standard(model, config, dl,K=[1,2,4,8],metrics=['recall','map'],is_init=is_evaluate_initial,is_plot_dist=is_plot_dist,is_recover= True) 

for key in scores.keys(): 
  summary_text += "{} :{:.3f}\n".format(key, scores[key])
  with open(config['result_path']+ floder_name +'/evaluate_model.txt','w+') as summary_file:
      summary_file.write(summary_text)