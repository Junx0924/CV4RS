import numpy as np
import os
import pickle as pkl
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
import architectures as archs
import datasampler   as dsamplers
import datasets      as dsets
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
import argparse
from pathlib import Path
################### INPUT ARGUMENTS ###################
def setup_parameters(parser):
    ##### Setup Parameters
    parser.add_argument('--dataset',         default='MLRSNet',   type=str,   help='Dataset to use. This version support BigEarthNet and MLRSNet with train/val/test split 40%/10%/50%')
    parser.add_argument('--source_path',  default="../Dataset",   type=str, help='Path to test data.')
    parser.add_argument('--save_path',    default="../Diva", type=str, help='Path to training result.')
    return parser
parser = argparse.ArgumentParser()
parser = setup_parameters(parser)
setup_opt =  parser.parse_args()

### set the dir to save downloaded pretrained model pth file
### for running on HPC: 
cur_path= os.path.dirname(os.path.realpath(__file__))
os.environ['TORCH_HOME'] = cur_path +'/architectures' 

#RERUN SPECIFIC RUNS!
train_result = setup_opt.save_path +'/'+ setup_opt.dataset
networks = [i.name for i in Path(train_result).iterdir()]

for network in networks:
    netfolder = train_result + '/' + network
    opt       = pkl.load(open(netfolder +'/hypa.pkl','rb'))
    model     = archs.select(opt.arch, opt)
    if 'bninception' in opt.arch and opt.dataset=='MLRSNet':
        model.load_state_dict(torch.load(netfolder +'/checkpoint_Combined_discriminative_selfsimilarity_shared_intra-0.5-1-1-1_e_recall@1.pth.tar')['state_dict'])
        weightslist     = [[0.5,1,1,1]]
    elif 'bninception' in opt.arch and opt.dataset=='BigEarthNet':
        model.load_state_dict(torch.load(netfolder+ '/checkpoint_Combined_discriminative_selfsimilarity_shared_intra-0.5-2-2-2_e_recall@1.pth.tar')['state_dict'])
        weightslist     = [[0.5,2,2,2]]
    elif 'resnet50' in opt.arch and opt.dataset=='MLRSNet':
        model.load_state_dict(torch.load(netfolder+ '/checkpoint_Combined_discriminative_selfsimilarity_shared_intra-0.5-1-1-1_e_recall@1.pth.tar')['state_dict'])
        weightslist     = [[0.5,1,1,1]]
    elif 'resnet50' in opt.arch and opt.dataset=='BigEarthNet':
        model.load_state_dict(torch.load(netfolder+ '/checkpoint_Combined_discriminative_selfsimilarity_shared_intra-0.5-2-2-2_e_recall@1.pth.tar')['state_dict'])
        weightslist     = [[0.5,2,2,2]]
    else:
        raise Exception('Setup not available!')


    """================================"""
    os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"


    """==============================="""
    dataloaders     = {}
    opt.source_path = setup_opt.source_path + '/'+ opt.dataset
    datasets        = dsets.select(opt.dataset, opt, opt.source_path)
    device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)


    """================================"""
    # weightslist     = [[0.5,1.25,1.25,1.25],[0.5,1,1,1],[0.5,0.85,0.85,0.85],[0.5,0.7,0.7,0.7]]
    target_labels   = []
    adjust          = lambda weights,x: torch.nn.functional.normalize(torch.cat([w*s_out for w,(_,s_out) in zip(weights,out.items())], dim=-1), dim=-1).cpu().detach().numpy().tolist()
    feature_colls   = [[] for _ in range(len(weightslist))]

    _ = model.to(device)
    _ = model.eval()
    for i,inp in enumerate(tqdm(dataloaders['testing'])):
        input_img,target = inp[1],inp[0]
        target_labels.extend(target.numpy().tolist())
        out = model(input_img.to(device))

        if isinstance(out,tuple): out, aux_f  = out

        for j,weights in enumerate(weightslist):
            feature_colls[j].extend(adjust(weights,out))




    target_labels   = np.array(target_labels)
    list_of_metrics = [metrics.select(metricname) for metricname in opt.evaluation_metrics]
    metrics_list    = [{} for _ in range(len(weightslist))]
    n_classes       = opt.n_classes
    for k,(weights,features) in enumerate(zip(weightslist,feature_colls)):
        features = np.vstack(features).astype('float32')

        ####################################
        cpu_cluster_index = faiss.IndexFlatL2(features.shape[-1])
        kmeans            = faiss.Clustering(features.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000
        ### Train Kmeans
        kmeans.train(features, cpu_cluster_index)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])

        ###################################
        faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
        faiss_search_index.add(centroids)
        _, computed_cluster_labels = faiss_search_index.search(features, 1)

        faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
        faiss_search_index.add(features)

        ##################################
        max_kval            = np.max([int(x.split('@')[-1]) for x in opt.evaluation_metrics if 'recall' in x])
        _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]


        for f,metric in enumerate(list_of_metrics):
            input_dict = {}
            if 'features' in metric.requires:         input_dict['features'] = features
            if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
            if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
            if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
            if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes
            metrics_list[k][metric.name] = metric(**input_dict)



    s = '{}:\n{}'.format(network,'\n'.join(str(wlist)+': '+', '.join('{0}:{1:1.3f}'.format(key,item) for key,item in mdict.items()) for wlist,mdict in zip(weightslist,metrics_list)))
    print(s)
