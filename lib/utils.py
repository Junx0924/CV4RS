from __future__ import print_function
from __future__ import division

from . import evaluation
from . import similarity
from . import faissext
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import PIL
import os
from osgeo import gdal
from sklearn.manifold import TSNE
import time
import random
import csv
import pandas as pd
import seaborn as sns


def predict_batchwise(model, dataloader, device,use_penultimate = False, is_dry_run=False,desc=''):
    """
    Get the embeddings of the dataloader from model
        Args:
            model: pretrained resnet50
            dataloader: torch dataloader
            device: torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            use_penultimate: use the embedding layer if it is false
        return:
            numpy array of embeddings, labels, indexs 
    """
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    if desc =='': desc ='Filling memory queue'
    with torch.no_grad():
        # use tqdm when the dataset is large (SOProducts)
        #is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc= desc):
            img_data, labels, indices = batch
            if not is_dry_run:
                img_data = img_data.to(device)
                img_embeddings = model(img_data, use_penultimate).data.cpu().numpy()
            else:
                # just a placeholder not to break existing code
                img_embeddings = np.array([-1])
            A[0].append(np.array(img_embeddings))
            A[1].append(labels.numpy())
            A[2].append(indices.numpy())
            
        result = [np.concatenate(A[i],axis =0) for i in range(len(A))]
    model.train()
    model.train(model_is_training) # revert to previous training state
    if is_dry_run:
        # do not return features if is_dry_run
        return [None, *result[1:]]
    else:
        return result


def get_weighted_embed(X,weights,sub_dim):
    """
    Get weighted embeddigns
        Args:
            X: numpy array, embeddings
            weights: list of weight of each sub embeddings, like [0.1, 1, 1]
            sub_dim: list of size of sub embeddings, like [96,160,256]
        return:
            weighted embeddings
    """
    assert len(weights) == len(sub_dim)
    assert X.shape[1] == sum(sub_dim)
    for i in range(len(sub_dim)):
        start = int(sum(sub_dim[:i]))
        stop = int(start + sub_dim[i])
        X[:, start:stop] = weights[i]*X[:, start:stop]
    return X


def evaluate_query_gallery(model, config, dl_query, dl_gallery, use_penultimate= False,  
                          LOG=None, log_key = 'Val',is_init=False,K = [1,2,4,8],metrics=['recall'], is_validation= False):
    """
    Evaluate the retrieve performance
    Args:
        model: pretrained resnet50
        dl_query: query dataloader
        dl_gallery: gallery dataloader
        use_penultimate: use the embedding layer if it is false
        K: [1,2,4,8]
        metrics: default ['recall']
        is_validation: if set true it will just do validation, no evaluation results will be generated
    Return:
        score: dict of score for different metrics
    """
    # calculate embeddings with model and get targets
    X_query, T_query, _ = predict_batchwise(model, dl_query, config['device'],use_penultimate,desc="Extraction Query Features")
    X_gallery, T_gallery, _ = predict_batchwise(model, dl_gallery, config['device'],use_penultimate,desc='Extraction Gallery Features')

    if 'evaluation_weight' in config.keys() and not is_init:
        X_query = get_weighted_embed(X_query,config['evaluation_weight'],config['sub_embed_sizes'])
        X_gallery = get_weighted_embed(X_gallery,config['evaluation_weight'],config['sub_embed_sizes'])

    # make sure the query and the gallery has same number of classes
    assert dl_query.dataset.nb_classes() == dl_gallery.dataset.nb_classes()

    k_closest_points, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query,k= max(K),gpu_id= config['gpu_id'])
    T_query_pred   = T_gallery[k_closest_points]

    scores={}
    for k in K:
        y_pred = np.array([ np.sum(y[:k],axis=0) for y in T_query_pred])
        y_pred[np.where(y_pred>1)]= 1
        for metric in metrics:
            s = evaluation.select(metric,T_query,y_pred)
            print("{}@{} : {:.3f}".format(metric, k, s))
            scores[metric+ '@'+str(k)] = s
            if LOG !=None:
                LOG.progress_saver[log_key].log(metric+ '@'+str(k),s,group=metric)
    
    df,shared_info = check_distance_ratio(np.vstack((X_query,X_gallery)), np.vstack((T_query,T_gallery)),LOG=LOG,log_key=log_key)
    
    if not is_validation:
        if 'result_path' not in config.keys():
            result_path = config['checkfolder'] +'/evaluation_results'
            if not os.path.exists(result_path): os.makedirs(result_path)
            config['result_path'] = result_path

        # plot intra and inter dist distribution
        plot_intra_inter_dist(df,shared_info, config['result_path'])

        ## recover n_closest images
        n_img_samples = 10
        n_closest = 4
        recover_save_path = config['result_path']+'/sample_recoveries.png'
        recover_query_gallery(X_query,X_gallery,dl_query.dataset.im_paths, dl_gallery.dataset.im_paths, recover_save_path,n_img_samples, n_closest,gpu_id=config['gpu_id'])
        tsne_save_path =  config['result_path']+'/tsne.png'
        check_tsne_plot(np.vstack((X_query,X_gallery)), np.vstack((T_query,T_gallery)), dl_query.dataset.conversion, tsne_save_path)  
        
        if 'Mirco_F1' in metrics:
            y_pred = np.array([ np.sum(y[:1], axis =0) for y in T_query_pred])
            TP, FP, TN, FN = evaluation.functions.multilabelConfussionMatrix(T_query,y_pred)
            save_path =config['result_path']+'/confussionMatrix.csv'
            with open(save_path,'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['TP']+list([int(i) for i in TP]))
                writer.writerow(['FP']+list([int(i) for i in FP]))
                writer.writerow(['TN']+list([int(i) for i in TN]))
                writer.writerow(['FN']+list([int(i) for i in FN]))
    return scores


def evaluate_standard(model, config,dl, use_penultimate= False, 
                    LOG=None, log_key = 'Val',is_init=False,K = [1,2,4,8],metrics=['recall'],is_validation= False):
    """
    Evaluate the retrieve performance
        Args:
            model: pretrained resnet50
            dl: dataloader
            use_penultimate: use the embedding layer if it is false
            K: [1,2,4,8]
            metrics: default ['recall']
            is_validation: if set false, evaluation results will be generated
        Return:
            scores: dict of score for different metrics
    """
    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dl, config['device'], use_penultimate, desc='Extraction Eval Features')
    if 'evaluation_weight' in config.keys() and not is_init:
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    
    k_closest_points, _ = faissext.find_nearest_neighbors(X, queries= X, k=max(K)+1,gpu_id= config['gpu_id'])
    # leave itself out
    T_pred = T[k_closest_points[:,1:]]

    scores={}
    for k in K:
        y_pred = np.array([ np.sum(y[:k],axis=0) for y in T_pred])
        y_pred[np.where(y_pred>1)]= 1
        for metric in metrics:
            s = evaluation.select(metric,T,y_pred)
            print("{}@{} : {:.3f}".format(metric, k, s))
            scores[metric+ '@'+str(k)] = s
            if LOG !=None:
                LOG.progress_saver[log_key].log(metric+ '@'+str(k),s,group=metric)

    df,shared_info = check_distance_ratio(X, T,LOG,log_key)

    if not is_validation:
        if 'result_path' not in config.keys():
            result_path = config['checkfolder'] +'/evaluation_results'
            if not os.path.exists(result_path): os.makedirs(result_path)
            config['result_path'] = result_path

        # plot the intra and inter dist distribution
        plot_intra_inter_dist(df,shared_info, config['result_path'])
    
        ## recover n_closest images
        n_img_samples = 10
        n_closest = 4
        save_path = config['result_path']+'/sample_recoveries.png'
        recover_standard(X,dl.dataset.im_paths,save_path,n_img_samples, n_closest,gpu_id=config['gpu_id'])
        check_tsne_plot(X,T, dl.dataset.conversion, config['result_path']+'/tsne.png') 

        if 'Mirco_F1' in metrics:
            y_pred = np.array([np.sum(y[:1], axis =0) for y in T_pred])
            TP, FP, TN, FN = evaluation.functions.multilabelConfussionMatrix(T,y_pred)
            save_path = config['result_path']+'/confussionMatrix.csv'
            with open(save_path,'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(['TP']+list(TP))
                writer.writerows(['FP']+list(FP))
                writer.writerows(['TN']+list(TN))
                writer.writerows(['FN']+list(FN))  
    return scores


def recover_standard(X, img_paths,save_path, n_img_samples = 10, n_closest = 4,gpu_id=None):
    """
    Recover the n closest similar images for sampled images
        Args:
            X: embeddings
            img_paths: the original image paths of embeddings
    """
    print('Start to recover {} similar images for each sampled image'.format(n_closest))
    start_time = time.time()
    sample_idxs = np.random.choice(np.arange(len(X)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X, queries= X[sample_idxs],
                                                k=n_closest+1,
                                                gpu_id= gpu_id)
    pred_img_paths = np.array([[img_paths[i] for i in ii] for ii in nns[:,1:]])
    sample_paths = [img_paths[i] for i in sample_idxs]
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    plot_recovered_images(image_paths,save_path)
    print("Recover similar images done! it takes: {:.2f} s.\n".format(time.time()- start_time))

def recover_query_gallery(X_query, X_gallery,query_img_paths,gallery_img_path, save_path, n_img_samples = 10, n_closest = 4,gpu_id=None):
    """
    Recover the n closest similar gallery images for sampled query images
        Args:
            X_query: query embeddings
            X_gallery: gallery embeddings
            query_img_paths: the original image paths of query embeddings
            gallery_img_path: the original image paths of gallery embeddings
    """
    print('Start to recover {} similar gallery images for each sampled query image'.format(n_closest))
    start_time = time.time()
    assert X_gallery.shape[1] == X_gallery.shape[1]
    sample_idxs = np.random.choice(np.arange(len(X_query)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query[sample_idxs],
                                                 k=n_closest,
                                                 gpu_id= gpu_id
        )
    pred_img_paths = np.array([[gallery_img_path[i] for i in ii] for ii in nns])
    sample_paths = [query_img_paths[i] for i in sample_idxs]
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    plot_recovered_images(image_paths,save_path)
    print("Recover done! Time elapsed: {:.2f} seconds.\n".format(time.time()- start_time))

def plot_recovered_images(image_paths,save_path):
    """
    Plot images and save them
        Args:
            image_paths: numpy array
            save_path
    """
    width = image_paths.shape[1]
    f,axes = plt.subplots(image_paths.shape[0],image_paths.shape[1])
    temp_sample_paths = image_paths.flatten()
    temp_axes = axes.flatten()
    for i in range(len(temp_sample_paths)):
        plot_path = temp_sample_paths[i]
        ax = temp_axes[i]
        if ".png" in plot_path or ".jpg" in plot_path:
            img_data = np.array(PIL.Image.open(plot_path))
        else:
            # get RGB channels from the band data of BigEarthNet
            tif_img =[]
            patch_name = plot_path.split("/")[-1]
            for band_name in ['B04','B03','B02']:
                img_path = plot_path +'/'+ patch_name+'_'+band_name+'.tif'
                band_ds = gdal.Open(img_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = np.array(raster_band.ReadAsArray()) 
                band_data = normalize(band_data,norm="max")*255
                tif_img.append(band_data)
            img_data =np.moveaxis(np.array(tif_img,dtype=int), 0, -1)
        ax.imshow(img_data)
        ax.set_xticks([])
        ax.set_yticks([])
        if i%width:
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

def classBalancedSamper(T,num_samples_per_class=2):
    """
    Get a list of category labels with its original index from multi-hot labels
        Args:
            T: multi-hot labels, eg.numpy array[n_samples x 60]
            num_samples_per_class: default 2
        Return:
            list of category labels
    """
    T_list = [ np.where(t==1)[0] for t in T] 
    T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
    classes = np.unique(T_list[:,1])
    image_dict = {str(c):[] for c in classes}
    [image_dict[str(c)].append(ind) for ind,c in T_list]
    new_T_list =[]
    for c in image_dict.keys():
        replace = True if len(image_dict[c])< num_samples_per_class else False
        inds = np.random.choice(image_dict[c],num_samples_per_class,replace=replace)
        new_T_list.append([inds[0],int(c)])
        new_T_list.append([inds[1],int(c)])
    return np.array(new_T_list)


def check_distance_ratio(X, T, LOG=None, log_key="Val",conversion=None):
    """
    log the distance ratios between intra-class and inter-class.
        Args:
            X: embeddings
            T: multi-hot labels
    """
    start_time = time.time()
    print('Start to calculate intra and inter dist ratio')
    # compute the l2 distance for each embedding
    dist = similarity.pairwise_distance(X)
    # get the number of shared labels
    shared = np.matmul(T,np.transpose(T))
    # only store the up triangle area to reduce computing
    ind_pairs = [[[i,j] for j in range(i) ] for i in range(1,len(X))]
    ind_pairs = np.array([item for sublist in ind_pairs for item in sublist])
    shared_info_keys = np.unique(shared[ind_pairs[:,0],ind_pairs[:,1]])
    shared_info ={key:[] for key in shared_info_keys}
    {shared_info[shared[p[0],p[1]]].append(dist[p[0],p[1]]) for p in ind_pairs}

    # get the labels for each embedding
    T_list = [ np.where(t==1)[0] for t in T] 
    T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
    labels = np.unique(T_list[:,1])
    embed_dict = {str(c):[] for c in labels}
    [embed_dict[str(c)].append(ind) for ind,c in T_list]

    # Compute average intra and inter l2 distance
    all_dist,all_labels,dist_type =[],[],[]
    for label in labels:
        intra_inds = embed_dict[str(label)]
        if len(intra_inds)<2: continue
        intra_pairs = [[[i,j] for j in range(i)] for i in range(1,len(intra_inds))]
        intra_pairs = np.array([item for sublist in intra_pairs for item in sublist])
        intra_dist = dist[intra_pairs[:,0],intra_pairs[:,1]]
        mean_intra = np.mean(intra_dist) 

        other_inds = list(set(range(len(X))) - set(intra_inds))
        inter_pairs = [[[i,j] for j in other_inds] for i in intra_inds]
        inter_pairs = np.array([item for sublist in inter_pairs for item in sublist])
        inter_dist =  dist[inter_pairs[:,0],inter_pairs[:,1]]
        mean_inter = np.mean(inter_dist)

        all_dist += list(intra_dist) + list(inter_dist)
        dist_type +=['intra']*len(intra_dist) +['inter']*len(inter_dist)
        label_name =  conversion[str(label)] if conversion !=None else str(label)
        all_labels +=[label_name]*(len(intra_dist)+len(inter_dist))
        
        if LOG !=None:
            LOG.progress_saver[log_key].log('distRatio@'+str(label),mean_intra/mean_inter, group ='distRatio')
    
    df = pd.DataFrame({'Distance': np.array(all_dist) ,
                'class':  np.array(all_labels) ,
                "dist_type":np.array(dist_type)})
    print("Calculate done! Time elapsed: {:.2f} s.\n".format(time.time()- start_time))
    return df,shared_info

def check_gradient(model,LOG,log_key):
    """
    Check the gradient of each layer in the model
    The gradient is supposed to bigger at the embedding layer
    gradually descrease to the first conv layer
    """
    # record the gradient of the weight of each layer in the model
    for name, param in model.named_parameters():
        if param.requires_grad == True and 'weight' in name and param.grad is not None:
            grads = param.grad.detach().cpu().numpy().flatten()
            grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
            LOG.progress_saver[log_key].log(name+'_l2',grad_l2)
            LOG.progress_saver[log_key].log(name+'_max',grad_max)


def check_tsne_plot(X,T,save_path,n_components=2):
    """
    Get the tsne plot of embeddings
        Args:
            X: embeddings
            T: multi-hot labels
            conversion: dictionary to convert category label to label name
    """
    # apply tsne to the embeddings
    print("Apply tsne to embeddings")
    time_start = time.time()
    tsne = TSNE(n_components=n_components, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    plt.figure(figsize=(16,10))
    plt.scatter(tsne_results[:,0], tsne_results[:,1])
    plt.savefig(save_path, format='png')
    plt.close()

def check_image_label(dataset,save_path, dset_type='train'):
    """
    Generally check if the datasets have similar distribution
     Check Avg. num labels per image
     Check Avg. num of labels shared per image
     Args:
        dataset: torch.dataset
    """
    save_path = save_path +'/stat_' + dset_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    avg_labels_per_image = np.mean(np.sum(dataset.ys,axis= 1))
    print("Avg. num labels per image = "+ str(avg_labels_per_image))
    images_per_label =[len(dataset.image_dict[label]) for label in dataset.image_dict.keys()]
    plt.figure(figsize=(15,6))
    plt.bar([c for c in dataset.image_dict.keys()] ,height=np.array(images_per_label)/sum(images_per_label))
    plt.xlabel("label")
    plt.ylabel("Percent of samples")
    plt.title("Distribution of samples for "+ dataset.dataset_name + " "+ dset_type + " dataset")
    plt.savefig(save_path +'/statistic_samples.png')
    plt.close()

    label_counts = np.sum(dataset.ys,axis= 1)
    count_dict = {}
    for item in label_counts:
        count_dict[item] = count_dict.get(item, 0) + 1
    num_labels = sorted([k for k in count_dict.keys()])
    counts = np.array([count_dict[k] for k in num_labels])
    plt.figure()
    plt.bar(num_labels,counts/np.sum(counts),edgecolor='w')
    plt.xlabel("label counts")
    plt.ylabel("Percent of samples")
    plt.title("Distribution of label counts for "+ dataset.dataset_name + " "+ dset_type+ " dataset")
    plt.savefig(save_path+'/statistic_labels.png')
    plt.close()

    # # get the number of shared labels
    shared = np.matmul(dataset.ys,np.transpose(dataset.ys))
    # only store the up triangle area to reduce computing
    shared_dict ={}
    ind_pairs = [[[i,j] for j in range(i) ] for i in range(1,len(dataset.ys))]
    ind_pairs = np.array([item for sublist in ind_pairs for item in sublist])
    shared_info = shared[ind_pairs[:,0],ind_pairs[:,1]]
    for c in shared_info:
        shared_dict[c]= shared_dict.get(c,0) +1
    num_shared_labels = sorted([k for k in shared_dict.keys()])
    counts = np.array([ shared_dict[k]  for k in num_shared_labels])
    plt.bar(num_shared_labels,counts/np.sum(counts),edgecolor='w')
    plt.xlabel("shared label counts")
    plt.ylabel("Percent of sample pairs")
    plt.title("Distribution of image pairs for "+ dataset.dataset_name + " "+ dset_type+ " dataset")
    plt.savefig(save_path+'/statistic_shared_labels.png', format='png')
    plt.close()    

def check_recall_histogram(T, T_pred,save_path,bins=10):
    """
    Split the scores of recall into bins, check the frequency for each score interval
    Args:
        score_per_sample: np array, flatten, length: total sample number
        T: original
    """
    y_true = [np.where(multi_hot==1)[0] for multi_hot in T]
    y_pred = [np.where(multi_hot==1)[0] for multi_hot in T_pred]
    score_per_sample = [evaluation.classification.recall(y_t, y_p) for y_t,y_p in zip(y_true,y_pred)]
    
    count,bin_edges = np.histogram(score_per_sample, bins=1.0/bins*np.arange(bins+1))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1.5,1.5])
    langs = [ str(int(bin_edges[i-1]*100))+'%'+'~'+str(int(bin_edges[i]*100)) +'%' for i in range(1,len(bin_edges))]
    ax.bar(langs,count/sum(count),edgecolor='w')
    plt.xlabel("recall")
    plt.ylabel("Percent of eval data")
    plt.title("recall@1 histogram of eval data")
    plt.savefig(save_path, format='png')
    plt.close()

def start_wandb(config):
    import wandb
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_key']
    os.environ["WANDB_MODE"] = "dryrun" # for wandb logging on HPC
    _ = os.system('wandb login --relogin {}'.format(config['wandb']['wandb_key']))
    # store this id to use it later when resuming
    if 'wandb_id' not in config['wandb'].keys():
        config['wandb']['wandb_id']= wandb.util.generate_id()
    wandb.init(id= config['wandb']['wandb_id'], resume="allow",project=config['wandb']['project'], group=config['wandb']['group'], name=config['log']['save_name'], dir=config['log']['save_path'])
    wandb.config.update(config, allow_val_change= True)
    return config

def plot_intra_inter_dist(df, shared_info,save_path):
    sns.set_style('whitegrid')
    print('Start to plot intra and inter dist')
    start_time = time.time()

    #plot and save the distribution of intra distance and inter distance for each class
    plt.figure()
    grid = sns.FacetGrid(df, col='class', hue="dist_type",  col_wrap=5)
    grid.map(sns.kdeplot, 'Distance')
    grid.add_legend()
    grid.savefig(save_path + '/dist_per_class.png',format='png')
    plt.close()

    # save the distribution of all the intra and inter distance
    plt.figure()
    grid = sns.FacetGrid(df, hue="dist_type")
    grid.map(sns.kdeplot, 'Distance')
    grid.add_legend()
    plt.title("Distance distribution")
    grid.savefig(save_path + '/dist_all.png',format='png')
    plt.close()

    #Plot the dist distribution for shared labels
    temp_list = [[[d,key] for d in shared_info[key]] for key in shared_info.keys()]
    temp_list = np.array([item for sublist in temp_list for item in sublist])
    shared_df = pd.DataFrame({"Distance":temp_list[:,0],
                             "labelShared":temp_list[:,1]})
    plt.figure()
    grid = sns.FacetGrid(shared_df, hue="labelShared")
    grid.map(sns.kdeplot, 'Distance')
    grid.add_legend()
    plt.title("Distance distribution of embedding pairs")
    grid.savefig(save_path + '/dist_shared.png',format='png')
    plt.close()

    print("Plot done! Time elapsed: {:.2f} s.\n".format(time.time()- start_time))