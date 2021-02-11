from __future__ import print_function
from __future__ import division

from . import evaluation
from . import similarity
from . import faissext
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import scale,normalize
import matplotlib.pyplot as plt
import PIL
import os
from osgeo import gdal
from sklearn.manifold import TSNE
import time
import random


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
        is_verbose = len(dataloader.dataset) > 0

        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader, desc= desc, disable=not is_verbose):
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


def evaluate_query_gallery(model, config, dl_query, dl_gallery, use_penultimate, backend, LOG=None, log_key = 'Val'):
    """
    Evaluate the retrieval performance (wmap, map, hamming loss)
        Args:
            model: pretrained resnet50
            dl_query: query dataloader
            dl_gallery: gallery dataloader
            use_penultimate: use the embedding layer if it is false
            backend: default faiss-gpu 
    """
    K =  4 
    # calculate embeddings with model and get targets
    X_query, T_query, I_query = predict_batchwise(model, dl_query, config['device'],use_penultimate,desc="Extraction Query Features")
    X_gallery, T_gallery, I_gallery = predict_batchwise(model, dl_gallery, config['device'],use_penultimate,desc='Extraction Gallery Features')

    if 'evaluation_weight' in config.keys():
        X_query = get_weighted_embed(X_query,config['evaluation_weight'],config['sub_embed_sizes'])
        X_gallery = get_weighted_embed(X_gallery,config['evaluation_weight'],config['sub_embed_sizes'])

    # make sure the query and the gallery has same number of classes
    nb_classes = dl_query.dataset.nb_classes()
    assert dl_query.dataset.nb_classes() == dl_gallery.dataset.nb_classes()

    k_closest_points, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query,k= K,gpu_id= torch.cuda.current_device())
    T_query_pred   = T_gallery[k_closest_points]

    score ={}
    wmap = evaluation.retrieval.select('wmap',T_query, T_query_pred, K)
    score['wmap@'+str(K)] = wmap
    if LOG !=None:
        LOG.progress_saver[log_key].log("retrieve_wmap@"+str(K),wmap)
    print("retrieval: wmap@{} : {:.2f}".format(K, wmap))

    map = evaluation.retrieval.select('map',T_query, T_query_pred, K)
    score['map@'+str(K)] = map
    if LOG !=None:
        LOG.progress_saver[log_key].log("retrieve_map@"+str(K),map)
    print("retrieval: map@{} : {:.3f}".format(K, 100*map))

    hl = evaluation.retrieval.select('hamming',T_query, T_query_pred, K)
    score['hamming@'+str(K)] = hl
    if LOG !=None:
        LOG.progress_saver[log_key].log("retrieve_hamming@"+str(K),hl)
    print("retrieval: hamming loss@{} : {:.3f}".format(K, hl))

    ### recover n_closest images
    # n_img_samples = 10
    # n_closest = 4
    # save_path = LOG.config['checkfolder']+'/sample_recoveries.png'
    # recover_query_gallery(X_query,X_gallery,dl_query.dataset.im_paths, dl_gallery.dataset.im_paths, save_path,n_img_samples, n_closest)
    return score

def evaluate_standard(model, config,dl, use_penultimate, backend,LOG=None, log_key = 'Val',with_f1 = True,is_init=False):
    """
    Evaluate the classification performance (recall @1,2,4,8, f1)
        Args:
            model: pretrained resnet50
            dl: dataloader
            use_penultimate: use the embedding layer if it is false
            backend: default faiss-gpu 
    """
    nb_classes = dl.dataset.nb_classes()
    K = [1, 2, 4, 8]
    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dl, config['device'], use_penultimate, desc='Extraction Eval Features')
    if 'evaluation_weight' in config.keys() and not is_init:
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    
    k_closest_points, _ = faissext.find_nearest_neighbors(X, queries= X, k=max(K)+1,gpu_id= torch.cuda.current_device())
    T_pred = T[k_closest_points[:,1:]]

    flag_checkpoint = False
    history_recall1 = 0
    if "recall" in LOG.progress_saver[log_key].groups.keys():
        history_recall1 = np.max(LOG.progress_saver[log_key].groups['recall']["recall@1"]['content'])

    score={}
    # calculate recall @ 1, 2, 4, 8
    for k in K:
        r_at_k = evaluation.classification.select('recall',T, T_pred, k)
        score['recall@'+str(k)] = r_at_k
        print("classification: recall@{} : {:.3f}".format(k, 100 * r_at_k))
        if not is_init:
            LOG.progress_saver[log_key].log("recall@"+str(k),r_at_k,group='recall')
            if k==1 and r_at_k > history_recall1:
                flag_checkpoint = True
    ### calculate f1
    if with_f1:
        f1_k =1
        f1 = evaluation.classification.select('f1',T, T_pred, k=f1_k)
        score['f1@'+str(f1_k)] = f1
        print("classification: f1@{} : {:.3f}".format(f1_k, 100 * f1))
        if not is_init:
            LOG.progress_saver[log_key].log("f1",f1)
    ### save checkpoint #####
    if  flag_checkpoint and not is_init:
        print("Best epoch! save to checkpoint")
        savepath = LOG.config['checkfolder']+'/checkpoint_{}.pth.tar'.format("recall@1")
        torch.save({'state_dict':model.state_dict(), 'opt':config, 'progress': LOG.progress_saver, 'aux':config['device']}, savepath)

        print("Get the inter and intra class distance for different classes")
        DistanceMeasure(X, T, config,LOG,log_key)

        n_img_samples = 10
        n_closest = 4
        save_path = LOG.config['checkfolder']+'/query_sample_recoveries.png'
        print("Recover the {} closest sample images".format(n_closest))
        recover_standard(X,dl.dataset.im_paths, save_path,n_img_samples, n_closest)
    return score

def plot_images(image_paths,save_path):
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


def DistanceMeasure(X, T,config, LOG, log_key):
    """
    log the distance ratios between intra-class and inter-class.
        Args:
            X: embeddings
            T: multi-hot labels
    """
    if 'evaluation_weight' in config.keys():
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    # compute the l2 distance mat of X
    # the diagonals are zeros
    dist = similarity.pairwise_distance(X)

    # get the labels for each embedding
    T_list = [ np.where(t==1)[0] for t in T] 
    T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
    classes = np.unique(T_list[:,1])
    image_dict = {str(c):[] for c in classes}
    [image_dict[str(c)].append(ind) for ind,c in T_list]
    
    # Compute average intra and inter l2 distance
    for label in image_dict.keys():
        inds = image_dict[label] 
        if len(inds)<2: continue
        intra_ind = np.array([[ [i,j] for j in inds if i != j] for i in inds]).reshape(-1,2)
        dist_intra =  dist[intra_ind[:,0],intra_ind[:,1]]
        std_intra = np.std(dist_intra) 
        mean_intra = np.mean(dist_intra)

        other_inds = list(set(range(len(X))) - set(range(len(X))).intersection(inds))
        inter_ind = np.array([[[i,j] for j in other_inds] for i in inds]).reshape(-1,2)
        dist_inter =  dist[inter_ind[:,0],inter_ind[:,1]]
        std_inter = np.std(dist_inter)
        mean_inter = np.mean(dist_inter)
        
        LOG.progress_saver[log_key].log('distRatio@'+str(label),mean_intra/mean_inter, group ='distRatio')
        LOG.progress_saver[log_key].log('intraStd@'+str(label),std_intra, group ='intraStd')
        LOG.progress_saver[log_key].log('interStd@'+str(label),std_inter, group ='interStd')

    
def GradientMeasure(model,LOG,log_key):
    # record the gradient of the weight of each layer in the model
    for name, param in model.named_parameters():
        if param.requires_grad == True and 'weight' in name and param.grad is not None:
            grads = param.grad.detach().cpu().numpy().flatten()
            grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
            LOG.progress_saver[log_key].log(name+'_l2',grad_l2)
            LOG.progress_saver[log_key].log(name+'_max',grad_max)


def classBalancedSamper(T,num_samples_per_class=2):
    """
    Get a list of category labels with its original index from multi-hot labels
        Args:
            T: multi-hot labels, eg.numpy array[n_samples x 60]
            num_samples_per_class: default 2
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


def recover_standard(X, img_paths,save_path, n_img_samples = 10, n_closest = 4):
    """
    Recover the n closest similar images for sampled images
        Args:
            X: embeddings
            img_paths: the original image paths of embeddings
    """
    sample_idxs = np.random.choice(np.arange(len(X)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X, queries= X[sample_idxs],
                                                k=n_closest+1,
                                                gpu_id= torch.cuda.current_device())
    pred_img_paths = np.array([[img_paths[i] for i in ii] for ii in nns[:,1:]])
    sample_paths = [img_paths[i] for i in sample_idxs]
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    plot_images(image_paths,save_path)


def recover_query_gallery(X_query, X_gallery,query_img_paths,gallery_img_path, save_path, n_img_samples = 10, n_closest = 4):
    """
    Recover the n closest similar gallery images for sampled query images
        Args:
            X_query: query embeddings
            X_gallery: gallery embeddings
            query_img_paths: the original image paths of query embeddings
            gallery_img_path: the original image paths of gallery embeddings
    """
    assert X_gallery.shape[1] == X_gallery.shape[1]
    sample_idxs = np.random.choice(np.arange(len(X_query)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query[sample_idxs],
                                                 k=n_closest,
                                                 gpu_id= torch.cuda.current_device()
        )
    pred_img_paths = np.array([[gallery_img_path[i] for i in ii] for ii in nns])
    sample_paths = [query_img_paths[i] for i in sample_idxs]
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    plot_images(image_paths,save_path)


def apply_tsne(X,T,conversion,save_path,n_components=2):
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
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    
    # get the labels for each embedding
    T_list = [ np.where(t==1)[0] for t in T] 
    T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
    classes = np.unique(T_list[:,1])
    image_dict = {str(c):[] for c in classes}
    [image_dict[str(c)].append(ind) for ind,c in T_list]

    # Fixing random state for reproducibility
    np.random.seed(19680801)
    number_of_colors = len(classes)
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    plt.figure(figsize=(16,10))
    for i,c in enumerate(classes):
        c_tsne_result = tsne_results[image_dict[str(c)]]
        plt.scatter(c_tsne_result[:,0], c_tsne_result[:,1], c=colors[i], alpha=0.5,label=conversion[str(c)])
    plt.legend(loc="upper right")
    plt.savefig(save_path, format='png')