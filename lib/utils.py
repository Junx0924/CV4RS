from __future__ import print_function
from __future__ import division

from . import evaluation
from . import similarity
from . import faissext
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import scale,normalize
import matplotlib.pyplot as plt, numpy as np, torch
import PIL
import os
from osgeo import gdal
import scipy

def predict_batchwise(model, dataloader, device,use_penultimate = False, is_dry_run=False,desc=''):
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
    assert len(weights) == len(sub_dim)
    for i in range(len(sub_dim)):
        start = int(sum(sub_dim[:i]))
        stop = int(start + sub_dim[i])
        X[:, start:stop] = weights[i]*X[:, start:stop]
    # L2 normalize weighted X   
    # X = normalize(X,axis= 1)
    return X


def evaluate_query_gallery(model,  config, dl_query, dl_gallery, use_penultimate, backend, LOG, log_key = 'Val',with_nmi = False,init_eval =False):
    K = [1, 2, 4, 8]
    # calculate embeddings with model and get targets
    X_query, T_query, I_query = predict_batchwise(model, dl_query, config['device'],use_penultimate,desc="Extraction Query Features")
    X_gallery, T_gallery, I_gallery = predict_batchwise(model, dl_gallery, config['device'],use_penultimate,desc='Extraction Gallery Features')

    if 'evaluation_weight' in config.keys() and not init_eval:
        X_query = get_weighted_embed(X_query,config['evaluation_weight'],config['sub_embed_sizes'])
        X_gallery = get_weighted_embed(X_gallery,config['evaluation_weight'],config['sub_embed_sizes'])

    nb_classes = dl_query.dataset.nb_classes()
    assert dl_query.dataset.nb_classes() == dl_gallery.dataset.nb_classes()
    
    # calculate full similarity matrix, choose only first `len(X_query)` rows
    # and only last columns corresponding to the column
    T_eval = torch.cat([torch.from_numpy(np.array(T_query,dtype = int)), torch.from_numpy(np.array(T_gallery, dtype = int))])
    X_eval = torch.cat([torch.from_numpy(X_query), torch.from_numpy(X_gallery)])
    D_check = similarity.pairwise_distance(X_eval)[:len(X_query), len(X_query):]
    D_check = torch.from_numpy(D_check)
    # get top k labels with smallest (`largest = False`) distance
    T_query_pred = T_gallery[D_check.topk(k = max(K), dim = 1, largest = False)[1]]

    flag_checkpoint = False
    history_recall1 = 0
    if "recall" not in LOG.progress_saver[log_key].groups.keys():
        flag_checkpoint = True
    else: 
        history_recall1 = np.max(LOG.progress_saver[log_key].groups['recall']["recall@1"]['content'])

    scores = {}
    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T_query, T_query_pred, k)
        recall.append(r_at_k)
        LOG.progress_saver[log_key].log("recall@"+str(k),r_at_k,group='recall')
        print("eval data: recall@{} : {:.3f}".format(k, 100 * r_at_k))
        if k==1 and r_at_k > history_recall1:
            flag_checkpoint = True
        
    scores['recall'] = recall

    ### save checkpoint #####
    if  flag_checkpoint:
        savepath = LOG.config['checkfolder']+'/checkpoint_{}.pth.tar'.format("recall@1")
        torch.save({'state_dict':model.state_dict(), 'opt':config, 'progress': LOG.progress_saver, 'aux':config['device']}, savepath)

    # if with_nmi:
    #     # calculate NMI with kmeans clustering
    #     nmi = evaluation.calc_normalized_mutual_information(
    #         T_eval.numpy(),
    #         similarity.cluster_by_kmeans(
    #             X_eval.numpy(), nb_classes, backend=backend
    #         )
    #     )
    #     LOG.progress_saver[log_key].log("nmi",nmi)
    #     print("NMI: {:.3f}".format(nmi * 100))
    #     scores['nmi'] = nmi
    # recover_closest_query_gallery( X_query,X_gallery,
    #                                 dl_query.dataset.im_paths,
    #                                 dl_gallery.dataset.im_paths,
    #                                 LOG.config['checkfolder']+'/sample_recoveries.png',
    #                                 n_image_samples=10, n_closest=4
    #                                 )
    return scores

def evaluate_standard(model, config,dl, use_penultimate, backend,LOG, log_key = 'Val', with_nmi = False):
    nb_classes = dl_train.dataset.nb_classes()
    K = [1, 2, 4, 8]
    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dl, use_penultimate, desc='Extraction Eval Features')
    if 'evaluation_weight' in config.keys():
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    scores = {}

    # calculate NMI with kmeans clustering
    # if with_nmi:
    #     nmi = evaluation.calc_normalized_mutual_information(
    #         T,
    #         similarity.cluster_by_kmeans(
    #             X, nb_classes, backend=backend
    #         )
    #     )
    #     logging.info("NMI: {:.3f}".format(nmi * 100))
    #     scores['nmi'] = nmi

    # get predictions by assigning nearest 8 neighbors with euclidian
    assert np.max(K) <= 8, ("Sorry, this is hardcoded here."
                " You would need to retrieve > 8 nearest neighbors"
                            " to calculate R@k with k > 8")
    T_pred = similarity.assign_by_euclidian_at_k(X, T, 8, backend=backend)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in K:
        r_at_k = evaluation.calc_recall_at_k(T, T_pred, k)
        recall.append(r_at_k)
        print("train data: recall@{} : {:.3f}".format(k, 100 * r_at_k))
        LOG.progress_saver[log_key].log("recall@"+str(k),r_at_k,group='recall')

    scores['recall'] = recall
    recover_closest_standard(X, dl.dataset.im_paths, 
                             LOG.config['checkfolder']+'/sample_recoveries.png',
                             n_image_samples=10, n_closest=4
                            )
    return scores



    ####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_query_gallery(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, \
                                  save_path, n_image_samples=10, n_closest=4):
    query_image_paths, gallery_image_paths   = np.array(query_image_paths), np.array(gallery_image_paths)
    sample_idxs = np.random.choice(np.arange(len(query_feature_matrix_all)), n_image_samples)

    nns, _ = faissext.find_nearest_neighbors(gallery_feature_matrix_all, queries= query_feature_matrix_all[sample_idxs],
                                                 k=n_closest,
                                                 gpu_id= torch.cuda.current_device()
        )
    
    image_paths = np.array([[gallery_image_paths[i] for i in ii] for ii in nns])
    sample_paths = query_image_paths[sample_idxs]
    temp = np.expand_dims(sample_paths,axis=1)
    image_paths  = np.concatenate([temp, image_paths],axis=1)

    f,axes = plt.subplots(n_image_samples, n_closest+1)

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
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

##########################
def recover_closest_standard(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=4):
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    nns, _ = faissext.find_nearest_neighbors(feature_matrix_all, queries= feature_matrix_all[sample_idxs],
                                                 k=n_closest,
                                                 gpu_id= torch.cuda.current_device()
        )
    image_paths = np.array([[image_paths[i] for i in ii] for ii in nns])
    sample_paths = image_paths[sample_idxs]
    temp = np.expand_dims(sample_paths,axis=1)
    image_paths  = np.concatenate([temp, image_paths],axis=1)

    f,axes = plt.subplots(n_image_samples, n_closest+1)

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
        if i%(n_closest+1):
            ax.axvline(x=0, color='g', linewidth=13)
        else:
            ax.axvline(x=0, color='r', linewidth=13)
    f.set_size_inches(10,20)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()

def DistanceMeasure(model,config,dataloader,LOG, log_key):
    """
    log the change of distance ratios
    between intra-class distances and inter-class distances.
    X: embedding lists
    label_dict: for each class contains all the indices belong to the same class

    """
    print("Start to evaluate the distance ratios between intra-class and inter-class")
    image_dict = dataloader.dataset.image_dict
    X, _,_ = predict_batchwise(model,dataloader,config['device'])
    if 'evaluation_weight' in config.keys():
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    #Compute average intra-class distance and center feature of each class.
    common_X , intra_dist =[],[]
    for label in image_dict.keys():
        inds = [ item[-1] for item in image_dict[label]]
        dists = scipy.spatial.distance.cdist(X[inds],X[inds],'cosine')
        dists = np.sum(dists)/(len(dists)**2-len(dists))
        x   = normalize(np.mean(X[inds],axis=0).reshape(1,-1)).reshape(-1)
        intra_dist.append(dists)
        common_X.append(x)
    
    # mean intra-class distance
    mean_intra_dist = np.mean(intra_dist)

    #Compute mean inter-class distances by common_X
    mean_inter_dist = scipy.spatial.distance.cdist(np.array(common_X),np.array(common_X),'cosine')
    mean_inter_dist = np.sum(mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

    LOG.progress_saver[log_key].log('inter',mean_inter_dist,group='cosine_dist')
    LOG.progress_saver[log_key].log('intra',mean_intra_dist,group='cosine_dist')

def GradientMeasure(model,LOG,log_key):
    # record the gradient of the weight of each layer in the model
    for name, param in model.named_parameters():
        if param.requires_grad == True and 'weight' in name and param.grad is not None:
            grads = param.grad.detach().cpu().numpy().flatten()
            grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
            LOG.progress_saver[log_key].log(name+'_l2',grad_l2)
            LOG.progress_saver[log_key].log(name+'_max',grad_max)