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


def evaluate_query_gallery(model, config, dl_query, dl_gallery, use_penultimate, backend, 
                          LOG=None, log_key = 'Val',is_init=False,K = [1,2,4,8],metrics=['recall'], recover_image =False):
    """
    Evaluate the retrieve performance
    Args:
        model: pretrained resnet50
        dl_query: query dataloader
        dl_gallery: gallery dataloader
        use_penultimate: use the embedding layer if it is false
        backend: default faiss-gpu 
        K: [1,2,4,8]
        metrics: default ['recall']
        recover_image: recover sampled image and its retrieved image
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

    k_closest_points, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query,k= max(K),gpu_id= torch.cuda.current_device())
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

    # check the intra and inter dist distribution
    intra_dist, inter_dist, labels = check_distance_ratio(X_query,T_query)
    plot_intra_inter_dist(intra_dist, inter_dist, labels, config['checkfolder'])

    if recover_image:
        ## recover n_closest images
        n_img_samples = 10
        n_closest = 4
        recover_save_path = config['checkfolder']+'/sample_recoveries.png'
        recover_query_gallery(X_query,X_gallery,dl_query.dataset.im_paths, dl_gallery.dataset.im_paths, recover_save_path,n_img_samples, n_closest)
        tsne_save_path =  config['checkfolder']+'/tsne.png'
        check_tsne_plot( X_query,T_query, dl_query.dataset.conversion,tsne_save_path)
        #check_tsne_plot(np.vstack((X_query,X_gallery)), np.vstack((T_query,T_gallery)), dl_query.dataset.conversion, config['checkfolder']+'/' + config['project'] +'/_tsne.png')  
    
    if 'Mirco_F1' in metrics:
        y_pred = np.array([ np.sum(y[:1], axis =0) for y in T_query_pred])
        TP, FP, TN, FN = evaluation.functions.multilabelConfussionMatrix(T_query,y_pred)
        save_path = config['checkfolder']+'/CSV_Logs/confussionMatrix.csv'
        with open(save_path,'w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['TP']+list([int(i) for i in TP]))
            writer.writerow(['FP']+list([int(i) for i in FP]))
            writer.writerow(['TN']+list([int(i) for i in TN]))
            writer.writerow(['FN']+list([int(i) for i in FN]))
    return scores


def evaluate_standard(model, config,dl, use_penultimate, backend,
                    LOG=None, log_key = 'Val',is_init=False,K = [1,2,4,8],metrics=['recall'],recover_image= False):
    """
    Evaluate the retrieve performance
        Args:
            model: pretrained resnet50
            dl: dataloader
            use_penultimate: use the embedding layer if it is false
            backend: default faiss-gpu
        Return:
            scores: dict of score for different metrics
    """
    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dl, config['device'], use_penultimate, desc='Extraction Eval Features')
    if 'evaluation_weight' in config.keys() and not is_init:
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    
    k_closest_points, _ = faissext.find_nearest_neighbors(X, queries= X, k=max(K)+1,gpu_id= torch.cuda.current_device())
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

    print("Get the inter and intra class distance for different classes")
    check_distance_ratio(X, T,LOG,log_key)
    
    if recover_image:
        ## recover n_closest images
        n_img_samples = 10
        n_closest = 4
        save_path = config['checkfolder']+'/sample_recoveries.png'
        recover_standard(X,dl.dataset.im_paths,save_path,n_img_samples, n_closest)
        check_tsne_plot(X,T, dl.dataset.conversion, config['checkfolder']+'/tsne.png') 

    if 'Mirco_F1' in metrics:
        y_pred = np.array([np.sum(y[:1], axis =0) for y in T_pred])
        TP, FP, TN, FN = evaluation.functions.multilabelConfussionMatrix(T,y_pred)
        save_path = config['checkfolder']+'/CSV_Logs/confussionMatrix.csv'
        with open(save_path,'w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(['TP']+list(TP))
            writer.writerows(['FP']+list(FP))
            writer.writerows(['TN']+list(TN))
            writer.writerows(['FN']+list(FN))  
    return scores


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
    plot_recovered_images(image_paths,save_path)


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
    plot_recovered_images(image_paths,save_path)


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


def check_distance_ratio(X, T, LOG=None, log_key="Val"):
    """
    log the distance ratios between intra-class and inter-class.
        Args:
            X: embeddings
            T: multi-hot labels
    """
    # compute the l2 distance mat of X
    # the diagonals are zeros
    start_time = time.time()
    
    dist = similarity.pairwise_distance(X)
    # get the labels for each embedding
    T_list = [ np.where(t==1)[0] for t in T] 
    T_list = np.array([[i,item] for i,sublist in enumerate(T_list) for item in sublist])
    classes = np.unique(T_list[:,1])
    embed_dict = {str(c):[] for c in classes}
    [embed_dict[str(c)].append(ind) for ind,c in T_list]
    labels = [label for label in embed_dict.keys()]

    # Compute average intra and inter l2 distance
    all_intra, all_inter, all_labels =[],[],[]
    for i in range(len(labels)):
        label = labels[i]
        all_labels.append(label)
        inds = embed_dict[label] 
        if len(inds)<2: continue
        intra_ind = np.array([[ [i,j] for j in inds if i != j] for i in inds]).reshape(-1,2)
        dist_intra =  dist[intra_ind[:,0],intra_ind[:,1]]
        all_intra.append(dist_intra)
        mean_intra = np.mean(dist_intra)

        other_inds = list(set(range(len(X))) - set(range(len(X))).intersection(inds))
        inter_ind = np.array([[[i,j] for j in other_inds] for i in inds]).reshape(-1,2)
        dist_inter =  dist[inter_ind[:,0],inter_ind[:,1]]
        all_inter.append(dist_inter)
        mean_inter = np.mean(dist_inter)
        if LOG !=None:
            LOG.progress_saver[log_key].log('distRatio@'+str(label),mean_intra/mean_inter, group ='distRatio')
    print("Calculate dist ratio takes: {:.2f} s.\n".format(time.time()- start_time))
    return all_intra, all_inter,all_labels

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


def check_tsne_plot(X,T,conversion,save_path,n_components=2):
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
    plt.close()

def check_image_label(dataset,save_path):
    """
    Generally check if the datasets have similar distribution
     Check Avg. num labels per image
     Check Avg. num of labels shared per image
     Args:
        dataset: torch.dataset
    """
    avg_labels_per_image = np.mean(np.sum(dataset.ys,axis= 1))
    print("Avg. num labels per image = "+ str(avg_labels_per_image) +'\n')
    images_per_label =[len(dataset.image_dict[label]) for label in dataset.image_dict.keys()]
    plt.figure()
    plt.bar(range(len(images_per_label)) ,height=images_per_label)
    plt.xlabel("labels")
    plt.ylabel("Number of images")
    plt.title("Distribution of images among labels")
    plt.savefig(save_path)
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

def plot_intra_inter_dist(intra_dist, inter_dist, labels,save_path,conversion= None):
    import seaborn as sns
    sns.set_style('whitegrid')

    new_save_path = save_path + '/dist_per_class.png'
    n = 4
    m = len(labels)//n if len(labels)%n ==0 else len(labels)//n+1
    plt.figure()
    fig, axes = plt.subplots(m, n, sharex='row', sharey='col')
    temp_axes = axes.flatten()

    all_intra, all_inter =[],[]
    #plot and save the distribution of intra distance and inter distance for each class
    for i in range(len(labels)):
        label = labels[i]
        dist_intra = intra_dist[i]
        all_intra = all_intra + list(dist_intra)
        dist_inter = inter_dist[i]
        all_inter = all_inter + list(dist_inter)
        class_name = conversion[str(label)] if conversion !=None else str(label)
        ax = temp_axes[i]
        ax.set_title('class_'+class_name)
        sns.kdeplot(np.array(dist_intra), bw=0.5, label ='Intra',ax = ax)
        sns.kdeplot(np.array(dist_inter), bw=0.5, label= 'Inter',ax = ax)
    fig.suptitle("Embedding distance for each class")
    fig.set_size_inches(10,20)
    fig.tight_layout()
    fig.savefig(new_save_path,format='png')
    plt.close()

    # save the distribution of all the intra and inter distance
    # new_save_path = save_path + '/dist_all.png'
    # plt.figure()
    # sns.kdeplot(np.array(all_intra), bw=0.5, label ='Intra-class')
    # sns.kdeplot(np.array(all_inter), bw=0.5, label= 'Inter-class')
    # plt.xlabel("Distance")
    # plt.ylabel("Distribution")
    # plt.title("Embedding distance distribution")
    # plt.savefig(new_save_path,format='png')
    # plt.close()