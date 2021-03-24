from __future__ import print_function
from __future__ import division
from lib.evaluation.examplebasedclassification import precision

from . import evaluation
from . import similarity
from . import faissext
import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
import PIL
import os
from osgeo import gdal
from sklearn.manifold import TSNE
import time
import json
import vaex as vx


def predict_batchwise(model, dataloader, device,use_penultimate = False, is_dry_run=False,desc=''):
    """
    Get the embeddings of the dataloader from model
        Args:
            model: pretrained resnet50
            dataloader: torch dataloader
            device: torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_penultimate: bool, if set false, use the embedding layer
        return:
            numpy array of embeddings, labels, indexs 
    """
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    A = [[],[],[]]
    if desc =='': desc ='Filling memory queue'
    with torch.no_grad():
        # use tqdm when the dataset is large (SOProducts)
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
            X: numpy array[n_samples X 512], embeddings
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


def start_wandb(config):
    import wandb
    os.environ['WANDB_API_KEY'] = config['wandb']['wandb_key']
    if config['wandb']['dry_run']:
        os.environ["WANDB_MODE"] = "dryrun" # for wandb logging on HPC
    _ = os.system('wandb login --relogin {}'.format(config['wandb']['wandb_key']))
    # store this id to use it later when resuming
    if 'wandb_id' not in config['wandb'].keys():
        config['wandb']['wandb_id']= wandb.util.generate_id()
    wandb.init(id= config['wandb']['wandb_id'], resume="allow",project=config['wandb']['project'], group=config['wandb']['group'], name=config['log']['save_name'], dir=config['log']['save_path'])
    wandb.config.update(config, allow_val_change= True)
    return config

def json_dumps(**kwargs):
    # __repr__ may contain `\n`, json replaces it by `\\n` + indent
    return json.dumps(**kwargs).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)
    
def evaluate_query_gallery(model, config, dl_query, dl_gallery, use_penultimate= False,  
                          LOG=None, log_key = 'Val',is_init=False,K = [1,2,4,8],metrics=['recall'], is_plot_dist= False,is_recover=  False,n_img_samples=4,n_closest=4):
    """
    Evaluate the retrieve performance
    Args:
        model: pretrained resnet50
        dl_query: query dataloader
        dl_gallery: gallery dataloader
        use_penultimate: bool, if set false, use the embedding layer
        K: default [1,2,4,8]
        metrics: default ['recall']
    Return:
        score: dict of score for different metrics
    """
    # calculate embeddings with model and get targets
    X_query, T_query, _ = predict_batchwise(model, dl_query, config['device'],use_penultimate,desc="Extraction Query Features")
    X_gallery, T_gallery, _ = predict_batchwise(model, dl_gallery, config['device'],use_penultimate,desc='Extraction Gallery Features')

    if 'evaluation_weight' in config.keys() and not is_init:
        X_query = get_weighted_embed(X_query,config['evaluation_weight'],config['sub_embed_sizes'])
        X_gallery = get_weighted_embed(X_gallery,config['evaluation_weight'],config['sub_embed_sizes'])
    X_query = normalize(X_query,axis=1)
    X_gallery = normalize(X_gallery,axis=1)
    
    # make sure the query and the gallery has same number of classes
    assert dl_query.dataset.nb_classes() == dl_gallery.dataset.nb_classes()

    k_closest_points, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query,k= max(K),gpu_id= config['gpu_ids'][0])
    T_query_pred   = T_gallery[k_closest_points]

    scores={}
    for k in K:
        y_pred = np.array([ np.sum(y[:k],axis=0) for y in T_query_pred])
        y_pred[np.where(y_pred>1)]= 1
        for metric in metrics:
            if metric !='map':
                s = evaluation.select(metric,T_query,y_pred)
                print("{}@{} : {:.3f}".format(metric, k, s))
                scores[metric+ '@'+str(k)] = s
                if LOG !=None:
                    LOG.progress_saver[log_key].log(metric+ '@'+str(k),s,group=metric)

    if 'map' in metrics or 'r_precision' in metrics:
        for R in K:
            map=[]
            precision=[]
            for i in range(len(T_query)):
                relevant=[]
                precision_r =[]
                s =0.0
                for j in range(R):
                    a = 1 if evaluation.select('precision',T_query[i].reshape(1,-1),T_query_pred[i,j].reshape(1,-1))==1 else 0
                    s += a
                    precision_r.append(s/(j+1))
                    relevant.append(a)
                map.append(np.mean(np.array(relevant)*np.array(precision_r)))
                precision.append(np.sum(relevant)/R)
            m = np.mean(map)
            p = np.mean(precision)
            scores['map'+ '@'+str(R)] = m
            scores['r_precision'+ '@'+str(R)] = p
            print("{}@{} : {:.3f}".format('map', R, m))
            print("{}@{} : {:.3f}".format('r_precision', R, p))
            if LOG !=None:
                LOG.progress_saver[log_key].log('map'+ '@'+str(R),m,group='map')
                LOG.progress_saver[log_key].log('r_precision'+ '@'+str(R),p,group='r_precision')
            
    X_stack = np.vstack((X_query,X_gallery))
    T_stack = np.vstack((T_query,T_gallery))
        
    if is_plot_dist or is_recover:
        if 'result_path' not in config.keys():
            result_path = config['checkfolder'] +'/evaluation_results'
            if not os.path.exists(result_path): os.makedirs(result_path)
            config['result_path'] = result_path
        dset_type = 'init_' + dl_query.dataset.dset_type  if is_init else 'final_' + dl_query.dataset.dset_type
        result_path = config['result_path']+'/'+dset_type
        if not os.path.exists(result_path): os.makedirs(result_path)
        
    # plot inter and intra distance for images which have the most frequent class label
    label_most = int(max(dl_query.dataset.image_dict, key=lambda k: len(dl_query.dataset.image_dict[k])))
    check_intra_inter_dist(X_stack, T_stack, class_label = label_most ,is_plot= is_plot_dist, LOG= LOG, project_name=config['project'], save_path=result_path)

    if is_recover:
        ## recover n_closest images
        retrieve_save_path = result_path+'/recoveries.png'
        counter = 1
        while os.path.exists(retrieve_save_path):
            retrieve_save_path = result_path+'/recoveries_'+str(counter)+'.png'
            counter += 1
        retrieve_query_gallery(X_query, T_query, X_gallery,T_gallery, dl_query.dataset.conversion, dl_query.dataset.im_paths, dl_gallery.dataset.im_paths, retrieve_save_path,n_img_samples, n_closest,gpu_id=config['gpu_ids'][0])
        plot_tsne(X_stack, result_path,config['project'])  
    return scores


def evaluate_standard(model, config,dl, use_penultimate= False, 
                    LOG=None, log_key = 'Val',K = [1,2,4,8],metrics=['recall'], is_init=False,is_plot_dist= False, is_recover=False,n_img_samples=4,n_closest=8):
    """
    Evaluate the retrieve performance
        Args:
            model: pretrained resnet50
            dl: dataloader
            use_penultimate: bool, if set true, use the second last layer of the model
            K: default [1,2,4,8]
            metrics: default ['recall']
            is_init: bool,  get unweighted subembedding vectors for Diva
            is_plot_dist: bool, if set true, plot the inter and intra embedding distance
            is_recover: bool, if set true, retrieve 'n_closest' images for 'n_img_samples' query images
        Return:
            scores: dict of score for different metrics
    """
    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dl, config['device'], use_penultimate, desc='Extraction Eval Features')
    if 'evaluation_weight' in config.keys() and not is_init:
        X = get_weighted_embed(X,config['evaluation_weight'],config['sub_embed_sizes'])
    X = normalize(X,axis=1)
    k_closest_points, _ = faissext.find_nearest_neighbors(X, queries= X, k=max(K)+1,gpu_id= config['gpu_ids'][0])
    # leave itself out
    T_pred = T[k_closest_points[:,1:]]

    scores ={}
    for k in K:
        y_pred = np.array([ np.sum(y[:k],axis=0) for y in T_pred])
        y_pred[np.where(y_pred>1)]= 1
        for metric in metrics:
            if metric !='map':
                s = evaluation.select(metric,T,y_pred)
                print("{}@{} : {:.3f}".format(metric, k, s))
                scores[metric+ '@'+str(k)] = s
                if LOG !=None:
                    LOG.progress_saver[log_key].log(metric+ '@'+str(k),s,group=metric)

    if 'map' in metrics or 'r_precision' in metrics:
        for R in K:
            map=[]
            precision=[]
            for i in range(len(T)):
                relevant=[]
                precision_r =[]
                s =0.0
                for j in range(R):
                    a = 1 if evaluation.select('precision',T[i].reshape(1,-1),T_pred[i,j].reshape(1,-1))==1 else 0
                    s += a
                    precision_r.append(s/(j+1))
                    relevant.append(a)
                map.append(np.mean(np.array(relevant)*np.array(precision_r)))
                precision.append(np.sum(relevant)/R)
            m = np.mean(map)
            p = np.mean(precision)
            scores['map'+ '@'+str(R)] = m
            scores['r_precision'+ '@'+str(R)] = p
            print("{}@{} : {:.3f}".format('map', R, m))
            print("{}@{} : {:.3f}".format('r_precision', R, p))
            if LOG !=None:
                LOG.progress_saver[log_key].log('map'+ '@'+str(R),m,group='map')
                LOG.progress_saver[log_key].log('r_precision'+ '@'+str(R),p,group='r_precision')
                       
    if is_plot_dist or is_recover:
        if 'result_path' not in config.keys():
            result_path = config['checkfolder'] +'/evaluation_results'
            if not os.path.exists(result_path): os.makedirs(result_path)
            config['result_path'] = result_path
        dset_type = 'init_' + dl.dataset.dset_type  if is_init else 'final_' + dl.dataset.dset_type
        result_path = config['result_path']+'/'+dset_type
        if not os.path.exists(result_path): os.makedirs(result_path)

    # log or inter and intra distance for images which have the most frequent class label
    label_most = int(max(dl.dataset.image_dict, key=lambda k: len(dl.dataset.image_dict[k])))
    check_intra_inter_dist(X, T, class_label = label_most ,is_plot= is_plot_dist, LOG= LOG, project_name =config['project'], save_path=result_path)
    
    if is_recover:
        ## recover n_closest images
        retrieve_save_path = result_path+'/recoveries_0.png'
        counter = 1
        while os.path.exists(retrieve_save_path):
            retrieve_save_path = result_path+'/recoveries_'+str(counter)+'.png'
            counter += 1
        retrieve_standard(X,T, dl.dataset.conversion, dl.dataset.im_paths,retrieve_save_path,n_img_samples, n_closest,gpu_id=config['gpu_ids'][0])
        #plot_tsne(X, result_path, config['project']) 
    return scores


def retrieve_standard(X, T, conversion,img_paths,save_path, n_img_samples = 4, n_closest = 4,gpu_id=None):
    """
    Retrieve the n closest similar images for sampled images
        Args:
            X: np.array, embeddings
            T: np.array, multi-hot labels
            conversion: dict, class name for each category label
            img_paths: np.array, the original image paths of embeddings
    """
    print('Start to recover {} similar images for {} sampled image'.format(n_closest,n_img_samples))
    start_time = time.time()
    np.random.seed(0)
    sample_idxs = np.random.choice(np.arange(len(X)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X, queries= X[sample_idxs],
                                                k=n_closest+1,
                                                gpu_id= gpu_id)
    pred_img_paths = np.array([[img_paths[i] for i in ii] for ii in nns[:,1:]])
    sample_paths = [img_paths[i] for i in sample_idxs]
    
    pred_img_labels = np.array([[T[i] for i in ii] for ii in nns[:,1:]])
    sample_labels = [T[i] for i in sample_idxs]
    
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    image_labels = np.concatenate([np.expand_dims(sample_labels,axis=1),pred_img_labels],axis=1)
    plot_retrieved_images(image_paths,image_labels,save_path,conversion)
    print("Recover similar images done! it takes: {:.2f} s.\n".format(time.time()- start_time))


def retrieve_query_gallery(X_query, T_query, X_gallery,T_gallery, conversion,query_img_paths,gallery_img_path, save_path, n_img_samples = 10, n_closest = 4,gpu_id=None):
    """
    Retrieve the n closest similar gallery images for sampled query images
        Args:
            X_query: np.array, query embeddings
            T_query: np.array, multi-hot labels
            X_gallery: np.array, gallery embeddings
            T_gallery: np.array, multi-hot labels
            conversion: dict, class name for each category label
            query_img_paths: np.array, the original image paths of query embeddings
            gallery_img_path: np.array, the original image paths of gallery embeddings
            
    """
    print('Start to recover {} similar gallery images for each sampled query image'.format(n_closest))
    start_time = time.time()
    assert X_gallery.shape[1] == X_gallery.shape[1]
    np.random.seed(0)
    sample_idxs = np.random.choice(np.arange(len(X_query)), n_img_samples)
    nns, _ = faissext.find_nearest_neighbors(X_gallery, queries= X_query[sample_idxs],
                                                 k=n_closest,
                                                 gpu_id= gpu_id
        )
    pred_img_paths = np.array([[gallery_img_path[i] for i in ii] for ii in nns])
    sample_paths = [query_img_paths[i] for i in sample_idxs]
    
    pred_img_labels = np.array([[T_gallery[i] for i in ii] for ii in nns[:,1:]])
    sample_labels = [T_query[i] for i in sample_idxs]
    
    image_paths = np.concatenate([np.expand_dims(sample_paths,axis=1),pred_img_paths],axis=1)
    image_labels = np.concatenate([np.expand_dims(sample_labels,axis=1),pred_img_labels],axis=1)
    plot_retrieved_images(image_paths,image_labels,save_path,conversion)
    print("Recover done! Time elapsed: {:.2f} seconds.\n".format(time.time()- start_time))


def plot_retrieved_images(image_paths,image_labels,save_path,conversion=None):
    """
    Plot images and save them
        Args:
            image_paths: numpy array
            image_labels: numpy array, multi-hot labels
            conversion: dict, class name for each category label
    """
    width = image_paths.shape[1]
    f,axes = plt.subplots(nrows =image_paths.shape[0],ncols=image_paths.shape[1])
    temp_sample_paths = image_paths.flatten()
    temp_sample_labels = [item for sublist in image_labels for item in sublist]
    temp_axes = axes.flatten()
    for i in range(len(temp_sample_paths)):
        plot_path = temp_sample_paths[i]
        ax = temp_axes[i]
        zoom = 1
        if ".png" in plot_path or ".jpg" in plot_path:
            img_data = np.array(PIL.Image.open(plot_path))
            zoom = 0.95
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
            zoom = 2
        # plot the text
        labels =  np.where(temp_sample_labels[i]==1)[0] 
        query_labels =  np.where(temp_sample_labels[int(i/width)*width]==1)[0]
        if i != int(i/width)*width: ax.text(0.01,0.9,str(len(set(labels).intersection(query_labels))) +' correct labels',fontsize=18,color = 'black')
        for j in range(len(labels)): 
            color='black' if labels[j] in query_labels else 'red'
            label_name = conversion[str(labels[j])] 
            if len(label_name)>25: 
                label_name = label_name[:25].rsplit(' ', 1)[0]
            ax.text(0.01, 0.8-j*0.1,str(label_name),fontsize=15, color = color) 
        # plot the image
        imagebox = OffsetImage(img_data, zoom=zoom)
        xy = (0.5, 0.7)
        ab = AnnotationBbox(imagebox, xy,
                            xybox=(0.7, 0.5),
                            xycoords='data')
        ax.add_artist(ab)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
    w_size = 30
    if width ==9: w_size = 55
    f.set_size_inches(w_size,15)
    f.tight_layout()
    f.savefig(save_path)
    plt.close()


def classBalancedSamper(T,num_samples_per_class=2):
    """
    Get a list of category labels with its original index from multi-hot labels
        Args:
            T: np.array[n_samples x 60], multi-hot labels
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

def check_intra_inter_dist(X, T, class_label,is_plot = False,LOG=None, log_key = 'Val', project_name="",save_path=""):
    """
    Plot the inter and intra embedding distance for images which have certain class label
        Args:
            X: np.array[n_samples x 512], embeddings
            T: np.array[n_samples x 60], multi-hot labels
    """
    start_time = time.time()
    n = len(X)
    assert class_label is not None 
    print('Start to calculate the inter and intra embedding distance for images which have class label ' + str(class_label))
    inds = np.where(T[:,class_label]==1)[0]
    other_inds = list(set(range(len(X))) - set(inds))
    m = len(inds)
    # compute the l2 distance for each normalized embedding pairs
    dist = similarity.pairwise_distance(X[inds])
    # only store the uptriangle area without diagonals because they are symmetrical matrix
    dist_intra = np.copy(dist[np.triu_indices(m, k = 1)])
    # free memory
    dist = None
    del dist
    dist_inter = np.sqrt(2 - 2*np.matmul(X[inds],np.transpose(X[other_inds]))).flatten()
    
    # get the number of shared labels
    shared = np.matmul(T[inds],np.transpose(T[inds]))
    shared_intra  = np.copy(shared[np.triu_indices(m, k = 1)])
    # free memory
    shared = None
    del shared
    shared_inter = np.matmul(T[inds],np.transpose(T[other_inds])).flatten()
    
    # pairs which shared this class label
    ds_intra = vx.from_arrays(x =dist_intra ,y=shared_intra)
    ds_temp = vx.from_arrays(x =dist_inter ,y=shared_inter)
    # distance to the images which don't share this class label and other labels
    ds_inter = ds_temp[ds_temp.y ==0]
    
    print("Calculate done! Time elapsed: {:.2f} s.\n".format(time.time()- start_time))
    if LOG !=None and class_label !=None:
        LOG.progress_saver[log_key].log('class@'+str(class_label),float(ds_intra.mean(ds_intra.x))/float(ds_inter.mean(ds_inter.x)), group ='distRatio')
    
    # Plot the dist distribution for shared labels
    if is_plot:
        print('Start to plot the distance density for image pairs which have class label ' + str(class_label))
        start_time = time.time()
        plt.figure()
        ds_inter.plot1d(ds_inter.x, limits='minmax',label='inter', n =True)
        ds_intra.plot1d(ds_intra.x, limits='minmax' ,label='intra', n =True)
        plt.title(project_name)
        plt.xlabel('Embedding pair distance')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(save_path + '/dist_shared_class_' + str(class_label) + '.png',format='png')
        plt.close()
        print("Plot done! Time elapsed: {:.2f} s.\n".format(time.time()- start_time))
      

def plot_tsne(X,save_path,project_name="",n_components=2):
    """
    Get the tsne plot of embeddings
        Args:
            X: np.array[n_samples x 512], embeddings
            T: np.array[n_samples x 60], multi-hot labels
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
    plt.title(project_name + ' tsne of embeddings')
    plt.savefig(save_path +'/tsne.png', format='png')
    plt.close()


def plot_dataset_stat(dataset,save_path):
    """
    Generally check the statistic of the dataset
     Check Avg. num labels per image
     Check Avg. num of labels shared per image
     Args:
        dataset: torch.dataset
    """
    save_path = save_path +'/stat_' + dataset.dset_type
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    images_per_label =[len(dataset.image_dict[label]) for label in dataset.image_dict.keys()]
    plt.figure(figsize=(15,6))
    plt.bar([int(c) for c in dataset.image_dict.keys()] ,height=np.array(images_per_label)/len(dataset.ys))
    plt.xlabel("label")
    plt.ylabel("Percent of samples")
    plt.title("Distribution of samples for "+ dataset.dataset_name + " "+ dataset.dset_type + " dataset")
    plt.savefig(save_path +'/statistic_samples.png')
    plt.close()

    label_counts = np.sum(dataset.ys,axis= 1)
    print("Avg. num labels per image = "+ str(np.mean(label_counts)))
    count_dict = {}
    for item in label_counts:
        count_dict[item] = count_dict.get(item, 0) + 1
    num_labels = sorted([int(k) for k in count_dict.keys()])
    counts = np.array([count_dict[k] for k in num_labels])
    plt.figure()
    plt.bar(num_labels,counts/np.sum(counts),edgecolor='w')
    plt.xlabel("label counts")
    plt.ylabel("Percent of samples")
    plt.title("Distribution of label counts for "+ dataset.dataset_name + " "+ dataset.dset_type+ " dataset")
    plt.savefig(save_path+'/statistic_labels.png')
    plt.close()

    # plot the label share statistic of the class which has the most samples
    label_most = max(dataset.image_dict, key = lambda k: len(dataset.image_dict[k]))
    # get the number of shared labels
    T = np.array(dataset.ys)
    inds = np.where(T[:,int(label_most)]==1)[0]
    T = T[inds]
    shared = np.matmul(T,np.transpose(T))
    # get rid of diagnal elements
    shared = shared * (np.ones((len(inds),len(inds)))- np.diag(np.ones(len(inds))))
    # only get the up triangle area without the diagonals
    shared = shared[np.triu_indices(len(inds), k = 1)]
    print("Avg. num labels shared per image = "+ str(np.mean(shared)))
    hist, bin_edges =np.histogram(shared,bins=num_labels,density=True)
    plt.bar(bin_edges[:len(hist)],hist*100,edgecolor='w')
    plt.xlabel("shared label num")
    plt.ylabel("Percent of sample pairs")
    plt.title("Shared label counts for images which has class label " + dataset.conversion[label_most])
    plt.savefig(save_path+'/statistic_shared_labels.png', format='png')
    plt.close()    
