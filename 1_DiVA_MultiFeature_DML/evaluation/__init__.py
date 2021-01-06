import faiss, matplotlib.pyplot as plt, numpy as np, torch
import PIL
from osgeo import gdal
from sklearn.preprocessing import scale, normalize
#######################
def evaluate(dataset, LOG, metric_computer, dataloaders, model, opt, evaltypes, device,
             aux_store=None, make_recall_plot=False, store_checkpoints=True, log_key='Test'):
    """
    Parent-Function to compute evaluation metrics, print summary string and store checkpoint files/plot sample recall plots.
    """
    if len(dataloaders)==1:
        computed_metrics, extra_infos = metric_computer.compute_standard(opt, model, dataloaders[0], evaltypes, device)
    else:
        computed_metrics, extra_infos = metric_computer.compute_query_gallery(opt,model, dataloaders[0], dataloaders[1], evaltypes, device)

    ###
    full_result_str = ''
    for evaltype in computed_metrics.keys():
        full_result_str += 'Embed-Type: {}:\n'.format(evaltype)
        for i,(metricname, metricval) in enumerate(computed_metrics[evaltype].items()):
            full_result_str += '{0}{1}: {2:4.4f}'.format(' | ' if i>0 else '',metricname, metricval)
        full_result_str += '\n'

    print(full_result_str)


    ###
    for evaltype in evaltypes:
        for storage_metric in opt.storage_metrics:
            parent_metric = evaltype+'_{}'.format(storage_metric.split('@')[0])
            if parent_metric not in LOG.progress_saver[log_key].groups.keys() or \
               computed_metrics[evaltype][storage_metric]>np.max(LOG.progress_saver[log_key].groups[parent_metric][storage_metric]['content']):
               print('Saved {}'.format(parent_metric))
               set_checkpoint(model, opt, LOG.progress_saver, LOG.prop.save_path+'/checkpoint_{}_{}.pth.tar'.format(evaltype, storage_metric), aux=aux_store)


    ###
    recall1 =0.0
    for evaltype in computed_metrics.keys():
        for eval_metric in opt.evaluation_metrics:
            if eval_metric =="recall@1" : recall1 = computed_metrics[evaltype][eval_metric]
            parent_metric = evaltype+'_{}'.format(eval_metric.split('@')[0])
            LOG.progress_saver[log_key].log(eval_metric, computed_metrics[evaltype][eval_metric],  group=parent_metric)


        ###
        if make_recall_plot:
            if len(dataloaders)==1:
                recover_closest_standard(extra_infos[evaltype]['features'],
                                         extra_infos[evaltype]['image_paths'],
                                         LOG.prop.save_path+'/sample_recoveries.png')
            else:
                recover_closest_query_gallery(extra_infos[evaltype]['query_features'],
                                              extra_infos[evaltype]['gallery_features'],
                                              extra_infos[evaltype]['query_image_paths'],
                                              extra_infos[evaltype]['gallery_image_paths'],
                                              LOG.prop.save_path+'/sample_recoveries.png')
    return recall1


###########################
def set_checkpoint(model, opt, progress_saver, savepath, aux=None):
    if 'experiment' in vars(opt):
        import argparse
        save_opt = {key:item for key,item in vars(opt).items() if key!='experiment'}
        save_opt = argparse.Namespace(**save_opt)
    else:
        save_opt = opt

    torch.save({'state_dict':model.state_dict(), 'opt':save_opt, 'progress':progress_saver, 'aux':aux}, savepath)




##########################
def recover_closest_standard(feature_matrix_all, image_paths, save_path, n_image_samples=10, n_closest=3):
    sample_idxs = np.random.choice(np.arange(len(feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(feature_matrix_all.shape[-1])
    faiss_search_index.add(feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(feature_matrix_all, n_closest+1)

    sample_paths = image_paths[closest_feature_idxs][sample_idxs]
    f,axes = plt.subplots(n_image_samples, n_closest+1)

    temp_sample_paths = sample_paths.flatten()
    temp_axes = axes.flatten()
    for i in range(len(temp_sample_paths)):
        plot_path = temp_sample_paths[i]
        ax = temp_axes[i]
        if plot_path.split(".")[1] =="png" or plot_path.split(".")[1] =="jpg":
            img_data = np.array(PIL.Image.open(plot_path))
        else:
            # get RGB channels from the band data of BigEarthNet
            tif_img =[]
            patch_name = (plot_path.split(".")[0]).split("/")[-1]
            for band_name in ['B04','B03','B02']:
                img_path = plot_path.split(".")[0] +'/'+ patch_name+'_'+band_name+'.tif'
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




####### RECOVER CLOSEST EXAMPLE IMAGES #######
def recover_closest_query_gallery(query_feature_matrix_all, gallery_feature_matrix_all, query_image_paths, gallery_image_paths, \
                                  save_path, n_image_samples=10, n_closest=3):
    query_image_paths, gallery_image_paths   = np.array(query_image_paths), np.array(gallery_image_paths)
    sample_idxs = np.random.choice(np.arange(len(query_feature_matrix_all)), n_image_samples)

    faiss_search_index = faiss.IndexFlatL2(gallery_feature_matrix_all.shape[-1])
    faiss_search_index.add(gallery_feature_matrix_all)
    _, closest_feature_idxs = faiss_search_index.search(query_feature_matrix_all, n_closest)

    
    image_paths  = gallery_image_paths[closest_feature_idxs]
    temp = np.expand_dims(query_image_paths,axis=1)
    image_paths  = np.concatenate([temp, image_paths],axis=1)

    sample_paths = image_paths[sample_idxs]

    f,axes = plt.subplots(n_image_samples, n_closest+1)

    temp_sample_paths = sample_paths.flatten()
    temp_axes = axes.flatten()
    for i in range(len(temp_sample_paths)):
        plot_path = temp_sample_paths[i]
        ax = temp_axes[i]
        if plot_path.split(".")[1] =="png" or plot_path.split(".")[1] =="jpg":
            img_data = np.array(PIL.Image.open(plot_path))
        else:
            # get RGB channels from the band data of BigEarthNet
            tif_img =[]
            patch_name = (plot_path.split(".")[0]).split("/")[-1]
            for band_name in ['B04','B03','B02']:
                img_path = plot_path.split(".")[0] +'/'+ patch_name+'_'+band_name+'.tif'
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
