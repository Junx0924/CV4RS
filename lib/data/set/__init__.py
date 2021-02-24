from lib.data.set.base import BaseDataset
from lib.data.set import BigEarthNet
from lib.data.set import MLRSNet
import numpy as np
import os
import torch
from tqdm import tqdm

def select(datapath,dset_type,transform,is_training = False,include_aux_augmentations=False, use_npmem = False):
    """
    Get the database for different type of purposes (train/val/test)
    Args:
        datapath: eg. /scratch/CV4RS/Dataset/MLRSNet
        dset_type: choose from {train/val/test}
        transform: dictonary, keys: sz_crop, input_shape
        is_training: if set true, apply random flip and crop for training, else apply center crop
        include_aux_augmentations: if set true, apply rotation to get augumented image data
    Return:
        torch.utils.data.Dataset
    """
    if 'MLRSNet' in datapath:
        image_list,conversion = MLRSNet.Give(datapath,dset_type)
        dataset_name ='MLRSNet'
    if 'BigEarthNet' in datapath:
        image_list,conversion = BigEarthNet.Give(datapath,dset_type)
        dataset_name ='BigEarthNet'

    if use_npmem:
        npmem_file =  datapath + '/'+ dset_type +'.dat'
        if os.path.exists(npmem_file) == False:
            # create npmem file
            print("Start to create " + npmem_file +"\n")
            s = transform['input_shape']
            dataset = BaseDataset(image_list,dataset_name)
            dl = torch.utils.data.DataLoader(
                dataset,
                num_workers= 8,
                shuffle= False,
                pin_memory= True,
                batch_size= 400
                )
            n = len(dl.dataset.im_paths)
            fp = np.memmap(npmem_file, dtype='float32', mode='w+', shape=(n,s[0]*s[1]*s[2]))
            for batch in tqdm(dl):
                img_data, labels, indices = batch 
                for cur_i,i in enumerate(indices):
                    fp[i,:]=img_data[cur_i].reshape(-1)
                fp.flush()
            print("Create " + npmem_file +" success!\n")
    else:
        npmem_file = ""
    
    return BaseDataset(image_list,dataset_name,npmem_file,conversion,transform,is_training,include_aux_augmentations)