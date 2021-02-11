from lib.data.set.base import BaseDataset
from lib.data.set import BigEarthNet
from lib.data.set import MLRSNet
import numpy as np
import os
import torch
from tqdm import tqdm
import h5py

def select(datapath,dset_type,transform,is_training = False,include_aux_augmentations=False, use_hdf5 = False):
    types = {"train":"train","query":"val","gallery":"test"}
    dset_type = types[dset_type]

    if 'MLRSNet' in datapath:
        image_list,conversion = MLRSNet.Give(datapath,dset_type)
        dataset_name ='MLRSNet'
    if 'BigEarthNet' in datapath:
        image_list,conversion = BigEarthNet.Give(datapath,dset_type)
        dataset_name ='BigEarthNet'

    if use_hdf5:
        hdf_file =  datapath + '/'+ dset_type +'.hdf5'
        if os.path.exists(hdf_file) == False:
            # create hdf5 file
            dataset = BaseDataset(image_list,dataset_name)
            dl = torch.utils.data.DataLoader(
                dataset,
                num_workers= 8,
                shuffle= False,
                pin_memory= True,
                batch_size= 128
                )
            with h5py.File(hdf_file, "w") as f:
                for batch in tqdm(dl, desc="Start to create hdf5 for "+dset_type):
                    img_data, labels, indices = batch 
                    for cur_i,i in enumerate(img_data):
                        patch_name = dl.dataset.im_paths[i].split('/')[-1]
                        f.create_dataset(patch_name, data=img_data[cur_i],compression='gzip',compression_opts=9)
            print("Create " + hdf_file +" success!\n")
    else:
        hdf_file = ""
    
    return BaseDataset(image_list,dataset_name,hdf_file,conversion,transform,is_training,include_aux_augmentations)