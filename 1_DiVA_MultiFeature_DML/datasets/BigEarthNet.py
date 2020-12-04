from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import json
import random

def Give(opt, datapath):
    with open(datapath + '/label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    conversion ={}
    image_dict  = {}
     
    for entry in Path(datapath).iterdir():
        if entry.is_dir()==True:
            patch_name = entry.name
            patch_folder_path = datapath +'/'+ patch_name  
            # count the number of tif files
            a = len([p.suffix for p in Path(patch_folder_path).iterdir() if p.suffix =='.tif'])
            if a == 12:
                patch_json_path = patch_folder_path + '/' + patch_name +  '_labels_metadata.json'
                with open(patch_json_path, 'rb') as f:
                    patch_json = json.load(f)
                original_labels = patch_json['labels']
                for label in original_labels:
                    key = label_indices['original_labels'][label]
                    conversion[key] = label
                    if not key in image_dict.keys():
                        image_dict[key] = []
                    image_dict[key].append( patch_folder_path+ '/'+ patch_name )
            
                
    keys = sorted(list(image_dict.keys()))
    # train/test 50%/50% split balanced in class.
    train_image_dict,test_image_dict ={},{}
    for key in keys:
        random.shuffle(image_dict[key])
        sample_num = len(image_dict[key])
        samples = image_dict[key]
        train_image_dict[key] = samples[:sample_num//2]
        test_image_dict[key] = samples[sample_num//2:]

    # Percentage with which the training dataset is split into training/validation.
    if opt.train_val_split!=1:
        val_image_dict = {}
        for key in keys:
            temp = np.array(train_image_dict[key])
            train_ixs   = np.array(list(set(np.round(np.linspace(0,len(temp)-1,int(len(temp)*opt.train_val_split)))))).astype(int)
            val_ixs     = np.array([x for x in range(len(temp)) if x not in train_ixs])
            train_image_dict[key] = temp[train_ixs]
            val_image_dict[key]   = temp[val_ixs]
        
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
        val_dataset.conversion   = conversion
    else:
        val_dataset = None

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = conversion

    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  = conversion

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = conversion

    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
