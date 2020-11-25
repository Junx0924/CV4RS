from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import json
#datapath = ""

def Give(opt, datapath):
    with open(datapath + 'label_indices.json', 'rb') as f:
        label_indices = json.load(f)
    
    conversion    = {i:x for x,i in label_indices['original_labels']}

    image_dict  = {}
    for entry in Path(datapath).iterdir():
        if entry.is_dir:
            patch_name = entry.name
            patch_folder_path = datapath +'\\'+ patch_name  
            # count the number of tif files
            a = len([p.suffix for p in Path(patch_folder_path).iterdir() if p.suffix =='.tif'])
            if a == 12:
                patch_json_path = patch_folder_path + '\\' + patch_name +  '_labels_metadata.json'
                with open(patch_json_path, 'rb') as f:
                    patch_json = json.load(f)
                original_labels = patch_json['labels']
                for label in original_labels:
                    key = label_indices['original_labels'][label]
                    if not key in image_dict.keys():
                        image_dict[key] = []
                    image_dict[key].append( patch_folder_path+ '\\'+ patch_name )
                
    keys = sorted(list(image_dict.keys()))
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    if opt.train_val_split!=1:
        if opt.train_val_split_by_class:
            train_val_split = int(len(train)*opt.train_val_split)
            train, val = train[:train_val_split], train[train_val_split:]
            train_image_dict = {key:image_dict[key] for key in train}
            val_image_dict = {key:image_dict[key] for key in val}
            test_image_dict = {key:image_dict[key] for key in test}
        else:
            train_image_dict, val_image_dict = {},{}

            for key in train:
                train_ixs   = np.array(list(set(np.round(np.linspace(0,len(image_dict[key])-1,int(len(image_dict[key])*opt.train_val_split)))))).astype(int)
                val_ixs     = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion   = conversion
    else:
        train_image_dict = {key:image_dict[key] for key in train}
        val_dataset = None

    test_image_dict = {key:image_dict[key] for key in test}



    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion = conversion
    test_dataset.conversion  = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
