from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import json
import random
import csv

# csv_file contains the patch_name
def read_csv(csv_file,datapath,label_indices):
    image_dict = {}
    conversion ={}

    with open(datapath +'/'+csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            patch_name = row[0]
            patch_folder_path = datapath +'/'+ patch_name  
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
    return image_dict,conversion

def Give(opt, datapath):
    with open(datapath + '/label_indices.json', 'rb') as f:
        label_indices = json.load(f)

    train_image_dict,train_conversion = read_csv('train.csv',datapath,label_indices)
    test_image_dict,test_conversion = read_csv('test.csv',datapath,label_indices)

    # Percentage with which the training dataset is split into training/validation.
    if opt.train_val_split!=1:
        val_image_dict,val_conversion = read_csv('val.csv',datapath,label_indices)
        
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
        val_dataset.conversion   = val_conversion
    else:
        val_dataset = None

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = train_conversion

    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  = test_conversion

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = train_conversion

    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
