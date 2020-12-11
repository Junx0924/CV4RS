from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import json
import random
import csv
import os


def read_csv(csv_filename,datapath,label_indices):
    image_dict = {}
    conversion ={}

    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            patch_name = row[0]
            patch_folder_path = datapath +'/'+ patch_name  
            patch_json_path = patch_folder_path + '/' + patch_name +  '_labels_metadata.json'
            if Path(patch_json_path).exists():
                with open(patch_json_path, 'rb') as f:
                    patch_json = json.load(f)
                original_labels = patch_json['labels']
                for label in original_labels:
                    key = label_indices['original_labels'][label]
                    conversion[key] = label
                    if not key in image_dict.keys():
                        image_dict[key] = []
                    #image_dict[key].append( patch_folder_path+ '/'+ patch_name )
                    image_dict[key].append(patch_name)
        
        keys = sorted(list(conversion.keys()))
        labelname_dict ={i:conversion[i] for i in keys}
    return image_dict,labelname_dict

def Give(opt, datapath):
    json_dir = os.path.dirname(__file__) + '/BigEarthNet_split'
    if Path(json_dir + '/train.json').exists():
        with open(json_dir +'/label_name.json') as json_file:
            conversion= json.load(json_file)
        with open(json_dir +'/train.json') as json_file:
            train_image_dict= json.load(json_file)
        with open(json_dir +'/test.json') as json_file:
            test_image_dict= json.load(json_file)
        with open(json_dir +'/val.json') as json_file:
            val_image_dict= json.load(json_file)

    else:
        with open(json_dir + '/label_indices.json', 'rb') as f:
            label_indices = json.load(f)
        train_image_dict,conversion = read_csv(json_dir +'/train.csv',datapath,label_indices)
        test_image_dict,conversion = read_csv(json_dir +'/test.csv',datapath,label_indices)
        val_image_dict,conversion = read_csv(json_dir +'/val.csv',datapath,label_indices)   

        # write the json file to disk
        with open(json_dir+'/label_name.json', 'w') as conversion_file:
            json.dump(train_conversion, conversion_file,separators=(",", ":"),allow_nan=False,indent=4)
        print("create label_name.json\n")
        with open(json_dir+'/train.json', 'w') as train_file:
            json.dump(train_image_dict, train_file,separators=(",", ":"),allow_nan=False,indent=4)
        print("create train.json\n")
        with open(json_dir+'/test.json', 'w') as test_file:
            json.dump(test_image_dict, test_file,separators=(",", ":"),allow_nan=False,indent=4)
        print("create test.json\n")
        with open(json_dir+'/val.json', 'w') as val_file:
            json.dump(val_image_dict, val_file,separators=(",", ":"),allow_nan=False,indent=4)
        print("create val.json\n")
        

    val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
    val_dataset.conversion   = conversion
  

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = conversion

    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  = conversion

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = conversion

    # for deep cluster feature
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    eval_train_dataset.conversion  = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
