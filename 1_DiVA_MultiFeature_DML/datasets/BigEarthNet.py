from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import json
import csv
import os
import threading
import concurrent.futures

class get_labels(threading.Thread):
    def __init__(self,datapath,patch_name,label_indices):
        threading.Thread.__init__(self)
        self.datapath = datapath
        self.patch_name = patch_name
        self.label_indices = label_indices
    def run(self):
        # Spectral band names to read related GeoTIFF files
        patch_json_path = self.datapath +'/'+ self.patch_name  + '/' + self.patch_name +  '_labels_metadata.json'
        if Path(patch_json_path).exists():
        # get patch label
            with open(patch_json_path, 'rb') as f:
                patch_json = json.load(f)
            original_labels = patch_json['labels']
            return self.patch_name ,original_labels

def Give(opt, datapath):
    json_dir = os.path.dirname(__file__) + '/BigEarthNet_split'
    json_files = ['/train.json','/val.json','/test.json','/label_name.json']
    if not Path(json_dir + '/train.json').exists():
        print("Start to get labels for patches in BigEarthNet")
        with open(json_dir + '/label_indices.json', 'rb') as f:
            label_indices = json.load(f)
        csv_list =['/train.csv','/val.csv','/test.csv']
        conversion ={}
        dict_list = []
        for i in range(len(csv_list)):
            with open(json_dir + csv_list[i]) as csv_file:
                patch_names =[ row[:-1] for row in csv_file]
            results =[]
            image_dict ={}
            with concurrent.futures.ThreadPoolExecutor(max_workers=opt.kernels) as executor:
                future_list= [executor.submit(get_labels, datapath,patch_name,label_indices) for patch_name in patch_names]
                results = [future.result().run() for future in concurrent.futures.as_completed(future_list)]
                for (patch_name,original_labels) in results:
                    for label in original_labels:
                        key = label_indices['original_labels'][label]
                        conversion[key] = label
                        if not key in image_dict.keys():
                            image_dict[key] = []
                        image_dict[key].append(patch_name)
            dict_list.append(image_dict)
        dict_list.append(conversion)   
        # write the json file to disk
        for i in range(len(json_files)):
            with open(json_dir + json_files[i], 'w') as json_file:
                json.dump(dict_list[i], json_file,separators=(",", ":"),allow_nan=False,indent=4)
                print("\ncreate ",json_dir + json_files[i])
    
    data_list =[]
    for i in range(len(json_files)):
        with open(json_dir + json_files[i], 'r') as json_file:
            data_list.append(json.load(json_file))
    conversion = data_list[3]
    train_image_dict,val_image_dict,test_image_dict ={},{},{}
    new_data_list = [train_image_dict,val_image_dict,test_image_dict]
    new_conversion_list =[]
    for i in range(len(new_data_list)):
        # make sure the class label is continuous
        keys = data_list[i].keys()
        new_keys = {key:i for i,key in enumerate(keys)} 
        new_conversion_list.append({new_keys[key]:conversion[key] for key in keys})
        for key in keys:
            new_data_list[i][new_keys[key]] = [datapath + '/' + patch_name +'.npy' for patch_name in data_list[i][key]]

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = new_conversion_list[0]

    val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
    val_dataset.conversion   = new_conversion_list[1]
    
    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  = new_conversion_list[2]

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = new_conversion_list[0]

    # for deep cluster feature
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    eval_train_dataset.conversion  =  new_conversion_list[0]


    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
