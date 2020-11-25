from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import csv
#datapath = 'E:\MLRSNet-master\\'
def Give(opt, datapath):
    label_path = datapath +'\\labels\\'
    image_sourcepath  = datapath +'\\Images\\'
    image_list =[] # image path
    image_labels =[] # image multi-label one hot
    for entry in Path(label_path).iterdir():
        if entry.is_file():
            with open(label_path + entry.name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                image_classes =next(csv_reader,None)[1:] # label names
                for row in csv_reader: 
                    image_path = image_sourcepath + entry.name +'\\'+row[0]
                    image_list.append(image_path)
                    image_labels.append(row[1:])
    
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list =  np.array(image_list)
    image_labels = np.array(image_labels,dtype=int)
    # use image_dict to store label number(0-59), image path
    image_dict  = {}
    for  i range(len(image_classes)):
        key = i
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append( image_list[np.where(image_labels[:,i]==1)[0]] )
    
    keys = sorted(list(image_dict.keys()))
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    if opt.train_val_split!=1:
        if opt.train_val_split_by_class:
            train_val_split = int(len(train)*opt.train_val_split)
            train, val      = train[:train_val_split], train[train_val_split:]
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
        val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
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