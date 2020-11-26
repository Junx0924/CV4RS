from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import csv
import random
import xlrd
import itertools


def Give(opt, datapath):
    # get the category names and label names
    category = {}
    category_path = datapath + '/Categories_names.xlsx'
    book = xlrd.open_workbook(category_path)
    sheet = book.sheet_by_index(1)
    for i in range(2,sheet.nrows):
        category_name = sheet.cell_value(rowx=i, colx=1)
        temp_label_name = np.unique(np.array([sheet.cell_value(rowx=i, colx=j).strip() for j in range(2,sheet.ncols) if sheet.cell_value(rowx=i, colx=j)!=""]))
        category[category_name] = temp_label_name
    
    

    label_folder = datapath +'/labels/'
    image_folder  = datapath +'/Images/'
    image_list =[] # image path
    image_labels =[]
    for entry in Path(label_folder).iterdir():
        if entry.suffix ==".csv":
            with open(label_folder + entry.name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                label_names =next(csv_reader,None)[1:]
                sort_ind =  np.argsort(label_names)
                if len(label_names)==60:
                    for row in csv_reader: 
                        image_path = image_folder + entry.stem +'/'+row[0]
                        image_list.append(image_path)
                        temp = np.array(row[1:])
                        image_labels.append(temp[sort_ind])
                
    label_names = np.sort(label_names)
    conversion = {} # to record the label names
    conversion= {i:x for i,x in enumerate(label_names)}  

    image_list = np.array(image_list)
    image_labels = np.array(image_labels,dtype=int)
    # use image_dict to store label number(0-59), image path
    image_dict  = {i:image_list[np.where(image_labels[:,i]==1)[0]] for i in range(len(label_names))}
    
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