from datasets.basic_dataset_scaffold import BaseDataset
import numpy as np
from pathlib import Path
import csv
import random
import xlrd
import itertools
import json

def split(image_dict,split_ratio):
    train_image_dict  = {} 
    other_image_dict  = {} 
    keys = sorted(list(image_dict.keys()))
    values = np.unique(list(itertools.chain.from_iterable(image_dict.values())))
    flag =  {ind:"undefine" for ind in values}

    for key in keys:
        samples_ind = image_dict[key]
        random.shuffle(samples_ind)
        sample_num = len(samples_ind)
        train_image_dict[key] =[]
        other_image_dict[key] =[]
        # check if there are some sample id already in train/nontrain
        for ind in samples_ind:
            if flag[ind] =="undefine":
                if len(train_image_dict[key])< int(sample_num*split_ratio):
                    train_image_dict[key].append(ind)
                    flag[ind] ="train"
                else:
                    if len(other_image_dict[key])< (sample_num - int(sample_num*split_ratio)):
                        other_image_dict[key].append(ind)
                        flag[ind] ="nontrain"
            elif flag[ind] =="train":
                if len(train_image_dict[key])< int(sample_num*split_ratio):
                    train_image_dict[key].append(ind)
            else:
                if len(other_image_dict[key])< (sample_num - int(sample_num*split_ratio)):
                    other_image_dict[key].append(ind)
            
    return train_image_dict,flag

def read_csv(datapath,csv_filename):
    file_list, file_label =[],[]
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            file_list.append(datapath+ str(row[0]))
            file_label.append(row[1:])
    file_list = np.array(file_list)
    file_label = np.array(file_label,dtype=int)
    file_image_dict  = {i:file_list[np.where(file_label[:,i]==1)[0]] for i in range(file_label.shape[1])}
    return file_image_dict

def Give(opt, datapath):
    # check the split train/test/val existed or not
    if not Path('./MLRSNet_split/train.csv').exists():
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
            if entry.suffix ==".csv" :
                with open(label_folder + entry.name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    label_names =next(csv_reader,None)[1:]
                    sort_ind =  np.argsort(label_names)
                    if len(label_names)==60:
                        for row in csv_reader: 
                            image_path = image_folder + entry.stem +'/'+row[0]
                            #image_list.append(image_path)
                            image_list.append('/Images/'+ entry.stem +'/'+row[0])
                            temp = np.array(row[1:])
                            image_labels.append(temp[sort_ind])
                    
        label_names = np.sort(label_names)
        label_names_dict = {} # to record the label names
        label_names_dict= {i:x for i,x in enumerate(label_names)}  

        for key in category.keys():
            labels = category[key]
            label_ind = [str(np.where(label_names==item)[0][0]) for item in labels ]
            category[key] = label_ind

        image_list = np.array(image_list)
        image_labels = np.array(image_labels,dtype=int)
        image_dict  = {i:np.where(image_labels[:,i]==1)[0] for i in range(len(label_names))}

        # split data into train/test 50%/50% balanced in class.
        temp_image_dict,flag_test =split(image_dict, 0.5)
        # split train into train/val 40%/10% balanced in class
        temp_image_dict,flag_val =split(temp_image_dict, 0.8)

        train   = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_val.keys())) if flag_val[ind]=="train"]
        val  = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_val.keys())) if flag_val[ind]=="nontrain"]
        test   = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_test.keys())) if flag_test[ind]=="nontrain"]

        with open('./datasets/MLRSNet_split/label_name.json', 'w+') as label_file:
            json.dump(label_names_dict, label_file,separators=(",", ":"),allow_nan=False,indent=4)
        with open('./datasets/MLRSNet_split/category.json', 'w+') as category_file:
            json.dump(category, category_file,separators=(",", ":"),allow_nan=False,indent=4)
        with open('./datasets/MLRSNet_split/train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train)
        with open('./datasets/MLRSNet_split/test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(test)
        with open('./datasets/MLRSNet_split/val.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(val)
    
    # train/val/test split exists
    with open('./datasets/MLRSNet_split/category.json') as json_file:
            category = json.load(json_file)
    with open('./datasets/MLRSNet_split/label_name.json') as json_file:
        conversion= json.load(json_file)
    train_image_dict = read_csv(datapath,'./datasets/MLRSNet_split/train.csv')
    test_image_dict = read_csv(datapath,'./datasets/MLRSNet_split/test.csv')
    val_image_dict = read_csv(datapath,'./datasets/MLRSNet_split/val.csv')
    

    val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
    val_dataset.conversion   = conversion

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = conversion

    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  =  conversion

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = conversion

    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}