import numpy as np
import csv
import random
import xlrd
import itertools
import json
import os
from PIL import Image
 

def split(image_dict,split_ratio):
    """This function extract a new image dict from the given image dict by the split_ratio

    Args:
        image_dict (dict): {'class_label': [image_index1, image_index2, image_index3....]}
        split_ratio (float): eg.0.8, split the image_dict into two image_dicts, the number of samples per class in image_dict1 is 80% of that of the original image dict

    Returns:
        dict: image_dict
        dict: a dict records the state  and the index for each unique image from the original image_dict
    """    
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


def read_csv(csv_filename,datapath):
    """reads a csv file and returns a list of file paths

    Args:
        csv_filename (str): file path, this file contains the image name and its multi-hot labels
        datapath (str): the source of dataset

    Returns:
        list: [image_path, multi-hot label]
    """    
    file_list, file_label =[],[]
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            file_list.append([datapath + str(row[0]),np.array(row[1:],dtype=int)])
    return file_list
 

def create_csv_split(csv_dir,datapath):
    """Split the dataset to train/val/test with ratio 50%/10%/40%
    Keep this ratio among classes
    Write the results to csv files

    Args:
        csv_dir (str): folder to store the resulted csv files
        datapath (str): eg. /scratch/CV4RS/Dataset/MLRSNet
    """    
    category = {}
    category_path = datapath + '/Categories_names.xlsx'
    book = xlrd.open_workbook(category_path)
    sheet = book.sheet_by_index(1)
    for i in range(2,sheet.nrows):
        category_name = sheet.cell_value(rowx=i, colx=1)
        temp_label_name = np.unique(np.array([sheet.cell_value(rowx=i, colx=j).strip() for j in range(2,sheet.ncols) if sheet.cell_value(rowx=i, colx=j)!=""]))
        if "chapparral" in temp_label_name: temp_label_name[np.where(temp_label_name=="chapparral")]= "chaparral"
        category[category_name] = temp_label_name

    label_folder = datapath +'/labels/'
    image_folder  = datapath +'/Images/'
    image_list =[] # image path
    image_labels =[]
        
    for entry in os.listdir(label_folder):
        if entry.split('.')[-1] =="csv" :
            with open(label_folder + entry) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                label_names =next(csv_reader,None)[1:]
                if len(label_names)==60:
                    sort_ind = np.argsort(label_names) 
                    for row in csv_reader: 
                        image_path = image_folder + entry.split('.')[0] +'/'+row[0]
                        #image_list.append(image_path)
                        image_list.append('/Images/'+ entry.split('.')[0] +'/'+row[0])
                        temp = np.array(row[1:])
                        image_labels.append(temp[sort_ind])
                else:
                    print(entry)

    label_names = np.sort(label_names)
    # to record the label names and its id
    label_names_dict= {i:x for i,x in enumerate(label_names)} 

    for key in category.keys():
        labels = np.array(category[key])
        label_ind = [str(np.where(label_names==item)[0][0]) for item in labels]
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
    
    with open(csv_dir +'/label_name.json', 'w+') as label_file:
        json.dump(label_names_dict, label_file,separators=(",", ":"),allow_nan=False,indent=4)
    with open(csv_dir +'/category.json', 'w+') as category_file:
        json.dump(category, category_file,separators=(",", ":"),allow_nan=False,indent=4)
    with open(csv_dir +'/train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train)
    with open(csv_dir +'/test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test)
    with open(csv_dir +'/val.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(val)  

def Give(datapath,dset_type):
    """Given a dataset path generate a list of image paths and multi-hot labels .

    Args:
        datapath (str): eg. /scratch/CV4RS/DatasetBigEarthNet
        dset_type (str): choose from {'train','val','test'}

    Returns:
        list: contains [image_path, multi-hot label]
    """    
    csv_dir = os.path.dirname(__file__) + '/MLRSNet_split'
    # check the split train/test/val existed or not
    if not os.path.exists(csv_dir +'/train.csv'):
        create_csv_split(csv_dir,datapath)
    
    with open(csv_dir +'/category.json') as json_file:
        category = json.load(json_file)
    with open(csv_dir +'/label_name.json') as json_file:
        conversion= json.load(json_file)
    train_list = read_csv(csv_dir +'/train.csv',datapath)
    val_list = read_csv(csv_dir +'/val.csv',datapath)
    test_list= read_csv(csv_dir +'/test.csv',datapath)

    dsets = {'train': train_list , 'val': val_list , 'test': test_list}
    return dsets[dset_type],conversion
