import numpy as np
import json
import csv
import os
from skimage.transform import resize
from osgeo import gdal
import itertools


def get_label(img_path,label_indices):
    """
    Args:
        img_path
        label_indices: dictionary, {'label_name': 'label_indice'}
    Return:
       category labels for single image
    """
    patch_name = img_path.split('/')[-1]
    patch_json_path = img_path + '/' + patch_name +  '_labels_metadata.json'
    # get patch label
    with open(patch_json_path, 'rb') as f:
        patch_json = json.load(f)
        original_labels = patch_json['labels']
    # record classes for each patch
    category_labels = [int(label_indices[label]) for label in original_labels]
    return category_labels

def Give(datapath,dset_type):
    """
    Args:
        datapath: eg. /scratch/CV4RS/Dataset/MLRSNet
        dset_type: choose from train/val/test
    Return:
        image_list: contains [image_path, multi-hot label]
    """
    csv_dir =  os.path.dirname(__file__) + '/BigEarthNet_split'
    # read label names
    with open(csv_dir + '/label_indices.json', 'rb') as f:
        label_json = json.load(f)
        label_indices = label_json['original_labels']
    label_names = {str(y):x for x,y in label_indices.items()}
    
    # read conversion
    if os.path.exists(csv_dir + '/new_labels.json'):
        with open(csv_dir + '/new_labels.json', 'rb') as f:
            new_conversion = json.load(f)

    # read csv files
    csv_list =['/train.csv','/val.csv','/test.csv']
    image_lists = [[] for i in range(len(csv_list))]
    file_list =[] # record the file path which has no labels
    len_set =[] # record the number of train/va/test images
    for i in range(len(csv_list)):
        csv_path= csv_dir + csv_list[i]
        counter = 0
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                counter +=1
                # check if csv file contains multi-hot labels
                if len(row) >1:  
                    img_path = datapath+ '/'+ row[0]
                    multi_hot = np.array(row[1:],dtype=int)
                    image_lists[i].append([img_path,multi_hot])
                else:
                    file_list.append(row[0])
        len_set.append(counter)

    # add multi-hot labels to csv
    if len(file_list)>0:                
        category_labels = [get_label(datapath+'/'+patch_name,label_indices) for patch_name in file_list]
        classes =[]
        for i, length in enumerate(len_set):
            start = int(sum(len_set[:i]))
            stop = int(start + len_set[i])
            classes.append(np.unique(list(itertools.chain.from_iterable(category_labels[start:stop]))))
        common_class =  list(set.intersection(*map(set, classes)))
        # make the class labels continuous
        new_keys = {str(key):i for i,key in enumerate(common_class)} 
        new_conversion = {new_keys[str(key)]:label_names[str(key)] for key in common_class}
        # save the new labels in json file
        if not os.path.exists(csv_dir + '/new_labels.json'):
            with open(csv_dir + '/new_labels.json', 'w') as fp:
                json.dump(new_conversion, fp)
        # get muli-hot labels
        new_file_list=[]
        new_file_list1=[]
        for i,label in enumerate(category_labels):
            multi_hot = np.zeros(len(common_class),dtype=int)
            # make the label index continuous
            new_label = [new_keys[str(l)] for l in set(common_class).intersection(label)]
            if len(new_label)>0:
                multi_hot[new_label]= 1
                temp = [file_list[i]] + list(multi_hot)
                new_file_list.append(temp)
                new_file_list1.append([datapath+'/'+file_list[i], multi_hot])
            else:
                if i< len_set[0]: len_set[0] = len_set[0]-1
                elif i< len_set[1]: len_set[1] = len_set[1]-1
                else: len_set[2] = len_set[2]-1

        # save multi-hot labels to csv file
        for i, length in enumerate(len_set):
            start = int(sum(len_set[:i]))
            stop = int(start + len_set[i])
            csv_path= csv_dir + csv_list[i]
            with open(csv_path,'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(new_file_list[start:stop])
                image_lists[i] = new_file_list1[start:stop]
    
    dsets = {'train': image_lists[0] , 'val': image_lists[1], 'test': image_lists[2]}
    return dsets[dset_type],new_conversion