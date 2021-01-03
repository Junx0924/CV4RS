"""
This script prepares the CUB-200-2011 dataset for BIER.
We assume that the CUB images are stored in the images/ subdirectory.
"""
from PIL import Image
from skimage.transform import resize
import numpy as np
import os
import random
import json
import csv

TARGET_SIZE = 256

def read_csv(csv_filename,datapath):
    """
    Read csv file which contains image_path and its labels

    Args:
        csv_filename : train.csv/test.csv/val.csv file path
        datapath: dataset path
    
    Returns:
        An image dict which the key is the label, the items are the image lists.
    """
    file_list, file_label =[],[]
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            file_list.append(datapath + str(row[0]))
            file_label.append(row[1:])
    file_list = np.array(file_list)
    file_label = np.array(file_label,dtype=int)
    file_image_dict  = {i:file_list[np.where(file_label[:,i]==1)[0]] for i in range(file_label.shape[1])}
    return file_image_dict


def collect_data(patch_path):
    """
    Collects all images from the given directory.

    Args:
        patch_path: The jpg image file path

    Returns:
        A preprocessed image with shape(channels,TARGET_SIZE,TARGET_SIZE)
    """
    pic = Image.open(patch_path)
    if len(pic.size)==2:
        pic = pic.convert('RGB')
    result_img =[]
    temp = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
    for i in range(temp.shape[0]):
        result_img.append(resize(temp[i],(TARGET_SIZE,TARGET_SIZE))) 
    return np.array(result_img)


def main():
    csv_dir = os.path.dirname(__file__) + '/MLRSNet_split'
    datapath = '/media/robin/Intenso/Dataset/MLRSNet'

    with open(csv_dir +'/category.json') as json_file:
        category = json.load(json_file)
    with open(csv_dir +'/label_name.json') as json_file:
        conversion= json.load(json_file)
    train_image_dict = read_csv(csv_dir +'/train.csv',datapath)
    test_image_dict = read_csv(csv_dir +'/test.csv',datapath)
    val_image_dict = read_csv(csv_dir +'/val.csv',datapath)

    # to downsize the train/val data
    random.seed(0)
    train_per_class = 0.3

    all_train_images = []
    all_train_labels = []
    for key in train_image_dict.keys():
        label = key
        temp_list = list(train_image_dict[key])
        k = int(len(temp_list)*train_per_class)
        for patch_path in random.sample(temp_list,k):
            temp = collect_data(patch_path)
            all_train_images.append(temp)
            all_train_labels.append(int(label))

    all_val_images = []
    all_val_labels = []
    for key in val_image_dict.keys():
        label = key
        for patch_path in val_image_dict[key]:
            temp = collect_data(patch_path)
            all_val_images.append(temp)
            all_val_labels.append(int(label))

    all_test_images = []
    all_test_labels = []
    for key in test_image_dict.keys():
        label = key
        for patch_path in test_image_dict[key]:
            temp = collect_data(patch_path)
            all_test_images.append(temp)
            all_test_labels.append(int(label))

    all_train_images = np.array(all_train_images)
    all_train_labels = np.array(all_train_labels)

    all_val_images = np.array(all_val_images)
    all_val_labels = np.array(all_val_labels)

    all_test_images = np.array(all_test_images)
    all_test_labels = np.array(all_test_labels)

    np.save(csv_dir + '/train_images.npy', all_train_images)
    np.save(csv_dir + '/train_labels.npy', all_train_labels)

    np.save(csv_dir + '/val_images.npy', all_val_images)
    np.save(csv_dir + '/val_labels.npy', all_val_labels)

    np.save(csv_dir + '/test_images.npy', all_test_images)
    np.save(csv_dir + '/test_labels.npy', all_test_labels)


if __name__ == '__main__':
    main()
