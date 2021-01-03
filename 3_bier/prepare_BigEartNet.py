"""
This script prepares the BigEarthNet dataset for BIER.
We assume that train/val/test json files are existed in BigEarthNet_split.
train/val/test json files contains class_label: patch_names
"""
import numpy as np
from osgeo import gdal
import os
import skimage
from skimage.transform import resize
import json

TARGET_SIZE = 256

def collect_data(patch_path):
    """
    Collects all images from the given patch_name directory.

    Args:
        patch_path: The npy file contain 12 bands

    Returns:
        A list file of preprocessed 12 bands with shape(channels,TARGET_SIZE,TARGET_SIZE)
    """
    if  not os.path.exists(patch_path):
        patch_name = (patch_path.split(".")[0]).split("/")[-1]
        #band_names = ['B01', 'B02', 'B03', 'B04', 'B05','B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        # only use the RGB channel
        band_names = ['B04','B03','B02']
        tif_img = []
        for band_name in band_names:
            tif_path = patch_path.split(".")[0]  + '/'+ patch_name+'_'+band_name+'.tif'
            band_ds = gdal.Open(tif_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = np.array(raster_band.ReadAsArray()) 
            temp = resize(band_data,(TARGET_SIZE,TARGET_SIZE))
            tif_img.append(temp)
        patch_data = np.array(tif_img)
        np.save(patch_path,patch_data)
    else:
        patch_data = np.load(patch_path)
        if len(patch_data)>3:
            temp_data =[]
            temp_data.append(resize(patch_data[3],(TARGET_SIZE,TARGET_SIZE))) # B04 as R
            temp_data.append(resize(patch_data[2],(TARGET_SIZE,TARGET_SIZE))) # B03 as G
            temp_data.append(resize(patch_data[1],(TARGET_SIZE,TARGET_SIZE)))# B02 as B
            patch_data = np.array(temp_data)
    return  patch_data


def main():
    json_dir = os.path.dirname(__file__) + '/BigEarthNet_split'
    datapath = '/media/robin/Intenso/Dataset/BigEarthNet'
    json_files = ['/train.json','/val.json','/test.json','/label_name.json']

    data_list =[]
    for i in range(len(json_files)):
        with open(json_dir + json_files[i], 'r') as json_file:
            data_list.append(json.load(json_file))
    conversion = data_list[3]
    train_image_dict,val_image_dict,test_image_dict ={},{},{}
    new_data_list = [train_image_dict,val_image_dict,test_image_dict]
    #new_conversion_list =[]
    for i in range(len(new_data_list)):
        # make sure the class label is continuous
        keys = data_list[i].keys()
        #new_keys = {key:i for i,key in enumerate(keys)} 
        #new_conversion_list.append({new_keys[key]:conversion[key] for key in keys})
        for key in keys:
            #new_data_list[i][new_keys[key]] = [datapath + '/' + patch_name +'.npy' for patch_name in data_list[i][key]]
            new_data_list[i][key] = [datapath + '/' + patch_name +'.npy' for patch_name in data_list[i][key]]
    
    all_train_images = []
    all_train_labels = []
    for key in train_image_dict.keys():
        label = key
        for patch_path in train_image_dict[key]:
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

    np.save(json_dir + '/train_images.npy', all_train_images)
    np.save(json_dir + '/train_labels.npy', all_train_labels)

    np.save(json_dir + '/val_images.npy', all_val_images)
    np.save(json_dir + '/val_labels.npy', all_val_labels)

    np.save(json_dir + '/test_images.npy', all_test_images)
    np.save(json_dir + '/test_labels.npy', all_test_labels)


if __name__ == '__main__':
    main()
