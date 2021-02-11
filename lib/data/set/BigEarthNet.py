import numpy as np
import json
import csv
import os
import multiprocessing
from skimage.transform import resize
from osgeo import gdal

def get_data(img_path):
    patch_name = img_path.split('/')[-1]
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05','B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    # get band image data
    tif_img = []
    for band_name in band_names:
        tif_path = img_path + '/'+ patch_name+'_'+band_name+'.tif'
        band_ds = gdal.Open(tif_path,  gdal.GA_ReadOnly)
        if not band_ds:
            continue
        raster_band = band_ds.GetRasterBand(1)
        band_data = np.array(raster_band.ReadAsArray()) 
        # interpolate the image to (120,120)
        temp = resize(band_data,(120,120))
        tif_img.append(temp)
    tif_img = np.array(tif_img)
    return patch_name, tif_img

def get_label(file_list,label_indices):
    image_dict = {}
    for img_path in file_list:
        patch_name = img_path.split('/')[-1]
        patch_json_path = img_path + '/' + patch_name +  '_labels_metadata.json'
        # get patch label
        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)
        original_labels = patch_json['labels']
        # record label names
        for label in original_labels:
            key = label_indices['original_labels'][label]
            if not key in image_dict.keys():
                image_dict[key] = []
            image_dict[key].append(patch_name)
    return image_dict

def Give(datapath,dset_type):
    csv_dir =  os.path.dirname(__file__) + '/BigEarthNet_split'
    
    # read label names
    with open(csv_dir + '/label_indices.json', 'rb') as f:
        label_indices = json.load(f)
    label_names = {str(y):x for x,y in label_indices['original_labels'].items()}

    # read csv files
    csv_list =['/train.csv','/val.csv','/test.csv']
    file_lists = []
    for i in range(len(csv_list)):
        with open(csv_dir + csv_list[i]) as csv_file:
            patch_path =[ datapath + '/' + row[:-1] for row in csv_file]
        file_lists.append(patch_path)


    # load image_dict from json files
    data_list =[]
    json_dir =[ item.split('.')[0] +'.json' for item in csv_list]
    for json_file, file_list in zip(json_dir, file_lists):
        json_path = datapath + json_file 
        if not os.path.exists(json_path):
            image_dict = get_label(file_list)
            with open(json_file, 'w') as json_f:
                json.dump(image_dict, json_f,separators=(",", ":"),allow_nan=False,indent=4)
                print("\nCreate ",json_file)
        with open(json_path, 'r') as json_f:
            data_list.append(json.load(json_f))
    
    # get the common keys from train/val/test image dict
    keys= [ data.keys() for data in data_list[:3]]
    keys = [ [k for k in i] for i in keys]
    keys = list(set.intersection(*map(set, keys)))
    new_keys = {key:i for i,key in enumerate(keys)} 
    new_conversion = {new_keys[key]:label_names[key] for key in keys}
    
    train_image_dict,val_image_dict,test_image_dict ={},{},{}
    new_dict_list = [train_image_dict,val_image_dict,test_image_dict]
    
    for i in range(len(new_dict_list)):
        for key in keys:
            new_dict_list[i][new_keys[key]] = [datapath + '/' + patch_name for patch_name in data_list[i][key]]
    
    dsets = {'train': train_image_dict , 'val': val_image_dict , 'test': test_image_dict}
    return dsets[dset_type],new_conversion