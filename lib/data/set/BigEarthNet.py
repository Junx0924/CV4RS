import numpy as np
from pathlib import Path
import json
import csv
import os
import multiprocessing
from skimage.transform import resize
from osgeo import gdal
import h5py

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


# hdf_file: hdf5 file record the images
# file_list: record the image paths
def store_hdf(hdf_file, file_list):
    image_dict ={}
    with h5py.File(hdf_file, "w") as f:
        f = h5py.File(hdf_file,'w')
        pool = multiprocessing.Pool(8)
        results = pool.imap(get_data, (img_path for img_path in file_list))
        for idx,(patch_name,img_data) in enumerate(results):
            if len(img_data) == 12 :
                f.create_dataset(patch_name, data=img_data.reshape(-1),compression='gzip',compression_opts=9)
            if (idx+1) % (len(file_list)//5)==0: print("processed {0:.0f}%".format((idx+1)/len(file_list)*100))
        pool.close()
        pool.join()


def Give(datapath,dset_type, use_hdf5):
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

    if use_hdf5:
        # create hdf5 file
        # store all the images in hdf5 files to further reduce disk I/O
        hdf_dir = ['/train.hdf5','/val.hdf5','/test.hdf5']
        for hdf_file, file_list in zip(hdf_dir, file_lists):
            hdf_path = datapath + hdf_file
            if not Path(hdf_path).exists():
                print("Start to create ", hdf_path," for BigEarthNet")
                store_hdf(hdf_path,file_list)
    
    # load image_dict from json files
    data_list =[]
    json_dir =[ item.split('.')[0] +'.json' for item in csv_list]
    for json_file, file_list in zip(json_dir, file_lists):
        json_path = datapath + json_file 
        if not Path(json_path).exists():
            image_dict = get_label(file_list)
            with open(json_file, 'w') as json_f:
                json.dump(image_dict, json_f,separators=(",", ":"),allow_nan=False,indent=4)
                print("\ncreate ",json_file)
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