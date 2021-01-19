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
        if not band_ds:
            continue
        band_ds = gdal.Open(tif_path,  gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = np.array(raster_band.ReadAsArray()) 
        # interpolate the image to (120,120)
        temp = resize(band_data,(120,120))
        tif_img.append(temp)
    tif_img = np.array(tif_img)
    # Spectral band names to read related GeoTIFF files
    patch_json_path = img_path + '/' + patch_name +  '_labels_metadata.json'
    # get patch label
    with open(patch_json_path, 'rb') as f:
        patch_json = json.load(f)
    original_labels = patch_json['labels']
    return patch_name ,original_labels, tif_img

# hdf_file: hdf5 file record the images
# file_list: record the image paths
def store_hdf(hdf_file, file_list,label_indices):
    image_dict ={}
    count = 0
    while (count < len(file_list)):
        if count==0: data_list = file_list
        else: 
            f_read = h5py.File(hdf_file,'r')
            data_list = [x for x in file_list if x not in list(f_read.keys())]
            f_read.close()
        f = h5py.File(hdf_file,'w')
        pool = multiprocessing.Pool(8)
        results = pool.imap(get_data, (img_path for img_path in data_list))
        for idx,(patch_name,original_labels,img_data) in enumerate(results):
            if len(img_data) == 12 :
                f.create_dataset(patch_name, data=img_data.reshape(-1),compression='gzip',compression_opts=9)
                # record label names
                for label in original_labels:
                    key = label_indices['original_labels'][label]
                    if not key in image_dict.keys():
                        image_dict[key] = []
                    image_dict[key].append(patch_name)
            if (idx+1) % 2000==0: print("processed {0:.0f}%".format((idx+1)/len(data_list)*100))
        pool.close()
        pool.join()
        f.close()
        f_read = h5py.File(hdf_file,'r')
        count = len(list(f_read.keys()))
        f_read.close()
    return image_dict

def Give(datapath):
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

    # create hdf5 file
    # store all the images in hdf5 files to further reduce disk I/O
    hdf_dir = ['/train.hdf5','/val.hdf5','/test.hdf5']
    for i in range(len(hdf_dir)):
        hdf_path = datapath + hdf_dir[i]
        if not Path(hdf_path).exists():
            print("Start to create ", hdf_path," for BigEarthNet")
            image_dict = store_hdf(hdf_path,file_lists[i],label_indices)
            # save image dict to jason file
            json_file = hdf_path.split('.')[0] +'.json'
            with open(json_file, 'w') as json_f:
                json.dump(image_dict, json_f,separators=(",", ":"),allow_nan=False,indent=4)
                print("\ncreate ",json_file)

    
    # read image dict from json file
    data_list =[]
    for i in range(len(hdf_dir)):
        hdf_path = datapath + hdf_dir[i]
        json_file = hdf_path.split('.')[0] +'.json'
        with open(json_file, 'r') as json_f:
            data_list.append(json.load(json_f))
    
    # get the common keys from train/va/test image dict
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
    return {'training': train_image_dict , 'validation': val_image_dict , 'testing': test_image_dict}