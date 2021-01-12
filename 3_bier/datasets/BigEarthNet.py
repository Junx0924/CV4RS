import numpy as np
from pathlib import Path
import json
import csv
import os
import threading
import concurrent.futures
from skimage.transform import resize
from osgeo import gdal

class get_labels(threading.Thread):
    def __init__(self,datapath,patch_name,label_indices):
        threading.Thread.__init__(self)
        self.datapath = datapath
        self.patch_name = patch_name
        self.label_indices = label_indices
    def run(self):
        img_path = self.datapath + '/' + self.patch_name +'.npy'
        if not Path(img_path).exists():
            patch_name = Path(img_path).stem
            band_names = ['B01', 'B02', 'B03', 'B04', 'B05','B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            tif_img = []
            for band_name in band_names:
                tif_path = img_path.split(".")[0]  + '/'+ patch_name+'_'+band_name+'.tif'
                band_ds = gdal.Open(tif_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = np.array(raster_band.ReadAsArray()) 
                # interpolate the image to (120,120)
                temp = resize(band_data,(120,120))
                tif_img.append(temp)
            tif_img = np.array(tif_img)
            np.save(img_path,tif_img)
        # Spectral band names to read related GeoTIFF files
        patch_json_path = self.datapath +'/'+ self.patch_name  + '/' + self.patch_name +  '_labels_metadata.json'
        if Path(patch_json_path).exists():
        # get patch label
            with open(patch_json_path, 'rb') as f:
                patch_json = json.load(f)
            original_labels = patch_json['labels']
            return self.patch_name ,original_labels

def Give(datapath):
    json_dir = os.path.dirname(__file__) + '/BigEarthNet_split'
    json_files = ['/train.json','/val.json','/test.json','/label_name.json']
    if not Path(json_dir + '/train.json').exists():
        print("Start to get labels for patches in BigEarthNet")
        with open(json_dir + '/label_indices.json', 'rb') as f:
            label_indices = json.load(f)
        csv_list =['/train.csv','/val.csv','/test.csv']
        conversion ={}
        dict_list = []
        for i in range(len(csv_list)):
            with open(json_dir + csv_list[i]) as csv_file:
                patch_names =[ row[:-1] for row in csv_file]
            results =[]
            image_dict ={}
            with concurrent.futures.ThreadPoolExecutor(max_workers=opt.kernels) as executor:
                future_list= [executor.submit(get_labels, datapath,patch_name,label_indices) for patch_name in patch_names]
                results = [future.result().run() for future in concurrent.futures.as_completed(future_list)]
                for (patch_name,original_labels) in results:
                    for label in original_labels:
                        key = label_indices['original_labels'][label]
                        conversion[key] = label
                        if not key in image_dict.keys():
                            image_dict[key] = []
                        image_dict[key].append(patch_name)
            dict_list.append(image_dict)
        dict_list.append(conversion)   
        # write the json file to disk
        for i in range(len(json_files)):
            with open(json_dir + json_files[i], 'w') as json_file:
                json.dump(dict_list[i], json_file,separators=(",", ":"),allow_nan=False,indent=4)
                print("\ncreate ",json_dir + json_files[i])
    
    data_list =[]
    for i in range(len(json_files)):
        with open(json_dir + json_files[i], 'r') as json_file:
            data_list.append(json.load(json_file))
    conversion = data_list[3]
    # get the common keys from train/va/test
    keys= [ data.keys() for data in data_list[:3]]
    keys = [ [k for k in i] for i in keys]
    keys = list(set.intersection(*map(set, keys)))
    new_keys = {key:i for i,key in enumerate(keys)} 
    new_conversion = {new_keys[key]:conversion[key] for key in keys}
    
    train_image_dict,val_image_dict,test_image_dict ={},{},{}
    new_data_list = [train_image_dict,val_image_dict,test_image_dict]
    for i in range(len(new_data_list)):
        for key in keys:
            new_data_list[i][new_keys[key]] = [datapath + '/' + patch_name +'.npy' for patch_name in data_list[i][key]]

    # train_dataset = BaseDataset(train_image_dict, is_validation=False)
    # train_dataset.conversion = new_conversion

    # val_dataset = BaseDataset(val_image_dict, is_validation=True)
    # val_dataset.conversion   = new_conversion
    
    # test_dataset  = BaseDataset(test_image_dict, is_validation=True)
    # test_dataset.conversion  = new_conversion
    return {'training':train_image_dict, 'validation':val_image_dict, 'testing':test_image_dict}