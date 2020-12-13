from datasets.basic_dataset_scaffold import BaseDataset
from PIL import Image
from osgeo import gdal
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
import numpy as np
from pathlib import Path
import json
import random
import csv
import os
import threading
import concurrent.futures

class preprocss_data(threading.Thread):
    def __init__(self,datapath,patch_name,label_indices):
        threading.Thread.__init__(self)
        self.datapath = datapath
        self.patch_name = patch_name
        self.label_indices = label_indices
    def run(self):
        # preprocess tif files and save it to png file in datapath
        # Spectral band names to read related GeoTIFF files
        new_patch_path = self.datapath +'/'+ self.patch_name + '.png'
        if not Path(new_patch_path).exists():
            band_names = ['B01', 'B02', 'B03', 'B04', 'B05','B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
            tif_img = []
            for band_name in band_names:
                img_path = self.datapath +'/'+ self.patch_name + '/'+ self.patch_name+'_'+band_name+'.tif'
                band_ds = gdal.Open(img_path,  gdal.GA_ReadOnly)
                raster_band = band_ds.GetRasterBand(1)
                band_data = np.array(raster_band.ReadAsArray()) 
                # interpolate the image to (256,256)
                temp = resize(band_data,(256,256))
                # normalize and scale
                temp = scale(normalize(temp))
                tif_img.append(temp)

            # apply PCA
            Data = np.transpose(np.array(tif_img), axes=[1, 2, 0])
            [m, n, l] = np.shape(Data)
            x = np.reshape(Data, (m*n, l))
            # apply PCA reduce the channel to 3
            pca = PCA(n_components=3, copy=True, whiten=False)
            x = pca.fit_transform(x)
            _, l = x.shape
            x = np.reshape(x, (m, n, l)) # (256,256,3)
            # convert np array to pil image
            m,n = np.max(x),np.min(x)
            x = (x - n)/(m- n)*255 # value between [0,255]
            x = Image.fromarray(np.uint8(x))
            x.save(new_patch_path,"PNG")

        # get patch label
        patch_json_path = self.datapath +'/'+ self.patch_name  + '/' + self.patch_name +  '_labels_metadata.json'
        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)
        original_labels = patch_json['labels']
        return self.patch_name ,original_labels

def Give(opt, datapath):
    json_dir = os.path.dirname(__file__) + '/BigEarthNet_split'
    if not Path(json_dir + '/train.json').exists():
        print("start to preprocess BigEarthNet")
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
                future_list= [executor.submit(preprocss_data, datapath,patch_name,label_indices) for patch_name in patch_names]
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
        json_files = ['/train.json','/val.json','/test.json','/label_name.json']
        for i in range(len(json_files)):
            with open(json_dir + json_files[i], 'w') as json_file:
                json.dump(dict_list[i], json_file,separators=(",", ":"),allow_nan=False,indent=4)
                print("\ncreate ",json_dir + json_files[i])
    else:
        with open(json_dir +'/label_name.json') as json_file:
            conversion= json.load(json_file)
        with open(json_dir +'/train.json') as json_file:
            train_image_dict= json.load(json_file)
        with open(json_dir +'/test.json') as json_file:
            test_image_dict= json.load(json_file)
        with open(json_dir +'/val.json') as json_file:
            val_image_dict= json.load(json_file)
        for key in conversion.keys():
            train_image_dict[key] = [datapath + '/' + patch_name for patch_name in train_image_dict[key]]
            test_image_dict[key] = [datapath + '/' + patch_name for patch_name in test_image_dict[key]]
            val_image_dict[key] = [datapath + '/' + patch_name for patch_name in val_image_dict[key]]

    val_dataset = BaseDataset(val_image_dict, opt, is_validation=True)
    val_dataset.conversion   = conversion
  

    train_dataset = BaseDataset(train_image_dict, opt)
    train_dataset.conversion = conversion

    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    test_dataset.conversion  = conversion

    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_dataset.conversion  = conversion

    # for deep cluster feature
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    eval_train_dataset.conversion  = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
