from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import hypia # for hypespetral image augmentation
import random
from osgeo import gdal
"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, opt, is_validation=False):
        self.is_validation = is_validation
        self.pars        = opt

        #####
        self.image_dict = image_dict

        #####
        self.init_setup()

        #####
        self.include_aux_augmentations = False #required by get selfsimilarity 

    def init_setup(self):
        self.n_files       = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))


        counter = 0
        temp_image_dict = {}
        for i,key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(x[0],int(key)) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list

        self.is_init = True

     
    def normal_transform(self,img):
        if self.pars.dataset =="MLRSNet":
            img = hypia.functionals.resize(img,256)
        if  self.pars.dataset =="BigEarthNet":
            img = hypia.functionals.resize(img,120)
        # normalize
        if 'bninception' not in self.pars.arch:
            img = hypia.functionals.normalise(img,0.485,0.229)
        else:
            img = hypia.functionals.normalise(img,0.502,0.0039)
        return torch.Tensor(img)
    
    def real_transform(self,img,idx):
        if self.pars.dataset =="MLRSNet":
            img = hypia.functionals.resize(img,256)
        if  self.pars.dataset =="BigEarthNet":
            img = hypia.functionals.resize(img,120)
        # apply rotation
        imrot_class = idx%4
        angle = np.array([0,90,180,270])[imrot_class]
        im_b = hypia.functionals.rotate(img, angle,reshape=True)
        im_b = self.normal_transform(im_b)
        # normalize
        if 'bninception' not in self.pars.arch:
            im_b = hypia.functionals.normalise(im_b,0.485,0.229)
        else:
            im_b = hypia.functionals.normalise(im_b,0.502,0.0039)
        return torch.Tensor(im_b),imrot_class
    
    # for images like png, jpg etc
    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img
    
    def read_tiff(self,img_path):
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
            with open(img_path, 'wb') as f:
                np.save(f,np.array(tif_img))
        return np.load(img_path)

    def __getitem__(self, idx):
        img_path = self.image_list[idx][0]
        img_label = self.image_list[idx][-1]
        imrot_class = -1

        if Path(img_path).suffix =='.png' or Path(img_path).suffix =='.jpg':
            pic = self.ensure_3dim(Image.open(img_path))
            input_image = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
        # hypespectral image (channels more than 3)
        else:
            input_image = self.read_tiff(img_path)
        
        if self.include_aux_augmentations:
            im_a = self.normal_transform(input_image)
            im_b,imrot_class= self.real_transform(input_image,idx)
            return (img_label, im_a, idx, im_b, imrot_class)
        else:
            im_a = self.normal_transform(input_image)
            return (img_label, im_a, idx)
        

    def __len__(self):
        return self.n_files
