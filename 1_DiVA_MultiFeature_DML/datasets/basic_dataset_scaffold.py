from torch.utils.data import Dataset
import torch
import numpy as np
from pathlib import Path
import hypia # for hypespetral image augmentation
import random
import h5py

"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, hdf5, opt, is_validation=False):
        self.is_validation = is_validation
        self.pars        = opt

        #####
        self.image_dict = image_dict
        self.hdf = hdf5

        #####
        self.init_setup()


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

        self.image_paths = np.array(self.image_list)[:,0]

        self.is_init = True

     
    def normal_transform(self,img):
        # if self.pars.dataset =="MLRSNet":
        #     img = hypia.functionals.resize(img,256)
        # if  self.pars.dataset =="BigEarthNet":
        #     img = hypia.functionals.resize(img,120)
        # normalize
        if 'bninception' not in self.pars.arch:
            img = hypia.functionals.normalise(img,0.485,0.229)
        else:
            img = hypia.functionals.normalise(img,0.502,0.0039)
        return torch.Tensor(img)
    
    def real_transform(self,img,idx):
        # if self.pars.dataset =="MLRSNet":
        #     img = hypia.functionals.resize(img,256)
        # if  self.pars.dataset =="BigEarthNet":
        #     img = hypia.functionals.resize(img,120)
        # apply rotation
        imrot_class = idx%4
        angle = np.array([90,270,180,0])[imrot_class]
        im_b = hypia.functionals.rotate(img, angle,reshape=False)
        im_b = self.normal_transform(im_b)
        # normalize
        if 'bninception' not in self.pars.arch:
            im_b = hypia.functionals.normalise(im_b,0.485,0.229)
        else:
            im_b = hypia.functionals.normalise(im_b,0.502,0.0039)
        return torch.Tensor(im_b),imrot_class
    
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx][0]
        img_label = self.image_list[idx][-1]
        imrot_class = -1

        # get the image data from hdf5 file
        patch_name = img_path.split('/')[-1]
        f = h5py.File(self.hdf, 'r')
        data = f[patch_name][()]
        f.close()

        if Path(img_path).suffix =='.png' or Path(img_path).suffix =='.jpg':
            input_image = data.reshape(3, 256, 256)
        # hypespectral image (channels more than 3)
        else:
            input_image = data.reshape(12,120,120)
        
        if self.is_validation:
            im_a = self.normal_transform(input_image)
            return (img_label, im_a, idx)
        else:
            im_a, _ = self.real_transform(input_image,idx)
            im_b,imrot_class= self.real_transform(input_image,idx)
            return (img_label, im_a, idx, im_b, imrot_class)
        

    def __len__(self):
        return self.n_files
