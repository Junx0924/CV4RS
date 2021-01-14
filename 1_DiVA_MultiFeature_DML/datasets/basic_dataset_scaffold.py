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
        img_dim = img.shape[1]
        if 'MLRSNet' in self.pars.dataset:
            crop = 224
        if 'BigEarthNet' in self.pars.dataset:
            crop = 100
        # for training randomly crop and randomly flip
        if not self.is_validation:
            # randomly crop
            if img_dim == crop: tl =[0,0]
            else: tl = np.random.choice(np.arange(img_dim-crop),2)
            img = hypia.functionals.crop(img, tl, crop, crop,channel_pos='first')
            # randomly flip
            choice = np.random.choice([1,2],1)
            if choice ==1:
                img = hypia.functionals.hflip(img) 
            else:
                img = hypia.functionals.vflip(img) 
        else:
            # for validation, use the center crop
            offset = (img_dim- crop)//2
            img = hypia.functionals.crop(img, [offset,offset], crop, crop,channel_pos='first')
            
        if 'resnet' in self.pars.arch:
            img = hypia.functionals.normalise(img,0.502,0.0039) 
        return torch.Tensor(img)
    
    def real_transform(self,img,idx):
        # apply rotation
        imrot_class = idx%4
        angle = np.array([0,90,180,270])[imrot_class]
        im_b = hypia.functionals.rotate(img, angle,reshape=False)
        # normalize
        if 'resnet' in self.pars.arch:
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

        if '.png' in img_path or '.jpg' in img_path:
            input_image = data.reshape(3, 256, 256)
        # hypespectral image (channels more than 3)
        else:
            input_image = data.reshape(12,120,120)
        
        im_a = self.normal_transform(input_image)
            
        if not self.is_validation:
            im_b,imrot_class= self.real_transform(input_image,idx)
            return (img_label, im_a, idx, im_b, imrot_class)
        else:
            return (img_label, im_a, idx)
        

    def __len__(self):
        return self.n_files
