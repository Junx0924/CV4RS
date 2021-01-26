from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import h5py
import hypia


    
class BaseDataset(torch.utils.data.Dataset):
    """
    We use the train set for training, the val set for
    query and the test set for retrieval
    """
    def __init__(self, image_dict, image_list,hdf_file, transform = None, is_training = False):
        torch.utils.data.Dataset.__init__(self)
        self.transform = transform
        self.is_training = is_training
        self.image_dict = image_dict
        self.image_list = image_list
        self.hdf_file = hdf_file

        self.im_paths, self.I, self.ys = [], [], []
        for item in self.image_list:
            self.im_paths.append(item[0])
            self.I.append(item[1]) # counter
            self.ys.append(item[2]) # label

    def __len__(self):
        return len(self.im_paths)

    def nb_classes(self):
        return len([key for key in self.image_dict.keys()])

    def __getitem__(self, index):
        img_path = self.im_paths[index]
        patch_name = img_path.split('/')[-1]
        f = h5py.File(self.hdf_file, 'r')
        im = f[patch_name][()]
        f.close()
        im = self.process_image(np.array(im, dtype=float),mirror=True)
        label = self.ys[index]
        if isinstance(label,list):
            label = torch.tensor(label, dtype=int)
        return im, label, index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, subset_indices):
        if subset_indices is not None:
            temp_list = [self.image_list[int(i)] for i in subset_indices]
            self.image_list = []
            self.ys =[]
            self.I = []
            self.im_paths = []
            self.image_dict ={}

            for ind in range(len(temp_list)):
                key = temp_list[ind][-1]
                img_path = temp_list[ind][0]
                self.image_list.append([img_path,ind,key])
                self.ys.append(key)
                self.I.append(ind)
                self.im_paths.append(img_path)
                if key not in self.image_dict.keys():
                    self.image_dict[key]=[]
                self.image_dict[key].append([img_path,ind])

    def process_image(self, img, mirror=True):
        """
        Preprocessing code. For training this function randomly crop images and
        flips the image randomly.
        For testing we use the center crop of the image.

        Args:
        img: np.array (1 dim)
        """
        img_shape = self.transform['input_shape']
        img = img.reshape(img_shape)
        img_dim = img_shape[1]
        crop = self.transform['sz_crop']
        mean = self.transform['mean']
        std = self.transform['std']
        if  self.is_training:
            # random_image_crop
            if img_dim == crop: tl =[0,0]
            else: tl = np.random.choice(range(img_dim-crop),2)
            img = hypia.functionals.crop(img, tl, crop, crop,channel_pos='first')
            if mirror:
                choice = np.random.choice([1,2],1)
                if choice ==1 :
                    img = hypia.functionals.hflip(img,channel_pos='first')
                else:
                    img = hypia.functionals.vflip(img,channel_pos='first')

        else:
            offset = (img_dim - crop) // 2
            img = hypia.functionals.crop(img, [offset,offset], crop, crop,channel_pos='first')

        # normalize
        img = hypia.functionals.normalise(img,mean,std) 
        return  torch.Tensor(img)