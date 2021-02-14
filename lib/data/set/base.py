from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import h5py
import hypia
from skimage.transform import resize
from osgeo import gdal
from PIL import Image
import os
import itertools


def normalize(img):
    """
    Get channel-wise normalized image data
    Args:
        img: np array (channel, width, height)
    Return:
        img: np array, channel-wise normalized
    """
    img_channel = img.shape[0]
    if img_channel==3: 
        img = img/255
    # calculate per-channel means and standard deviations
    img_mean = np.array([np.mean(np.reshape(img[i,:,:],-1)) for i in range(img_channel)]).reshape(-1,1,1)
    img_std = np.array([np.std(np.reshape(img[i,:,:],-1)) for i in range(img_channel)]).reshape(-1,1,1)
    img = (img - img_mean)/(img_std + 0.00000001)
    return img

def get_BigEarthNet(img_path):
    """
    Get image data from BigEarthNet dataset
    Args: 
        img_path
    Return: 
        img_data: flatten np array
    """
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
    return tif_img.reshape(-1)

def get_MLRSNet(img_path):
    """
    Get image data from MLRSNet dataset
    Args: 
        img_path
    Return: 
        img_data: flatten np array
    """
    pic = Image.open(img_path)
    if len(pic.size)==2:
        pic = pic.convert('RGB')
    pic = pic.resize((256,256))
    img_data = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
    return img_data.reshape(-1)
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, dataset_name, hdf_file="", conversion = None,transform = None, is_training = False, include_aux_augmentations= False):
        """
        The train data is randomly flip and cropped, the eval data is center cropped
        Args:
            image_list: contains file_paths and multi-hot labels
            dataset_name: choose from {"MLRSNet", "BigEarthNet}
            hdf_file: the path of hdf_file, if set use_hdf true, it will be automatically generated
            conversion: dictionary, {'label': label_name}
            transform: dictonary, keys: sz_crop, input_shape
            is_training: if set true, apply random flip and crop for training, else apply center crop
            include_aux_augmentations: if set true, apply rotation to get augumented image data
        """
        torch.utils.data.Dataset.__init__(self)
        self.dataset_name = dataset_name
        self.transform = transform
        self.is_training = is_training
        self.hdf_file = hdf_file
        self.include_aux_augmentations = include_aux_augmentations
        self.conversion = conversion
        self.im_paths, self.I, self.ys = [], [], []
        for i,item in enumerate(image_list):
            self.im_paths.append(item[0])
            self.I.append(i) # counter
            self.ys.append(item[1]) # muti hot label
        
        # in case of incomplete data
        # get rid of class which has only 1 sample
        # num_samples = np.sum(self.ys,axis=0)
        # class_ind = np.where(num_samples==1)[0]
        # if len(class_ind)>0:  
        #     self.ys[:,class_ind]= 0
        #     del_inds= np.unique(np.where(self.ys==0)[0])
        #     keep_inds = list(set(self.I) - set(del_inds))
        #     self.ys = self.ys[keep_inds]
        #     self.im_paths =self.im_paths[keep_inds]
        #     self.I = np.arange(len(self.ys))

        category_labels = [np.where(label ==1)[0] for label in self.ys]
        unique_labels = np.unique(list(itertools.chain.from_iterable(category_labels)))
        self.image_dict ={str(key):[] for key in unique_labels}
        [[self.image_dict[str(cc)].append(i) for cc in c] for i,c in enumerate(category_labels)]

    def __len__(self):
        return len(self.im_paths)

    def nb_classes(self):
        return len([key for key in self.image_dict.keys()])

    def __getitem__(self, index):
        label = torch.tensor(self.ys[index], dtype=int)
        img_path = self.im_paths[index]
        if os.path.exists(self.hdf_file):
            patch_name = img_path.split('/')[-1]
            f = h5py.File(self.hdf_file, 'r')
            im = np.array(f[patch_name][()], dtype=float)
            f.close()
        else:
            if self.dataset_name =='BigEarthNet':
                im = get_BigEarthNet(img_path)
            else:
                im = get_MLRSNet(img_path)

        if self.transform ==None:
            return (im, label,index)
        else:
            im_a = self.process_image(im)
            if self.include_aux_augmentations:
                # apply rotation to augument images
                angle =  np.random.choice([0,90,180,270],size=1)
                im_b = hypia.functionals.rotate(im.reshape(self.transform['input_shape']), angle[0],reshape=False)
                im_b = self.process_image(im_b)
                return im_a, label, index, im_b
            else:
                return im_a, label, index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, subset_indices):
        if subset_indices is not None:
            self.ys =[self.ys[i] for i in subset_indices]
            self.I = [i for i in range(len(subset_indices))]
            self.im_paths = [self.im_paths[i] for i in subset_indices]
            # update image_dict
            category_labels = [np.where(label ==1)[0] for label in self.ys]
            unique_labels = np.unique(list(itertools.chain.from_iterable(category_labels)))
            self.image_dict ={str(key):[] for key in unique_labels}
            [[self.image_dict[str(cc)].append(i) for cc in c] for i,c in enumerate(category_labels)]

    def process_image(self, img):
        """
        Preprocessing images. For training this function randomly crop images and
        flips the image randomly.
        For testing we use the center crop of the image.

        Args:
        img: np.array
        """
        img_shape =self.transform['input_shape']
        img = img.reshape(img_shape)
        img_dim = img_shape[1]
        img_channel = img_shape[0]
        crop = self.transform['sz_crop']
        if self.is_training:
            # random_image_crop
            if img_dim == crop: tl =[0,0]
            else: tl = np.random.choice(range(img_dim-crop),2)
            img = hypia.functionals.crop(img, tl, crop, crop,channel_pos='first')
            # apply random flip
            choice = np.random.choice([1,2],1)
            if choice ==1 :
                img = hypia.functionals.hflip(img,channel_pos='first')
            else:
                img = hypia.functionals.vflip(img,channel_pos='first')
        else:
            offset = (img_dim - crop) // 2
            img = hypia.functionals.crop(img, [offset,offset], crop, crop,channel_pos='first')

        img = normalize(img)
        return  torch.Tensor(img) 