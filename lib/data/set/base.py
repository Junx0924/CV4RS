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
    """Get channel - wise normalized image data

    Args:
        img (numpy array): shape (channel, width, height)

    Returns:
        numpy array: channel-wise normalized image data
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
    """Get image data from BigEarthNet dataset

    Args:
        img_path (str): image path

    Returns:
        numpy: flatten image data
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
        img_path (str)
    Return: 
        numpy array: flattened img data
    """
    pic = Image.open(img_path)
    if len(pic.size)==2:
        pic = pic.convert('RGB')
    pic = pic.resize((256,256))
    img_data = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
    return img_data.reshape(-1)
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, dataset_name, npmem_file="", conversion = None,transform = None, is_training=False,dset_type = 'train', include_aux_augmentations= False):
        """create dataset .

        Args:
            image_list (list): contains file_paths  and multi-hot labels 
            dataset_name (str): choose from {"MLRSNet", "BigEarthNet}
            npmem_file (str, optional): the path of npmem_file, if set use_npmem true, it will be automatically generated
            conversion (dict, optional): dictionary, {'label': label_name}
            transform (dict, optional): keys: sz_crop, input_shape. Defaults to None.
            is_training (bool, optional):if set, apply random flip and crop to the data split, else apply center crop. Defaults to False.
            dset_type (str, optional): select from {'train','val,'test'}. Defaults to 'train'.
            include_aux_augmentations (bool, optional): if set true, apply rotation to get augumented image data. Defaults to False.
        """        
        torch.utils.data.Dataset.__init__(self)
        self.dataset_name = dataset_name
        self.transform = transform
        self.dset_type = dset_type
        self.is_training = is_training
        self.npmem_file = npmem_file
        self.include_aux_augmentations = include_aux_augmentations
        self.conversion = conversion
        self.im_paths, self.I, self.ys = [], [], []
        for i,item in enumerate(image_list):
            self.I.append(i) # counter
            self.im_paths.append(item[0])
            self.ys.append(item[1]) # muti hot label
        
        category_labels = [np.where(label ==1)[0] for label in self.ys]
        unique_labels = np.unique(list(itertools.chain.from_iterable(category_labels)))
        self.image_dict ={str(key):[] for key in unique_labels}
        [[self.image_dict[str(cc)].append(i) for cc in c] for i,c in enumerate(category_labels)]

    def __len__(self):
        return len(self.I)

    def nb_classes(self):
        return len([key for key in self.image_dict.keys()])

    def __getitem__(self, index):
        label = torch.tensor(self.ys[index], dtype=int)
        img_path = self.im_paths[index]
        if os.path.exists(self.npmem_file):
            s = self.transform['input_shape']
            count = s[0]*s[1]*s[2]
            im =np.fromfile(self.npmem_file,count= count,dtype='float32',offset = index*4*count)
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
            # need to update self.I
            self.I = [ self.I[i]  for i in subset_indices]
            # ned to update self.image_dict
            ys =[ self.ys[i] for i in subset_indices]
            category_labels = [np.where(label ==1)[0] for label in ys]
            unique_labels = np.unique(list(itertools.chain.from_iterable(category_labels)))
            self.image_dict ={str(key):[] for key in unique_labels}
            [[self.image_dict[str(cc)].append(i) for cc in c] for i,c in zip(self.I,category_labels)]

    def process_image(self, img):
        """Preprocessing images .
        For training this function randomly crop images and
        flips the image randomly.
        For testing we use the center crop of the image.

        Args:
            img (numpy array) 

        Returns:
            torch.Tensor: shape [12, 100, 100] for BigEarthNet, [3, 224, 224] for MLRSNet
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