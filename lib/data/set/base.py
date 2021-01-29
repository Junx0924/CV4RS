from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import h5py
import hypia
from skimage.transform import resize
from osgeo import gdal
from PIL import Image

def get_BigEarthNet(img_path):
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
    pic = Image.open(img_path)
    if len(pic.size)==2:
        pic = pic.convert('RGB')
    pic = pic.resize((256,256))
    img_data = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
    return img_data.reshape(-1)
    
class BaseDataset(torch.utils.data.Dataset):
    """
    We use the train set for training, the val set for
    query and the test set for retrieval
    """
    def __init__(self, image_dict, image_list,hdf_file, transform = None, is_training = False, include_aux_augmentations= False):
        torch.utils.data.Dataset.__init__(self)
        self.transform = transform
        self.is_training = is_training
        self.image_dict = image_dict
        self.image_list = image_list
        self.hdf_file = hdf_file
        self.include_aux_augmentations = include_aux_augmentations

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
        if self.hdf_file !="":
            patch_name = img_path.split('/')[-1]
            f = h5py.File(self.hdf_file, 'r')
            im = np.array(f[patch_name][()], dtype=float)
            f.close()
        elif self.transform['input_shape'][0] ==12:
            im = get_BigEarthNet(img_path)
        else:
            im = get_MLRSNet(img_path)

        input = im.reshape(self.transform['input_shape'])
        im_a = self.process_image(input,mirror=True)
        label = self.ys[index]
        if isinstance(label,list):
            label = torch.tensor(label, dtype=int)
        
        if self.include_aux_augmentations and self.is_training:
            def rotation(img,idx):
                # apply rotation
                imrot_class = idx%4
                angle = np.array([0,90,180,270])[imrot_class]
                im_b = hypia.functionals.rotate(img, angle,reshape=False)
                return im_b,imrot_class
            im_b,imrot_class= rotation(input, index)
            im_b = self.process_image(im_b,mirror=False)
            return (im_a, label, index, im_b, imrot_class)
        else:
            return (im_a, label, index)

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
        img_shape =self.transform['input_shape']
        img_dim = img_shape[1]
        img_channel = img_shape[0]
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

        # normalize the image to its own mean and std
        img_mean = np.array([np.mean(np.reshape(img[i,:,:],-1)) for i in range(img_channel)]).reshape(-1,1,1)
        img_std = np.array([np.std(np.reshape(img[i,:,:],-1)) for i in range(img_channel)]).reshape(-1,1,1)
        img = (img -img_mean)/img_std
        # normalize to the assigned mean and std
        if isinstance(mean,list): mean = np.array(mean).reshape(-1,1,1)
        if isinstance(std,list): std = np.array(std).reshape(-1,1,1)
        img = (img - mean)/std
        return  torch.Tensor(img)