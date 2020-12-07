
from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from osgeo import gdal
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize


# Spectral band names to read related GeoTIFF files
band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']


# for Geotiff images
def process_geotiff(img):
    tif_img = []
    for band_name in band_names:
        img_path = img + '_' + band_name + '.tif'
        band_ds = gdal.Open(img_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = np.array(raster_band.ReadAsArray())
        # interpolate the image to (256,256)
        temp = resize(band_data, (256, 256))
        # normalize and scale
        temp = scale(normalize(temp))
        tif_img.append(temp)
    Data = np.transpose(np.array(tif_img), axes=[1, 2, 0])
    [m, n, l] = np.shape(Data)
    # apply PCA reduce the channel to 3
    x = np.reshape(Data, (m * n, l))
    pca = PCA(n_components=3, copy=True, whiten=False)
    x = pca.fit_transform(x)
    _, l = x.shape
    x = np.reshape(x, (m, n, l))  # (256,256,3)
    # convert np array to pil image
    m, n = np.max(x), np.min(x)
    x = (x - n) / (m - n) * 255  # value between [0,255]
    x = PIL.Image.fromarray(np.uint8(x))
    return x


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes, transform=None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        img_path = self.im_paths[index]

        which = ""
        if img_path.lower().endswith(".jpg") | img_path.lower().endswith(".png"):
            # jpg or png
            im = PIL.Image.open(img_path)
        else:
            # geotiff
            im = process_geotiff(img_path)

        # convert gray to rgb
        if len(list(im.split())) == 1:
            im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
