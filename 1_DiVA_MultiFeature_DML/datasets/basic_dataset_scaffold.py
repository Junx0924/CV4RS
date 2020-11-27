from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import gdal
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize

# Spectral band names to read related GeoTIFF files
band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

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
        if 'bninception' not in opt.arch:
            # self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        else:
            # normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[1., 1., 1.])
            self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])
            # self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])

        transf_list = []

        self.crop_size = crop_im_size = 224 if 'resnet' in opt.arch else 227


        #############
        self.normal_transform = []
        if not self.is_validation:
            self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)



        #############
        self.real_transform = []
        self.real_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomGrayscale(p=0.2),
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.RandomHorizontalFlip(0.5)])
        self.real_transform.extend([transforms.ToTensor(), normalize])
        self.real_transform = transforms.Compose(self.real_transform)

        #####
        self.include_aux_augmentations = True #required by get selfsimilarity 
        self.predict_rotations         = None


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
        self.image_list = [[(x[0],key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list

        self.is_init = True

   # for images like png, jpg etc
    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img
    
    # for Geotiff images
    def process_Geotiff(self, img):
        tif_img = []
        for band_name in band_names:
            img_path = img +'_' + band_name + '.tif'
            band_ds = gdal.Open(img_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = np.array(raster_band.ReadAsArray()) 
            # interpolate the image to (256,256)
            temp = resize(band_data,(256,256))
            # normalize and scale
            temp = scale(normalize(temp))
            tif_img.append(temp)
        Data = np.transpose(np.array(tif_img), axes=[1, 2, 0])
        [m, n, l] = np.shape(Data)
        # apply PCA reduce the channel to 3
        x = np.reshape(Data, (m*n, l))
        pca = PCA(n_components=3, copy=True, whiten=False)
        x = pca.fit_transform(x)
        _, l = x.shape
        x = np.reshape(x, (m, n, l)) # (256,256,3)
        # convert np array to pil image
        m,n = np.max(x),np.min(x)
        x = (x - n)/(m- n)*255 # value between [0,255]
        x = Image.fromarray(np.uint8(x))
        return x

    def __getitem__(self, idx):
        imrot_class = -1
        # for images like png,jpg etc
        if (("jpg" in self.image_list[idx][0]) or ("png" in self.image_list[idx][0]) ):
            input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))
        # for hyper-spectral images
        else:
            input_image = self.process_Geotiff(self.image_list[idx][0])
            

        if self.include_aux_augmentations:
            im_a = self.real_transform(input_image) if self.pars.realistic_main_augmentation else self.normal_transform(input_image)

            if not self.predict_rotations:
                im_b = self.real_transform(input_image) if self.pars.realistic_augmentation else self.normal_transform(input_image)
            else:
                class ImRotTrafo:
                    def __init__(self, angle):
                        self.angle = angle
                    def __call__(self, x):
                        return transforms.functional.rotate(x, self.angle)

                imrot_class = idx%4
                angle      = np.array([0,90,180,270])[imrot_class]
                imrot_aug  = [ImRotTrafo(angle), transforms.Resize((256,256)), transforms.RandomCrop(size=self.crop_size),
                            transforms.ToTensor(), self.f_norm]
                imrot_aug  = transforms.Compose(imrot_aug)
                im_b        = imrot_aug(input_image)

            if 'bninception' in self.pars.arch:
                im_a, im_b = im_a[range(3)[::-1],:], im_b[range(3)[::-1],:]

            return (self.image_list[idx][-1], im_a, idx, im_b, imrot_class)
        else:
            im_a = self.real_transform(input_image) if self.pars.realistic_main_augmentation else self.normal_transform(input_image)
            if 'bninception' in self.pars.arch:
                im_a = im_a[range(3)[::-1],:]
            return (self.image_list[idx][-1], im_a, idx)
        

    def __len__(self):
        return self.n_files
