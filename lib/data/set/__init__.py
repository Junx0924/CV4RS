from lib.data.set.base import BaseDataset
from lib.data.set import BigEarthNet
from lib.data.set import MLRSNet

def one_hot(y,num_classes):
    label = [0]*num_classes
    for i in y:
        label[i] = 1
    return label

def select(datapath,dset_type,transform,is_training = False, is_onehot = False,include_aux_augmentations=False, use_hdf5 = True):
    types = {"train":"train","query":"val","gallery":"test"}
    dset_type = types[dset_type]
    if use_hdf5:
        hdf_file =  datapath + '/'+ dset_type +'.hdf5'
    else:
        hdf_file = ""

    if 'MLRSNet' in datapath:
        dict_temp = MLRSNet.Give(datapath,dset_type,use_hdf5)
    if 'BigEarthNet' in datapath:
        dict_temp = BigEarthNet.Give(datapath,dset_type,use_hdf5)
    
    # add file index to image_dict
    image_dict = {}
    counter = 0
    if not is_onehot:
        for key in dict_temp.keys():
            image_dict[key]= []
            for path in dict_temp[key]:
                image_dict[key].append([path, counter])
                counter += 1
        image_list = [[(x[0],x[1],int(key)) for x in image_dict[key]] for key in image_dict.keys()]
        image_list = [x for y in  image_list for x in y]
    else:
        temp_dict ={}
        num_classes = len([key for key in dict_temp.keys()])
        for key in dict_temp.keys():
            for img_path in dict_temp[key]:
                if img_path not in temp_dict.keys():
                    temp_dict[img_path]=[counter,[]]
                    counter = counter + 1
                temp_dict[img_path][1].append(int(key))
        image_list = [(img_path,temp_dict[img_path][0],one_hot(temp_dict[img_path][1],num_classes)) for img_path in temp_dict.keys()]
    
        for key in dict_temp.keys():
            image_dict[key]= []
            for img_path in dict_temp[key]:
                counter = temp_dict[img_path][0]
                image_dict[key].append([img_path, counter])

    return BaseDataset(image_dict,image_list,hdf_file,transform,is_training,include_aux_augmentations)