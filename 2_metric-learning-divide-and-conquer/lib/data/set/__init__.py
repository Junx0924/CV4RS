from lib.data.set.base import BaseDataset
from lib.data.set import BigEarthNet
from lib.data.set import MLRSNet


def select(root,dset_type,transform,is_training = False):
    types = {"train":"train","query":"val","gallery":"test"}
    dset_type = types[dset_type]
    hdf_file =  root + '/'+ dset_type +'.hdf5'

    if 'MLRSNet' in root:
        dict_temp = MLRSNet.Give(root,dset_type)
    if 'BigEarthNet' in root:
        dict_temp = BigEarthNet.Give(root,dset_type)
    
    # add file index to image_dict
    image_dict = {}
    counter = 0
    for key in dict_temp.keys():
        image_dict[key]= []
        for path in dict_temp[key]:
            image_dict[key].append([path, counter])
            counter += 1
    image_list = [[(x[0],x[1],int(key)) for x in image_dict[key]] for key in image_dict.keys()]
    image_list = [x for y in  image_list for x in y]
    
    return BaseDataset(image_dict,image_list,hdf_file,transform,is_training)