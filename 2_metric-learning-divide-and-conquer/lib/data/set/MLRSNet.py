import numpy as np
from pathlib import Path
import csv
import random
import xlrd
import itertools
import json
import os
import h5py
from PIL import Image
import multiprocessing
 

def split(image_dict,split_ratio):
    train_image_dict  = {} 
    other_image_dict  = {} 
    keys = sorted(list(image_dict.keys()))
    values = np.unique(list(itertools.chain.from_iterable(image_dict.values())))
    flag =  {ind:"undefine" for ind in values}

    for key in keys:
        samples_ind = image_dict[key]
        random.shuffle(samples_ind)
        sample_num = len(samples_ind)
        train_image_dict[key] =[]
        other_image_dict[key] =[]
        # check if there are some sample id already in train/nontrain
        for ind in samples_ind:
            if flag[ind] =="undefine":
                if len(train_image_dict[key])< int(sample_num*split_ratio):
                    train_image_dict[key].append(ind)
                    flag[ind] ="train"
                else:
                    if len(other_image_dict[key])< (sample_num - int(sample_num*split_ratio)):
                        other_image_dict[key].append(ind)
                        flag[ind] ="nontrain"
            elif flag[ind] =="train":
                if len(train_image_dict[key])< int(sample_num*split_ratio):
                    train_image_dict[key].append(ind)
            else:
                if len(other_image_dict[key])< (sample_num - int(sample_num*split_ratio)):
                    other_image_dict[key].append(ind)
            
    return train_image_dict,flag

# csv_filename: record the image name and its labels
# datapath: the source of dataset
def read_csv(csv_filename,datapath):
    file_list, file_label =[],[]
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader: 
            file_list.append(datapath + str(row[0]))
            file_label.append(row[1:])
    file_list = np.array(file_list)
    file_label = np.array(file_label,dtype=int)
    file_image_dict  = {i:file_list[np.where(file_label[:,i]==1)[0]] for i in range(file_label.shape[1])}
    # # randomly sample up 30% for quick running
    # for key in file_image_dict.keys():
    #     temp = list(file_image_dict[key])
    #     k = int(len(temp)*0.3)
    #     file_image_dict[key]= np.array(random.sample(temp,k))
    return file_image_dict,file_list


def get_data(img_path):
    patch_name = img_path.split('/')[-1]
    pic = Image.open(img_path)
    if len(pic.size)==2:
        pic = pic.convert('RGB')
    pic = pic.resize((256,256))
    img_data = np.array(pic.getdata()).reshape(-1, pic.size[0], pic.size[1])
    return patch_name,img_data.reshape(-1)
 
# hdf_file: hdf5 file record the images
# file_list: record the image paths
def store_hdf(hdf_file, file_list):
    count = 0
    while (count < len(file_list)):
        if count==0: data_list = file_list
        else: 
            f_read = h5py.File(hdf_file,'r')
            data_list = [x for x in file_list if x not in list(f_read.keys())]
            f_read.close()
        
        f = h5py.File(hdf_file,'w')
        pool = multiprocessing.Pool(8)
        result = pool.imap(get_data, (img_path for img_path in data_list))
        for idx,(patch_name, img_data) in enumerate(result):
            f.create_dataset(patch_name, data=img_data, dtype='i',compression='gzip',compression_opts=9)
            if (idx+1) % 2000==0: print("processed {0:.0f}%".format((idx+1)/len(data_list)*100))
        pool.close()
        pool.join()
        f.close()
        f_read = h5py.File(hdf_file,'r')
        count = len(list(f_read.keys()))
        f_read.close()
            
def Give(datapath,dset_type):
    csv_dir = os.path.dirname(__file__) + '/MLRSNet_split'
    # check the split train/test/val existed or not
    if not Path(csv_dir +'/train.csv').exists():
        category = {}
        category_path = datapath + '/Categories_names.xlsx'
        book = xlrd.open_workbook(category_path)
        sheet = book.sheet_by_index(1)
        for i in range(2,sheet.nrows):
            category_name = sheet.cell_value(rowx=i, colx=1)
            temp_label_name = np.unique(np.array([sheet.cell_value(rowx=i, colx=j).strip() for j in range(2,sheet.ncols) if sheet.cell_value(rowx=i, colx=j)!=""]))
            if "chapparral" in temp_label_name: temp_label_name[np.where(temp_label_name=="chapparral")]= "chaparral"
            category[category_name] = temp_label_name

        label_folder = datapath +'/labels/'
        image_folder  = datapath +'/Images/'
        image_list =[] # image path
        image_labels =[]
            
        for entry in Path(label_folder).iterdir():
            if entry.suffix ==".csv" :
                with open(label_folder + entry.name) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    label_names =next(csv_reader,None)[1:]
                    if len(label_names)==60:
                        sort_ind = np.argsort(label_names) 
                        for row in csv_reader: 
                            image_path = image_folder + entry.stem +'/'+row[0]
                            #image_list.append(image_path)
                            image_list.append('/Images/'+ entry.stem +'/'+row[0])
                            temp = np.array(row[1:])
                            image_labels.append(temp[sort_ind])
                    else:
                        print(entry.name)

        label_names = np.sort(label_names)
        # to record the label names and its id
        label_names_dict= {i:x for i,x in enumerate(label_names)} 

        for key in category.keys():
            labels = np.array(category[key])
            label_ind = [str(np.where(label_names==item)[0][0]) for item in labels]
            category[key] = label_ind

        image_list = np.array(image_list)
        image_labels = np.array(image_labels,dtype=int)
        image_dict  = {i:np.where(image_labels[:,i]==1)[0] for i in range(len(label_names))}

        # split data into train/test 50%/50% balanced in class.
        temp_image_dict,flag_test =split(image_dict, 0.5)
        # split train into train/val 40%/10% balanced in class
        temp_image_dict,flag_val =split(temp_image_dict, 0.8)

        train   = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_val.keys())) if flag_val[ind]=="train"]
        val  = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_val.keys())) if flag_val[ind]=="nontrain"]
        test   = [[image_list[ind]]+list(image_labels[ind,:]) for ind in sorted(list(flag_test.keys())) if flag_test[ind]=="nontrain"]
        
        with open(csv_dir +'/label_name.json', 'w+') as label_file:
            json.dump(label_names_dict, label_file,separators=(",", ":"),allow_nan=False,indent=4)
        with open(csv_dir +'/category.json', 'w+') as category_file:
            json.dump(category, category_file,separators=(",", ":"),allow_nan=False,indent=4)
        with open(csv_dir +'/train.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train)
        with open(csv_dir +'/test.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(test)
        with open(csv_dir +'/val.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(val)
    
    with open(csv_dir +'/category.json') as json_file:
        category = json.load(json_file)
    with open(csv_dir +'/label_name.json') as json_file:
        conversion= json.load(json_file)
    train_image_dict,train_list = read_csv(csv_dir +'/train.csv',datapath)
    val_image_dict,val_list = read_csv(csv_dir +'/val.csv',datapath)
    test_image_dict ,test_list= read_csv(csv_dir +'/test.csv',datapath)
    
    # store all the images in hdf5 files to further reduce disk I/O
    train_h5 = datapath +'/train.hdf5'
    if not Path(train_h5).exists(): 
        print("Start to create ", train_h5," for MLRSNet")
        store_hdf(train_h5, train_list)
    val_h5 = datapath +'/val.hdf5'
    if not Path(val_h5).exists(): 
        print("Start to create ", val_h5," for MLRSNet")
        store_hdf(val_h5, val_list)
    test_h5 = datapath +'/test.hdf5'
    if not Path(test_h5).exists(): 
        print("Start to create ", test_h5," for MLRSNet")
        store_hdf(test_h5, test_list)

    dsets = {'train': train_image_dict , 'val': val_image_dict , 'test': test_image_dict}
    return dsets[dset_type]