import datasets.BigEarthNet
import datasets.MLRSNet

def select(dataset_name,source_path):
    data_path = source_path + '/' + dataset_name
    if 'MLRSNet' in dataset_name:
        return MLRSNet.Give(data_path)

    if 'BigEarthNet' in dataset_name:
        return BigEarthNet.Give(data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196, stanford_online_products & in-shop!'.format(dataset))
