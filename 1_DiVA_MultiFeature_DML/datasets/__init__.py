import datasets.BigEarthNet
import datasets.MLRSNet

def select(dataset, opt, data_path):
    if 'MLRSNet' in dataset:
        return MLRSNet.Give(opt, data_path)

    if 'BigEarthNet' in dataset:
        return BigEarthNet.Give(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196, stanford_online_products & in-shop!'.format(dataset))
