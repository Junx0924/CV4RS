
import torchvision
import torch
from math import ceil
import logging
import torch
import numpy as np
import torchvision
import torch
from torch.nn import Linear, Dropout, AvgPool2d, MaxPool2d
from torch.nn.init import xavier_normal_

"""============================================================="""
def increase_channels(m, num_channels=None, copy_weights=0):
    """
    takes as input a Conv2d layer and returns the Conv2d layer with `num_channels` input channels
    and all the previous weights copied into the new layer.
    """
    # number of input channels the new module should have
    new_in_channels = num_channels if num_channels is not None else m.in_channels + 1
    
    bias = False if m.bias is None else True
    
    # Creating new Conv2d layer
    new_m = torch.nn.Conv2d(in_channels=new_in_channels, 
                        out_channels=m.out_channels, 
                        kernel_size=m.kernel_size, 
                        stride=m.stride, 
                        padding=m.padding,
                        bias=bias)
    
    # Copying the weights from the old to the new layer
    new_m.weight[:, :m.in_channels, :, :] = m.weight.clone()
    
    #Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
    for i in range(new_in_channels - m.in_channels):
        channel = m.in_channels + i
        new_m.weight[:, channel:channel+1, :, :] = m.weight[:, copy_weights:copy_weights+1, : :].clone()
    new_m.weight = torch.nn.Parameter(new_m.weight)

    return new_m

def resnet50(config, pretrained = True):
    # start edits
    # this would load the model from the internet
    # model = torchvision.models.resnet50(pretrained = pretrained)
    # istead, load it from disk
    model = torchvision.models.resnet50(pretrained=False)

    state_dict = torch.load(config['pretrained_weights_file'])
    model.load_state_dict(state_dict)
    # end edits
    if config["dataset_selected"] =="BigEarthNet":
        input_channels = 12
        model.conv1 = increase_channels(model.conv1, input_channels)

    model.features = torch.nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4
    )

    model.sz_features_output = 2048

    if config['frozen']:
        # child_counter = 0
        # for child in model.children():
        #     if child_counter == 0 and config['dataset_selected'] =="BigEarthNet":
        #         continue
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = False
        for module in filter(lambda m: type(m) == torch.nn.BatchNorm2d, model.modules()):
            module.eval()
            module.train = lambda _: None
    return model


def make_parameters_dict(model, filter_module_names):
    """
    Separates model parameters into 'backbone' and other modules whose names
    are given in as list in `filter_module_names`, e.g. ['embedding_layer'].
    """

    # init parameters dict
    D = {k: [] for k in ['backbone', *filter_module_names]}
    for name, param in model.named_parameters():
        name = name.split('.')[0]
        if name not in filter_module_names:
            D['backbone'] += [param]
        else:
            D[name] += [param]

    # verify that D contains same number of parameters as in model
    nb_total = len(list(model.parameters()))
    nb_dict_params = sum([len(D[d]) for d in D])
    assert nb_total == nb_dict_params
    return D


def init_splitted(layer, sub_embed_sizes):
    # initialize splitted embedding parts separately
    from math import ceil
    for c in range(len(sub_embed_sizes)):
        start = int(sum(sub_embed_sizes[:c]))
        stop = int(start + sub_embed_sizes[c])
        i = torch.arange(start,stop).long()

        _layer = torch.nn.Linear(layer.weight.shape[1], len(i))
        layer.weight.data[i] = xavier_normal_(_layer.weight.data, gain = 1)
        layer.bias.data[i] = _layer.bias.data


def embed_model(model, config, sz_embedding, normalize_output=True):
    if config['dataset_selected'] =="BigEarthNet": k_s = 4
    else: k_s = 7
    model.features_pooling = AvgPool2d(k_s,
        stride=1, padding=0, ceil_mode=True, count_include_pad=True
    )
    model.features_dropout = Dropout(0.01)

    # choose arbitrary parameter for selecting GPU/CPU
    dev = list(model.parameters())[0].device
    if type(model) != torchvision.models.ResNet:
        model.sz_features_output = _sz_features[type(model)]
    torch.random.manual_seed(config['random_seed'] + 1)
    model.embedding = Linear(model.sz_features_output, sz_embedding).to(dev)

    # for fair comparison between different cluster sizes
    torch.random.manual_seed(config['random_seed'] + 1)
    np.random.seed(config['random_seed'] + 1)

    sub_embed_sizes = config['sub_embed_sizes']
    init_splitted(model.embedding, sub_embed_sizes)

    model.parameters_dict = make_parameters_dict(
        model = model,
        filter_module_names = ['embedding']
    )

    assert normalize_output

    learner_neurons = [None] * len(sub_embed_sizes)

    for c in range(len(sub_embed_sizes)):
        start = int(sum(sub_embed_sizes[:c]))
        stop = int(start + sub_embed_sizes[c])
        learner_neurons[c] = np.arange(start,stop)
    model.learner_neurons = learner_neurons

    def forward(x, use_penultimate=False):
        x = model.features(x)
        x = model.features_pooling(x)
        x = model.features_dropout(x)
        x = x.view(x.size(0), -1)
        if not use_penultimate:
            x = model.embedding(x)
            for idxs in model.learner_neurons:
                x[:, idxs] = torch.nn.functional.normalize(
                    x[:, idxs], p=2, dim=1
                )
        else:
            # normalize the entire penultimate layer
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x
    model.forward = forward


def make(config):
    model = resnet50(config=config, pretrained = True)
    embed_model(
        model = model,
        config = config,
        sz_embedding = config['sz_embedding'],
        normalize_output = True
    )
    return model
