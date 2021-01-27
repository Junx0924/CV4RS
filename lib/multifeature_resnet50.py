"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import  torchvision

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
    new_m = nn.Conv2d(in_channels=new_in_channels, 
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
    new_m.weight = nn.Parameter(new_m.weight)

    return new_m

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

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        # set pretrained = False to stop downloading weight from internet
        self.model = torchvision.models.resnet50(pretrained = False)
        state_dict = torch.load(config['pretrained_weights_file'])
        self.model.load_state_dict(state_dict)  

        self.feature_dim = 2048
        # This increases the number of input channels for our network
        if config["dataset_selected"] =="BigEarthNet":
            input_channels = 12
            self.model.conv1 = increase_channels(self.model.conv1, input_channels)

        if config['frozen']:
            child_counter = 0
            for child in self.model.children():
                if child_counter == 0 and config['dataset_selected'] =="BigEarthNet":
                    continue
                else:
                    for param in child.parameters():
                        param.requires_grad = False

        self.model.features = torch.nn.Sequential(
            self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
            self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4)

        # add pooling layer
        k_s = 4 if config['dataset_selected'] =="BigEarthNet" else 7
        self.model.features_pooling = torch.nn.AvgPool2d(k_s,stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        # add dropout layer
        self.model.features_dropout = torch.nn.Dropout(0.01)

        # add new embedding layer
        out_dict = nn.ModuleDict()
        sub_embed_sizes = config['sub_embed_sizes']

        if 'diva_features' in config.keys():
            embed_names = config['diva_features']
        else:
            embed_names = ['embed_' + str(i+1) for i in range(len(sub_embed_sizes))]

        for name,size in zip(embed_names,sub_embed_sizes):
            out_dict[name] = torch.nn.Linear(self.feature_dim, size)
        
        self.model.last_linear  = out_dict

        self.parameters_dict = make_parameters_dict( model = self.model, filter_module_names = ['embedding'])
    
    def forward(self, x, use_penultimate=False):
        x = self.model.features(x)
        x = self.model.features_pooling(x)
        x = self.model.features_dropout(x)
        x = x.view(x.size(0), -1)
        if not use_penultimate:
            new_x = []
            for key,linear_map in self.model.last_linear.items():
                temp = torch.nn.functional.normalize(linear_map(x), dim=-1)
                new_x.append(temp)
            new_x = torch.cat(new_x,dim=1)
        else:
            # normalize the entire penultimate layer
            new_x = torch.nn.functional.normalize(x, p=2, dim=1)
        return new_x
        
