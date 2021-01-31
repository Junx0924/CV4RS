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

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        # set pretrained = False to stop downloading weight from internet
        model = torchvision.models.resnet50(pretrained = False)
        state_dict = torch.load(config['pretrained_weights_file'])
        model.load_state_dict(state_dict)  

        self.feature_dim = model.fc.in_features
        # This increases the number of input channels for our network
        if config["dataset_selected"] =="BigEarthNet":
            input_channels = 12
            self.conv1 = increase_channels(model.conv1, input_channels)
        else:
            self.conv1 = model.conv1

        self.features = torch.nn.Sequential(
            self.conv1, model.bn1, model.relu, model.maxpool,
           model.layer1, model.layer2, model.layer3, model.layer4)

        if config['frozen']:
            child_counter = 0
            for child in self.features:
                if child_counter == 0 and config['dataset_selected'] =="BigEarthNet":
                    continue
                else:
                    for param in child.parameters():
                        param.requires_grad = False

        # add pooling layer
        k_s = 4 if config['dataset_selected'] =="BigEarthNet" else 7
        self.features_pooling = torch.nn.AvgPool2d(k_s,stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        # add dropout layer
        self.features_dropout = torch.nn.Dropout(0.01)

        # add new embedding layer
        out_dict = nn.ModuleDict()
        sub_embed_sizes = config['sub_embed_sizes']

        if 'diva_features' in config.keys():
            embed_names = config['diva_features']
        else:
            embed_names = ['embed_' + str(i+1) for i in range(len(sub_embed_sizes))]

        for name,size in zip(embed_names,sub_embed_sizes):
            out_dict[name] = torch.nn.Linear(self.feature_dim, size)
        
        self.last_linear  = out_dict

        #Separates model parameters into 'backbone' and 'embedding'
        self.parameters_dict = {'backbone':[],'embedding':[]}
        for child in self.features:
            for param in child.parameters():
                self.parameters_dict['backbone'].append(param)
        for child in self.last_linear.values():
            for param in child.parameters():
                self.parameters_dict['embedding'].append(param)
    
    def forward(self, x, use_penultimate=False):
        x = self.features(x)
        x = self.features_pooling(x)
        x = self.features_dropout(x)
        x = x.view(x.size(0), -1)
        if not use_penultimate:
            new_x = []
            for key,linear_map in self.last_linear.items():
                temp = torch.nn.functional.normalize(linear_map(x), dim=-1)
                new_x.append(temp)
            new_x = torch.cat(new_x,dim=1)
        else:
            # normalize the entire penultimate layer
            new_x = torch.nn.functional.normalize(x, p=2, dim=1)
        return new_x
        
