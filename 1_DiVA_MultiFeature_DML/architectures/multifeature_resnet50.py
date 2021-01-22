"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm

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
    def __init__(self, opt):
        super(Network, self).__init__()
        self.pars  = opt
        self.name  = opt.arch
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
           
        self.feature_dim =  self.model.last_linear.in_features
        # This increases the number of input channels for our network
        if opt.num_in_channels>3:
            self.model.conv1 = increase_channels(self.model.conv1, opt.num_in_channels)

        if 'frozen' is in opt.arch:
                child_counter = 0
                for child in model.children():
                    if child_counter == 0 and opt.dataset =="BigEarthNet":
                        continue
                    else:
                        for param in child.parameters():
                            param.requires_grad = False

        # add new linear layer
        out_dict = nn.ModuleDict()
        for mode in opt.diva_features:
            out_dict[mode] = torch.nn.Linear(self.feature_dim, opt.embed_dim)
        self.model.last_linear  = out_dict
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # for the new added linear layer
        out_dict = {}
        for key,linear_map in self.model.last_linear.items():
             # for distance weighted minner, normalize the embedings is required
            if 'normalize' in self.pars.arch:
                out_dict[key] = torch.nn.functional.normalize(linear_map(x), dim=-1)
            else:
                out_dict[key] = linear_map(x)
        return out_dict, x
        
