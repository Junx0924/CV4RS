"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm



"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        if self.pars.num_in_channels >3:
            self.model = multichannel_resnet(self.pars.num_in_channels)
        else:
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.feature_dim = self.model.feature_dim
        self.model.last_linear = torch.nn.Linear(self.feature_dim, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        enc_out = x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)

        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        return x, enc_out


class multichannel_resnet(nn.Module):
    def __init__(self, num_in_channels=4):
        super().__init__()
       
        model =ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
        
        ##For reference: layers to use (in order):
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        
        # This is the most important line of code here. This increases the number of in channels for our network
        self.conv1 = self.increase_channels(model.conv1, num_in_channels)
        
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
        self.feature_dim = model.last_linear.in_features
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc()
        
        return x
        
    def increase_channels(self, m, num_channels=None, copy_weights=0):


        """
        takes as input a Conv2d layer and returns the a Conv2d layer with `num_channels` input channels
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