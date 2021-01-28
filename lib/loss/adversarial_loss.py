import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""================================================================================================="""
class Adversarial(torch.nn.Module):
    def __init__(self,hidden_adversarial_size,direction_dict,decorrnet_lr=0.00001, need_weight_loss = False):
        super(Adversarial,self).__init__()
       
        self.directions = direction_dict # {direction:{dim: str ,weight: float}}
        self.proj_dim   = hidden_adversarial_size
        self.lr   =  decorrnet_lr
        self.need_weight_loss = need_weight_loss
        #Projection network
        self.regressors = nn.ModuleDict()
        for key in self.directions.keys():
            source_dim, target_dim =  self.directions[key]['dim'].split('-')
            self.regressors[key] = torch.nn.Sequential(torch.nn.Linear(int(source_dim), self.proj_dim), 
                                                                torch.nn.ReLU(), 
                                                                torch.nn.Linear(self.proj_dim, int(target_dim))
                                                            ).to(torch.float).cuda()

    # source: shape (batch_size, dim1)
    # target: shape (batch_size, dim2)
    def forward(self, feature_dict):
        feature_names = np.unique(np.array([key.split('-') for key in self.directions.keys()]).reshape(-1))
        assert len(feature_names) == len([key for key in feature_dict.keys()])

        #Apply gradient reversal on input embeddings.
        adj_feature_dict = {key:torch.nn.functional.normalize(grad_reverse(features),dim=-1) for key, features in feature_dict.items()}
        #Project one embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = 0
        for key in self.directions.keys():
            source, target = key.split('-')
            source_data, target_data = adj_feature_dict[source], adj_feature_dict[target]
            weight = self.directions[key]['weight']
            regressor = self.regressors[key]
            sim_loss += -1.*weight*torch.mean(torch.mean((target_data*torch.nn.functional.normalize(regressor(source_data),dim=-1))**2,dim=-1))
        
        # get the regressor weights and bias
        if self.need_weight_loss:
            weight_loss = 0.0
            for regressor in self.regressors.values():
                for i in range(len(regressor)):
                    # relu layer has no weights and bias
                    if i !=1:
                        W_hat = regressor[i].weight.data
                        B_hat = regressor[i].bias.data
                        weight_loss += torch.mean((torch.sum(W_hat * W_hat, axis=1) - 1)**2) + torch.max(torch.tensor([0.0,torch.sum(B_hat * B_hat) - 1.0])) 
            weight_loss = weight_loss / len(self.regressors)
            # return similarity loss
            # return the weight and bias of regressor to penalize large weights and bias
            return sim_loss, weight_loss
        else:
            return sim_loss



### Gradient Reversal Layer
class GradRev(torch.autograd.Function):
    """
    Implements an autograd class to flip gradients during backward pass.
    """
    @staticmethod
    def forward(self, x):
        """
        Container which applies a simple identity function.

        Input:
            x: any torch tensor input.
        """
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        """
        Container to reverse gradient signal during backward pass.

        Input:
            grad_output: any computed gradient.
        """
        return (grad_output * -1.)

### Gradient reverse function
def grad_reverse(x):
    """
    Applies gradient reversal on input.

    Input:
        x: any torch tensor input.
    """
    #return GradRev()(x)
    return GradRev.apply(x)
