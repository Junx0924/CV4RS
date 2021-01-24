import torch, torch.nn as nn
import torch.nn.functional as F

"""================================================================================================="""
class Adversarial(torch.nn.Module):
    def __init__(self,hidden_adversarial_size,decorrnet_lr=0.00001):
        super(Adversarial,self).__init__()
        self.proj_dim   = hidden_adversarial_size
        self.lr   =  decorrnet_lr

    # source: shape (batch_size, dim1)
    # target: shape (batch_size, dim2)
    def forward(self, source, target):
        assert len(source) == len(target)
        # l2 normalize feature
        source = F.normalize(source,p =2,dim =1)
        target = F.normalize(target,p =2,dim =1)
        
        # Apply gradient reversal on input embeddings.
        source =  torch.nn.functional.normalize(grad_reverse(source),dim=-1)  
        target =  torch.nn.functional.normalize(grad_reverse(target),dim=-1) 

        #Projection network
        regressor = torch.nn.Sequential(
                 torch.nn.Linear(source.shape[1], self.proj_dim),
                 torch.nn.ReLU(), 
                 torch.nn.Linear(self.proj_dim, target.shape[1])).to(torch.float).cuda()

        # Project one sub embedding to the space of the other (with normalization), then compute the correlation.
        source_proj = torch.nn.functional.normalize(regressor(source),dim=-1)
        similarity_loss  = -1.0*torch.mean(torch.mean((target*source_proj)**2,dim=-1))
           
        return similarity_loss



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
