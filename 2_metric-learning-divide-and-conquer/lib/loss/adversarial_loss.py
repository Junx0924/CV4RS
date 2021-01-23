import torch, torch.nn as nn
import itertools


"""================================================================================================="""

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class adversarial(torch.nn.Module):
    def __init__(self, sub_embed_sizes,hidden_adversarial_size,lambda_weight):
        super(adversarial,self).__init__()
        self.sub_dim  = sub_embed_sizes
        self.weights    = lambda_weight
        self.proj_dim   = hidden_adversarial_size

        temp =  list(itertools.combinations(sub_embed_sizes,2))
        self.directions = [ str(i[0])+'-'+str(i[1])  for i in temp]

        #Projection network
        self.regressors = nn.ModuleDict()
        for direction in self.directions:
            start_dim,end_dim =  direction.split('-')
            self.regressors[direction] = torch.nn.Sequential(
                 torch.nn.Linear(int(start_dim), self.proj_dim),
                 torch.nn.ReLU(), 
                 torch.nn.Linear(self.proj_dim, int(end_dim))).to(torch.float).to(opt.device)

        #Learning Rate for Projection Network
        self.lr        = opt.diva_decorrnet_lr

    # feature: shape (batch_size, sz_embedding_size)
    def forward(self, feature):
        # l2 normalize feature
        feature_dict ={}
        for i in range(len(self.sub_dim)):
            start = int(sum(self.sub_dim[:i]))
            stop = int(start + self.sub_dim[i])
            fvecs = F.normalize(feature[:, start:stop],p =2,dim =1)
            feature_dict[str(self.sub_dim[i])]= fvecs

        # Apply gradient reversal on input embeddings.
        adj_feature_dict = {key:torch.nn.functional.normalize(grad_reverse(features),dim=-1) for key, features in feature_dict.items()}
        # Project one sub embedding to the space of the other (with normalization), then compute the correlation.
        sim_loss = 0
        for weight, direction in zip(self.weights, self.directions):
            source, target = direction.split('-')
            sim_loss += -1.*weight*torch.mean(torch.mean((adj_feature_dict[target]*torch.nn.functional.normalize(self.regressors[direction](adj_feature_dict[source]),dim=-1))**2,dim=-1))
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
