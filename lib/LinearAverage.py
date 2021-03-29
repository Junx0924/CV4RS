import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, embed, indexes, memory, params):
        """Perform forward pass of the mini batch

        Args:
            embed (torch tensor): embeddings of the mini batch
            indexes (list):  shape (N, embeding dim)
            memory (torch tensor): memory bank
            params (float): temperature and momentum

        Returns:
            torch tensor: the similarity mat between the mini-batch embeddings and the memory bank
        """        
        T = params[0].item()
        batchSize = embed.size(0)

        # inner product
        out = torch.mm(embed.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(embed, memory, indexes, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        embed, memory, indexes, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(embed)

        # update the non-parametric data
        weight_pos = memory.index_select(0, indexes.data.view(-1)).resize_as_(embed)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(embed.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, indexes, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, embed_dim, N, T=0.05, momentum=0.5):
        """Build the memory bank for Scalable Neighborhood Component Analysis

        Args:
            embed_dim (int): dimension of embedding vector
            N (int): the length of dataset
            T (float, optional): temperature, Defaults to 0.05.
            momentum (float, optional): momentum for non-parametric updates. Defaults to 0.5.
        """        
        super(LinearAverage, self).__init__()
        self.nLem = N
        self.register_buffer('params',torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(embed_dim/3)
        self.register_buffer('memory', torch.rand(N, embed_dim).mul_(2*stdv).add_(-stdv))

    def forward(self, embed, indexes):
        """Returns the similarity matrices of the given embeddings .

        Args:
            embed (torch tensor): embeddings of the mini batch
            indexes (list): index of the mini batch

        Returns:
            torch tensor: the similarity mat between the mini-batch embeddings and the memory bank
        """        
        embed_sim = LinearAverageOp.apply(embed, indexes, self.memory, self.params)
        return embed_sim