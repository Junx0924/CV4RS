import torch
from torch import nn
from torch.autograd import Function

import math

eps = 1e-8

# refer to the paper: https://www.umbc.edu/rssipl/people/aplaza/Papers/Journals/2020.TGRS.GRN.pdf
class NCACrossEntropy(nn.Module): 
    ''' \sum_{j=C} log(p_{ij})
        Store all the labels of the dataset.
        Only pass the indexes of the training instances during forward. 
    '''
    def __init__(self, labels, margin=0):
        """
        Args:
            labels: all the labels for training dataset
                    tensor shape [len(training dataset), num_classes], multi-hot encoding like [1,0,1,0]
            margin: classification margin
        """
        super(NCACrossEntropy, self).__init__()
        self.register_buffer('labels_sim', torch.FloatTensor(labels.size(0),labels.size(0)))

        num_classes = labels.size(1)
        labels_sim = torch.mm(labels,labels.t())
        # scale the similarity to [0,1]
        self.labels_sim = labels_sim/num_classes
        self.margin = margin

    def forward(self, embed_sim, indexes):
        """
        Args:
            embed_sim : tensor,shape [batchsize x len(training dataset)]
                        the similarity mat between the mini-batch embeddings and the memory bank
            indexes: indexes for the mini-batch
        return:
            loss
        """
        batchSize = embed_sim.size(0)
        N = embed_sim.size(1)
        assert embed_sim.size(1) == self.labels_sim.size(1)
        
        # similarity labels matrix for currect batch
        label_sim = torch.index_select(self.labels_sim, 0, indexes.data).view(batchSize, N) 

       # exclude itself
        exp = torch.exp(embed_sim)
        exp.data.scatter_(1, indexes.data.view(-1,1), 0)

        p = torch.mul(exp, label_sim).sum(dim=1)
        Z = exp.sum(dim=1)

        Z_exclude = Z - p
        p = p.div(math.exp(self.margin))
        Z = Z_exclude + p

        prob = torch.div(p, Z)
        prob_masked = torch.masked_select(prob, prob.ne(0))
        loss = prob_masked.log().sum(0)

        return - loss / batchSize