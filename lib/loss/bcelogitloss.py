import torch
from torch import nn
import torch.nn.functional as F

class BCELogitLoss(nn.Module):
    def __init__(self, embed_dim,num_labels, bce_lr = 0.00001,weight=0):
        super(BCELogitLoss, self).__init__()
        self.num_labels = num_labels
        #self.pos_weight = weight*torch.ones([num_labels]) 
        self.regressor = torch.nn.Linear(embed_dim, num_labels).to(torch.float).cuda()
        self.lr = bce_lr

    def forward(self, feature, label):
        """
        Args:
            feature: tensor, shape (batchsize, embed_dim)
            label: tensor, shape (batchsize, num_labels), like [1,0,1,0,1,1,1]
        """
        input = self.regressor(feature)
        #loss = F.binary_cross_entropy_with_logits(input=input, target=label, weight=self.pos_weight)
        loss = F.binary_cross_entropy_with_logits(input=input, target=label)
        return loss
