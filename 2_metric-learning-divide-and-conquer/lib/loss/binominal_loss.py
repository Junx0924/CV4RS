from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class BinomialLoss(nn.Module):
    def __init__(self, alpha=40, beta=2.0, margin=0.5, hard_mining=True,  **kwargs):
        super(BinomialLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining
    
    # inputs: normalized, shape (batchsize, embedding dim)
    # targets: labels, shape(batchsize,)
    def forward(self, inputs, targets):
        assert len(inputs) == len(targets)
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        c = 0

        pair_grad=torch.zeros(n,n).cuda() # record grad for each pair
        pair_loss=torch.zeros(n,n).cuda() # record loss for each pair
        for i in range(n):
            pos_ind = torch.where(targets==targets[i])[0]
            # remove itself
            pos_pair_ = sim_mat[i][pos_ind]
            ind_select = torch.where(pos_pair_<1)[0]
            pos_ind = pos_ind[ind_select]
            pos_pair_ = sim_mat[i][pos_ind]

            neg_ind = torch.where(targets!=targets[i])[0]
            neg_pair_ = sim_mat[i][neg_ind]

            # sorting ascending, get the index
            pos_pair_,pos_sort = torch.sort(pos_pair_)
            neg_pair_,neg_sort = torch.sort(neg_pair_)
            pos_ind =pos_ind[pos_sort]
            neg_ind =neg_ind[neg_sort]

            if self.hard_mining:
                
                hard_neg_ind = torch.where(neg_pair_ + 0.1 > pos_pair_[0])[0]
                neg_ind = neg_ind[hard_neg_ind]

                hard_pos_ind = torch.where(pos_pair_ - 0.1 <  neg_pair_[-1])[0]
                pos_ind = pos_ind[hard_pos_ind]
                
                if len(neg_ind) < 1 or len(pos_ind) < 1:
                    c += 1
                    continue 
                
                pos_pair = Variable(sim_mat[i][pos_ind], requires_grad = True)
                neg_pair = Variable(sim_mat[i][neg_ind], requires_grad = True)
                pos_loss = 2.0/self.beta * torch.log(1 + torch.exp(-self.beta*(pos_pair - self.margin)))
                neg_loss = 2.0/self.alpha * torch.log(1 + torch.exp(self.alpha*(neg_pair - self.margin)))

            else:  
                pos_pair = Variable(sim_mat[i][pos_ind], requires_grad = True)
                neg_pair = Variable(sim_mat[i][neg_ind], requires_grad = True) 

                pos_loss = torch.log(1 + torch.exp(-self.beta*(pos_pair - self.margin)))
                neg_loss = torch.log(1 + torch.exp(self.alpha*(neg_pair - self.margin)))
            # get the gradient of pos_loss and neg_loss respect to input similarity
            pos_grad = torch.autograd.grad(torch.mean(pos_loss),pos_pair,retain_graph=True)[0]
            neg_grad = torch.autograd.grad(torch.mean(neg_loss),neg_pair,retain_graph=True)[0]

            if len(neg_pair) == 0:
                c += 1
                continue
            
            pair_loss[i][pos_ind] = pos_loss
            pair_loss[i][neg_ind] = neg_loss
            pair_grad[i][pos_ind] = pos_grad
            pair_grad[i][neg_ind] = neg_grad
            
        return pair_loss.flatten(), pair_grad.flatten()