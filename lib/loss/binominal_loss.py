from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np

class BinomialLoss(nn.Module):
    def __init__(self, C =25,alpha=2.0, beta=0.5, eta_style=True,**kwargs):
        """
        Boosted bionminal loss
        Implement according to paper: https://arxiv.org/abs/1801.04815
        Args:
            C: parameter for binomial deviance.
            alpha: parameter for binomial deviance.
            beta: margin for binomial deviance.
        """
        super(BinomialLoss, self).__init__()
        self.C = 25
        self.alpha = alpha
        self.beta = beta
        self.eta_style = eta_style
        self.initial_acts =0.0 if eta_style == True else 0.5
        self.shrinkage = 1.0 if eta_style == True else 0.06
    
    def forward(self,normed_fvecs, T):
        """
        Args:
            normed_fvecs: multi-feature dictionary, each value contains sub embeddings [batchsize x sub embedding size]
            T: tensor, category labels, shape(batchsize, )
        """
        n = len(T)
        # init boosting_weights for each label pair
        boosting_weights = torch.ones(n*n).cuda()
        # Pairwise labels
        a = torch.cat(n*[torch.unsqueeze(T, 0)])
        b = torch.transpose(a, 0, 1)
        pairs = torch.flatten(a==b)*1.0
        m = 1.0 * pairs + (-self.C * (1.0 - pairs))
        W = torch.flatten(1.0 - torch.eye(n)).cuda()
        # initial weight for each label pair (not include itself)
        W = W * pairs / torch.sum(pairs) + W * (1.0 - pairs) / torch.sum(1.0 - pairs)
        
        loss =0.0
        acts = self.initial_acts
        Ds =[]
        i = 0.0
        for fvec in normed_fvecs.values():
            Ds.append(torch.matmul(fvec, fvec.t()))
            D = torch.flatten(Ds[-1])
            my_act = self.alpha* (D - self.beta)* m
            my_loss = torch.log(1.0 + torch.exp(-1.0*my_act))
            tmp = torch.sum(my_loss* boosting_weights * W)/ len(normed_fvecs)
            loss +=tmp
            if self.eta_style:
                nu = 2.0/( 1.0 + 1.0 + i)
                if self.shrinkage != 1.0:
                    acts = (1.0 - nu)* acts + nu * self.shrinkage * D
                    inputs = self.alpha * (acts - self.beta) * m
                    booster_loss = torch.sum(torch.log( 1.0 + torch.exp(-1.0*inputs)))
                    boosting_weights = -1*torch.autograd.grad(booster_loss, inputs,create_graph=True)[0].detach()
                else:
                    acts = (1.0- nu)* acts + nu * self.shrinkage * my_act
                    booster_loss = torch.sum(torch.log( 1.0 + torch.exp(-1.0*acts)))
                    boosting_weights = -1*torch.autograd.grad(booster_loss, acts,create_graph=True)[0].detach()
            else:
                # simpler variant of the boosting algorithm
                acts += self.shrinkage * ( D - self.beta) * self.alpha * m
                booster_loss = torch.sum(torch.log( 1.0 + torch.exp(-1.0*acts)))
                cls_weight = 1.0* pairs + (1.0 - pairs) * 2.0
                boosting_weights = -1* cls_weight* torch.autograd.grad(booster_loss, acts,create_graph=True)[0].detach()
            i +=1

        return loss