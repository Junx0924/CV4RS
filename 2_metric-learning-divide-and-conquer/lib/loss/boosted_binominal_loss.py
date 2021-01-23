import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class boosted_binominal(torch.nn.Module):
    def __init__(self, sub_embed_sizes,shrinkage=0.06,lambda_div=0.0, C=25, alpha=2.0, beta=0.5, initial_acts=0.5,eta_style=False):
        super(boosted_binominal, self).__init__()
        self.sub_dim = sub_embed_sizes
        self.shrinkage = shrinkage
        self.lambda_div = lambda_div
        self.alpha = alpha
        self.beta = beta
        self.initial_acts = initial_acts
        self.eta_style = eta_style
        self.C = C


    # feature: shape (batch_size, sz_embedding_size)
    def forward(self, feature, labels):
        shape = feature.shape
        num_learners = len(self.sub_dim)
        # Pairwise labels
        a = torch.cat(shape[0]*[torch.unsqueeze(labels, 0)])
        b = torch.transpose(a, 0, 1)
        pairs = torch.flatten(a==b)*1.0

        m = 1.0 * pairs + (-self.C * (1.0 - pairs))
        W = torch.flatten(1.0- torch.eye(shape[0]))
        if torch.cuda.is_available(): W = W.cuda()
        W = W * pairs / torch.sum(pairs) + W * (1.0 - pairs) / torch.sum(1.0 - pairs)

        # l2 normalize feature
        normed_fvecs = []
        for i in range(len(self.sub_dim)):
            start = int(sum(self.sub_dim[:i]))
            stop = int(start + self.sub_dim[i])
            fvecs = F.normalize(feature[:, start:stop],p =2,dim =1)
            normed_fvecs.append(fvecs)

        # boosting_weights_init
        boosting_weights = torch.ones(shape[0] * shape[0])
        if torch.cuda.is_available(): boosting_weights = boosting_weights.cuda()
        loss =0.0
        acts = torch.ones(shape[0]*shape[0])*self.initial_acts
        if torch.cuda.is_available(): acts = acts.cuda()
        Ds = []
        for i in range(len(self.sub_dim)):
            fvec = normed_fvecs[i]
            Ds.append(torch.matmul(fvec, torch.transpose(fvec,0,1)))
            D = torch.flatten(Ds[-1])
            my_act = self.alpha * (D - self.beta) * m
            my_loss = torch.log(torch.exp(-1.0*my_act) + 1.0)
            tmp = torch.mean(my_loss * boosting_weights * W) / num_learners
            loss += tmp

            if self.eta_style:
                nu = 2.0 / (1.0 + 1.0 + i)
                if self.shrinkage != 1.0:
                    acts = (1.0 - nu) * acts + nu * self.shrinkage * D
                    inputs = Variable(self.alpha * (acts - self.beta) * m,  requires_grad = True)
                    booster_loss = torch.log(torch.exp(-(inputs)) + 1.0)
                    boosting_weights =  -1*torch.autograd.grad(torch.sum(booster_loss),inputs)
                else:
                    acts = Variable(acts,  requires_grad = True)
                    acts = (1.0 - nu) * acts + nu * self.shrinkage * my_act
                    booster_loss = torch.log(tf.exp(-acts) + 1.0)
                    boosting_weights =  -1*torch.autograd.grad(torch.sum(booster_loss),acts)
            else:
                # simpler variant of the boosting algorithm.
                acts = Variable(acts,  requires_grad = True)
                acts  = acts + self.shrinkage * my_act
                booster_loss = torch.log(torch.exp(-(acts)) + 1.0)
                cls_weight = 1.0 * pairs + (1.0 - pairs) * 2.0
                boosting_weights =  -1*cls_weight*torch.autograd.grad(torch.sum(booster_loss),acts)
        return loss
               