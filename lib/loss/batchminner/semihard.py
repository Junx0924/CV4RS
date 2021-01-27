from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else res.clamp(min=eps).sqrt()


def topk_mask(input, dim, K=10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim=dim, **kwargs)[1]
    return torch.zeros_like(input.data).scatter(dim, index, 1.0)


class Semihard(nn.Module):
    """
    Sample for each anchor negative examples
        are K closest points on the distance >= cutoff
    Inputs:
        x: embeddings, shape (batch_size, embedding dim)
        labels: labels, shape (batch_size,1)
    Outputs:
        - a_indices: indices of anchors.
        - p_indices: indices of positive embeddings.
        - n_indices: indices of negative embeddings.
    """

    def __init__(self, cutoff=0.5, infinity=1e6, eps=1e-6):
        super(Semihard, self).__init__()
        self.cutoff = cutoff
        self.infinity = infinity
        self.eps = eps

    def forward(self, x, labels):
        """
        x: input tensor of shape (batch_size, embed_dim)
        labels: tensor of class labels of shape (batch_size,)
        """
        d = pdist(x)
        pos = torch.eq(
            *[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]
        ).type_as(d) - (torch.eye( len(d))).type_as(d)
        num_neg = int(pos.data.sum()) // len(pos)
        neg = topk_mask(
            d + self.infinity * ((pos > 0) + (d < self.cutoff)).type_as(d),
            dim=1,
            largest=False,
            K=num_neg
        )

        a_indices = []
        p_indices = []
        n_indices = []

        for i in range(len(d)):
            a_indices.extend([i] * num_neg)
            p_indices.extend(
                np.atleast_1d(pos[i].nonzero().squeeze().cpu().numpy())
            )
            n_indices.extend(
                np.atleast_1d(neg[i].nonzero().squeeze().cpu().numpy())
            )

            if len(a_indices) != len(p_indices) or len(a_indices) != len(n_indices):
                min_len = min(map(len, [a_indices, p_indices, n_indices]))
                a_indices = a_indices[:min_len]
                p_indices = p_indices[:min_len]
                n_indices = n_indices[:min_len]

        assert len(a_indices) == len(p_indices) == len(n_indices), \
                '{}, {}, {}'.format(
                    *map(len, [a_indices, p_indices, n_indices])
                )

        return a_indices, p_indices, n_indices
