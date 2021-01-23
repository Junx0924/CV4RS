import torch
import torch.nn.functional as F
import numpy as np
from .batchminner.semihard import Semihard


class MarginLoss(torch.nn.Module):
    """Margin based loss.
    Parameters
    ----------
    nb_classes: int
        Number of classes in the train dataset.
        Used to initialize class-specific boundaries beta.
    margin : float
        Margin between positive and negative pairs.
    nu : float
        Regularization parameter for beta.
    class_specific_beta : bool
        Are class-specific boundaries beind used
    batchminer: return semihard triplets
    Inputs:
       E: embeddings, shape (batch_size, embedding dim)
       T: labels, shape (batch_size,1)
    Outputs:
        Loss value.
    """

    def __init__(self, nb_classes, beta=1.2, margin=0.2, nu=0.0,
 		 class_specific_beta=False, **kwargs):
        super(MarginLoss, self).__init__()

        self.nb_classes = nb_classes
        self.class_specific_beta = class_specific_beta
        if class_specific_beta:
            assert nb_classes is not None
            beta = torch.ones(nb_classes, dtype=torch.float32) * beta
        else:
            beta = torch.tensor([beta], dtype=torch.float32)
        self.beta = torch.nn.Parameter(beta)
        self.margin = margin
        self.nu = nu
        self.batchminer = Semihard()

    def forward(self, feature, labels):
        anchor_idx, anchors, positives, negatives = self.batchminer(feature, labels)
        anchor_classes = labels[anchor_idx]

        if anchor_classes is not None:
            if self.class_specific_beta:
                # select beta for every sample according to the class label
                beta = self.beta[anchor_classes]
            else:
                beta = self.beta
            beta_regularization_loss = torch.norm(beta, p=1) * self.nu
        else:
            beta = self.beta
            beta_regularization_loss = 0.0
        try:
            d_ap = ((positives - anchors)**2).sum(dim=1) + 1e-8
        except Exception as e:
            print(e)
            print(positives.shape, anchors.shape)
            raise e
        d_ap = torch.sqrt(d_ap)
        d_an = ((negatives - anchors)**2).sum(dim=1) + 1e-8
        d_an = torch.sqrt(d_an)

        pos_loss = F.relu(d_ap - beta + self.margin)
        neg_loss = F.relu(beta - d_an + self.margin)

        pair_cnt = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).type_as(pos_loss)
        loss = torch.sum(pos_loss + neg_loss)
        if pair_cnt > 0.0:
            # Normalize based on the number of pairs.
            loss = (loss + beta_regularization_loss) / pair_cnt
        return loss
