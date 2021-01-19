import torch, torch.nn as nn
from .sampler import Sampler


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
        Are class-specific boundaries beind used?

    Inputs:
        - anchors: sampled anchor embeddings.
        - positives: sampled positive embeddings.
        - negatives: sampled negative embeddings.
        - anchor_classes: labels of anchors. Used to get class-specific beta.

    Outputs:
        Loss value.
    """
    def __init__(self, nb_classes, beta=1.2, margin=0.2, nu=0.0,
 		 class_specific_beta=False, **kwargs):
        super(MarginLoss, self).__init__()
        self.n_classes         = nb_classes

        self.margin             = margin
        self.nu                 = nu
        self.beta_constant      = class_specific_beta
        self.beta_val           = beta

        if self.beta_constant:
            self.beta = beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(self.n_classes)*self.beta_val)

        self.sampler = Sampler()

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets = self.sampler(batch, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0],:], 'Positive':batch[triplet[1],:], 'Negative':batch[triplet[2]]}

                pos_dist = ((train_triplet['Anchor']-train_triplet['Positive']).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((train_triplet['Anchor']-train_triplet['Negative']).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss = torch.sum(pos_loss+neg_loss)
            else:
                loss = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: loss = loss + beta_regularisation_loss.to(torch.float).to(d_ap.device)
        else:
            loss = torch.tensor(0.).to(torch.float).to(batch.device)

        return loss
