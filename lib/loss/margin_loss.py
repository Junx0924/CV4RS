import torch
import torch.nn.functional as F
import numpy as np



class MarginLoss(torch.nn.Module):
    def __init__(self, nb_classes, beta=1.2,beta_lr =0.0005, margin=0.2, nu=0.1, is_beta_trainable=True,
 		 class_specific_beta=False, batchminner = None,**kwargs):
        """Initialize the class .

        Args:
            nb_classes (int):  Number of classes in the train dataset. Used to initialize class-specific boundaries beta
            beta (float, optional): margin beta. Defaults to 1.2.
            beta_lr (float, optional): learning rate for beta. Defaults to 0.0005.
            margin (float, optional): Margin between positive and negative pairs. Defaults to 0.2.
            nu (float, optional): Regularisation Parameter for beta values if they are learned.. Defaults to 0.1.
            is_beta_trainable (bool, optional): if set, beta is trainable. Defaults to True.
            class_specific_beta (bool, optional): if set, beta is trainable for each class. Defaults to False.
            batchminner (optional):  Defaults to None.
        """        
        super(MarginLoss, self).__init__()

        self.nb_classes = nb_classes
        self.class_specific_beta = class_specific_beta
        self.is_beta_trainable = is_beta_trainable
        if self.is_beta_trainable:
            if class_specific_beta:
                assert nb_classes is not None
                beta = torch.ones(nb_classes,dtype=torch.float32)*beta
            else:
                beta = torch.tensor([beta], dtype=torch.float32)
        
            self.beta = torch.nn.Parameter(beta)
            # Learning Rate for class margin parameters in MarginLoss
            self.beta_lr = beta_lr 
            self.nu = nu
        else:
            self.beta = beta
        self.margin = margin
        self.batchminner = batchminner

    def forward(self, feature, labels):
        """Calculate the loss .

        Args:
            feature (tensor): embedding vectors, shape(batchsize x sub embedding size)
            labels (tensor):  if batchminner is multiLabelSemihard, shape(batchsize, L); else category labels shape(batchsize, L)
        """        
        anchor_idx, pos_idx, neg_idx = self.batchminner(feature, labels)
        if isinstance( labels, torch.Tensor):  labels =  labels.detach()
        anchors = feature[anchor_idx] 
        positives = feature[pos_idx]
        negatives = feature[neg_idx]
        anchor_classes = labels[anchor_idx]

        if self.is_beta_trainable:
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
