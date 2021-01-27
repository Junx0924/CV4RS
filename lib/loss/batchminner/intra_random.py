import numpy as np, torch
import itertools as it
import random

class intra_random():
    def __call__(self, batch, labels):
        if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
        unique_classes   = np.unique(labels)
        indices          = np.arange(len(batch))
        class_dict       = {i:indices[labels==i] for i in unique_classes}

        anchors, positives, negatives = [], [],[]
        # y_anchor = y_positive = y_negative
        for cls in np.random.choice(list(class_dict.keys()), len(labels), replace=True):
            a,p,n = np.random.choice(class_dict[cls], 3, replace=True)
            anchors.append(a)
            positives.append(p)
            negatives.append(n)
        return anchors, positives, negatives
