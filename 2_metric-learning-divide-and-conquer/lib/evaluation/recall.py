from __future__ import division

import numpy as np
import itertools

def get_label(one_hot):
    labels = np.where(one_hot ==1)[0]
    return list(labels)

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = 0
    for t, y in zip(T,Y):
        t_labels = get_label(t) # true lables
        y_labels = [get_label(yy) for yy in y[:k]] # predicted labels
        y_labels = list(itertools.chain.from_iterable(y_labels))
        y_labels = np.unique(y_labels)
        ss = sum([1 for i in t_labels if i in y_labels])
        # only count the sample which has all of true lables in the predicted labels
        if ss == len(t_labels):
            s = s + 1
    return s / (1. * len(T))

