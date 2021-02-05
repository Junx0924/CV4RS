import numpy as np
import itertools

def get_label(multihot):
    labels = np.where(multihot ==1)[0]
    return list(labels)

def recall(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)

def precision(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    len1 = len(y_pred)
    if len1 == 0:
        return 0
    else:
        return len(i) / len1

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0
    else:
        return 2 * (p * r) / (p + r)

def select(metric,y_true, y_pred, k=1):
    s = 0
    # label is multi-hot encoding
    assert len(y_true[0])>1
    assert len(y_pred[0][0])>1
    
    y_true = [get_label(t) for t in y_true]
    y_pred = [[get_label(t) for t in y[:k]] for y in y_pred]
    if metric=='recall':
        for t,y in zip(y_true,y_pred):
            y = np.unique(list(itertools.chain.from_iterable(y)))
            s += recall(t,y)
    elif metric =='precision':
        for t,y in zip(y_true,y_pred):
            y = np.unique(list(itertools.chain.from_iterable(y)))
            s += precision(t,y)
    elif metric =='f1':
        for t,y in zip(y_true,y_pred):
            y = np.unique(list(itertools.chain.from_iterable(y)))
            s += f1(t,y)
    else:
        raise Exception('metric {} not available!'.format(metric))
    return s / len(y_true)