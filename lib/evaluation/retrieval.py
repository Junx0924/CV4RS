import numpy as np
import itertools
from scipy.spatial import distance

def get_label(multihot):
    labels = np.where(multihot ==1)[0]
    return list(labels)

# implement according to paper: https://www.umbc.edu/rssipl/people/aplaza/Papers/Journals/2020.TGRS.GRN.pdf
# weighted mean average precision
# the number of labels shared between ground truth and predicted
def wmap(y_true,y_pred):
    s =0
    k = len(y_pred)
    deta =  [ 1 if len(set(y_true).intersection(yy))>0 else 0 for yy in y_pred]
    # average cumulative gains
    acg = [sum([len(set(y_true).intersection(yy))/r for yy in y_pred[:r]]) for r in range(1,k+1)]
    if sum(deta) >0:
        s = sum([ d[0]*d[1] for d in zip(deta,acg)])/sum(deta)
    return s



# mean average precision
def map(y_true,y_pred):
    s =0
    k = len(y_pred)
    relevant = [sum([1 for yy in y_pred[:r] if len(set(y_true).intersection(yy))>0]) for r in range(1,k+1)]
    deta =  [ 1 if len(set(y_true).intersection(yy))>0 else 0 for yy in y_pred]
    if sum(deta)>0:
        s = sum([ d[0]*d[1]/r for d,r in zip(zip(deta,relevant),range(1,k+1))])/sum(deta)
    return s

def select(metric,y_true, y_pred, k=8):
    s = 0
    # label is multi-hot encoding
    assert len(y_true[0])>1
    assert len(y_pred[0][0])>1

    if metric =='map':
        y_true = [get_label(t) for t in y_true]
        y_pred = [[get_label(t) for t in y[:k]] for y in y_pred]
        for t,y in zip(y_true,y_pred):
            s += map(t,y)
    elif metric =='wmap':
        y_true = [get_label(t) for t in y_true]
        y_pred = [[get_label(t) for t in y[:k]] for y in y_pred]
        for t,y in zip(y_true,y_pred):
            s += wmap(t,y)
    elif metric =='hamming':
        # HL evaluates the fraction of labels that are incorrectly predicted
        for t,y in zip(y_true,y_pred):
            s+= sum([distance.hamming(t, yy) for yy in y[:k]])/k
    else:
        raise Exception('metric {} not available!'.format(metric))
    return s / len(y_true)
    