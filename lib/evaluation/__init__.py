from . import examplebasedclassification
from . import labelbasedclassification


def select(metric,y_true, y_pred):
    """
    Args:
        metric
        y_true:  ground truth, matrix (n_samples, multi-hot)
        y_pred:  prediction, matrix (n_samples, multi-hot)
    """
    # example based metrics
    s =0.0
    if metric =='recall':
        s = examplebasedclassification.recall(y_true,y_pred)
    elif metric =='precision':
        s = examplebasedclassification.precision(y_true,y_pred)
    elif metric =='hamming_loss':
        s = examplebasedclassification.hammingLoss(y_true,y_pred)
    elif metric == 'accuracy':
        s = examplebasedclassification.accuracy(y_true,y_pred)
    elif metric =='subset_accuracy':
        s = examplebasedclassification.subsetAccuracy(y_true,y_pred)
    elif metric =='F_Measure':
        s = examplebasedclassification.fbeta(y_true,y_pred)
   
    # label based metric
    elif metric =='Mirco_F1':
        s = labelbasedclassification.fbetaMicro(y_true,y_pred)
    elif metric =='Macro_F1':
        s = labelbasedclassification.fbetaMacro(y_true,y_pred)
    else:
        raise Exception('Loss {} not available!'.format(metric))
    return s






