import sklearn.cluster
import sklearn.metrics.cluster

def calculate(ys, xs_clustered):
    """
    normalized_mutual_information
    """
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)
