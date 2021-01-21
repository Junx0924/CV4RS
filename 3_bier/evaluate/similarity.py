import numpy as np

import scipy
from . import faissext

__all__ = [
    'pairwise_distance',
    'assign_by_euclidian_at_k'
]


FAISS_BACKEND = 'faiss'
FAISS_GPU_BACKEND = 'faiss-gpu'
_DEFAULT_BACKEND_ = FAISS_GPU_BACKEND
_backends_ = [FAISS_BACKEND, FAISS_GPU_BACKEND]

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """ 
    Get the index of top k sorted largest/smallest elements from array
    Args:
        array: 2-D numpy array of size 
        top_k:  top k elements
        axis: axis
        reverse: if set true, return index of k largest elements (sorted descend),
                if set false, return index of k smallest elements (sorted Ascend)
    Returns:
        top_sorted_indexes: index
    """
    if reverse:
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis),
                                  range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)
    # resort partition
    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    #top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_indexes
    
def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
        a: 2-D numpy array of size [number of data, feature dimension].
        squared: Boolean, whether or not to square the pairwise distances.
    Returns:
        pairwise_distances: 2-D numpy array of size [number of data, number of data].
    """
    a =  np.atleast_2d(a)
    if squared:
        pdist = scipy.spatial.distance.pdist(a,'sqeuclidean')
    else :
        pdist = scipy.spatial.distance.pdist(a, 'euclidean')
    

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pdist = np.clip(pdist,min=0.0)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - np.eye(pdist.shape)
    pdist = np.multiply(pdist, mask_offdiagonals)

    return pdist

def assign_by_euclidian_at_k(X, T, k, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    nns, _ = faissext.find_nearest_neighbors(X,
                                            k=k,
                                            gpu_id=None if backend != FAISS_GPU_BACKEND
                                            else torch.cuda.current_device()
    )
    return np.array([[T[i] for i in ii] for ii in nns])


def cluster_by_kmeans(X, nb_clusters, gpu_id=None, backend=_DEFAULT_BACKEND_):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    C = faissext.do_clustering(
        X,
        num_clusters = nb_clusters,
        gpu_ids = None if backend != FAISS_GPU_BACKEND
            else torch.cuda.current_device(),
        niter=100,
        nredo=5,
        verbose=1
    )
    return C

