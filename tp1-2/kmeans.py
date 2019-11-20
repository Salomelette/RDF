import os
import numpy as np
from tools import *
from sift import *
from sklearn.cluster import KMeans


def compute_visual_dict(sift, n_clusters=1000, n_init=1, verbose=1):
    # reorder data
    dim_sift = sift[0].shape[-1]
    sift = [s.reshape(-1, dim_sift) for s in sift]
    sift = np.concatenate(sift, axis=0)
    # remove zero vectors
    keep = ~np.all(sift==0, axis=1)
    sift = sift[keep]
    # randomly pick sift
    ids, _ = compute_split(sift.shape[0], pc=0.05)
    sift = sift[ids]
    
    kmeans = KMeans(n_clusters=n_clusters,init='random',n_init=n_init,verbose=verbose)
    kmeans.fit(sift)
    cluster=np.vstack((kmeans.cluster_centers_,np.zeros(sift.shape[1])))
    return cluster