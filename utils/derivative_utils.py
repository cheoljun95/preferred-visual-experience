import numpy as np
from pathlib import Path


def normalize_and_rescale(weights, scores):
    # weights: (d, voxel #)
    # scores: (scores,)
    
    if len(scores.shape)==1:
        scores=scores[None,:]
    norms = np.linalg.norm(weights,axis=0)
    norms[norms==0] = 1
    rescale_factors = scores.clip(0)**.5
    rescale_factors[np.isnan(rescale_factors)]=0
    rescale_factors = rescale_factors/norms
    weights = weights*rescale_factors
    
    return weights


def normalize(weights):
    # weights: (d, voxel #)
    
    norms = np.linalg.norm(weights,axis=0)
    norms[norms==0] = 1
    weights = weights/norms[None,:]
    
    return weights

