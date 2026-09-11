import numpy as np
import scipy.stats
from sklearn.utils.validation import check_random_state
from himalaya.scoring import r2_score
import torch
import tqdm

def generate_leave_one_run_out(n_samples, run_onsets, random_state=None,
                               n_runs_out=1):
    """Generate a leave-one-run-out split for cross-validation.

    Generates as many splits as there are runs.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the training set.
    run_onsets : array of int of shape (n_runs, )
        Indices of the run onsets.
    random_state : None | int | instance of RandomState
        Random state for the shuffling operation.
    n_runs_out : int
        Number of runs to leave out in the validation set. Default to one.

    Yields
    ------
    train : array of int of shape (n_samples_train, )
        Training set indices.
    val : array of int of shape (n_samples_val, )
        Validation set indices.
    """
    random_state = check_random_state(random_state)

    n_runs = len(run_onsets)
    # With permutations, we are sure that all runs are used as validation runs.
    # However here for n_runs_out > 1, a run can be chosen twice as validation
    # in the same split.
    all_val_runs = np.array(
        [random_state.permutation(n_runs) for _ in range(n_runs_out)])

    all_samples = np.arange(n_samples)
    runs = np.split(all_samples, run_onsets[1:])
    if any(len(run) == 0 for run in runs):
        raise ValueError("Some runs have no samples. Check that run_onsets "
                         "does not include any repeated index, nor the last "
                         "index.")

    for val_runs in all_val_runs.T:
        train = np.hstack(
            [runs[jj] for jj in range(n_runs) if jj not in val_runs])
        val = np.hstack([runs[jj] for jj in range(n_runs) if jj in val_runs])
        yield train, val


def explainable_variance(data, bias_correction=True, do_zscore=True):
    """Compute explainable variance for a set of voxels.

    Parameters
    ----------
    data : array of shape (n_repeats, n_times, n_voxels)
        fMRI reponses of the repeated test set.
    bias_correction: bool
        Perform bias correction based on the number of repetitions.
    do_zscore: bool
        z-score the data in time. Only set to False if your data time courses
        are already z-scored.

    Returns
    -------
    ev : array of shape (n_voxels, )
        Explainable variance per voxel.
    """
    if do_zscore:
        data = scipy.stats.zscore(data, axis=1)

    mean_var = data.var(axis=1, dtype=np.float64, ddof=1).mean(axis=0)
    var_mean = data.mean(axis=0).var(axis=0, dtype=np.float64, ddof=1)
    ev = var_mean / mean_var

    if bias_correction:
        n_repeats = data.shape[0]
        ev = ev - (1 - ev) / (n_repeats - 1)
    return ev

'''
def permutation_test(pred, target, n_repeat=5000):
    # pred: (T, V)
    # target: (T, V)
        
    scores = r2_score(pred, target)
    idxs = np.arange(pred.shape[0])
    cnt = torch.zeros_like(scores)
    for _ in tqdm.tqdm(range(n_repeat)):
        np.random.shuffle(idxs)
        perm_scores = r2_score(pred[idxs], target)
        cnt += perm_scores>scores
    
    pvals = cnt/n_repeat
    
    return pvals
'''

def permutation_test(pred, target, n_repeat=1000, block_size=10, score_fn=r2_score):
    # pred: (T, V)
    # target: (T, V)

    scores = score_fn(pred, target)
    T = pred.shape[0]

    # Build block indices
    n_blocks = T // block_size
    T_blocked = n_blocks * block_size  # trim trailing frames if not divisible

    cnt = torch.zeros_like(scores)
    for _ in tqdm.tqdm(range(n_repeat)):
        block_idxs = np.random.permutation(n_blocks)
        # Expand blocks back to frame indices
        idxs = np.concatenate([
            np.arange(b * block_size, (b + 1) * block_size)
            for b in block_idxs
        ])
        perm_scores = score_fn(pred[idxs], target[:T_blocked])
        cnt += perm_scores > scores

    pvals = cnt / n_repeat
    return pvals
