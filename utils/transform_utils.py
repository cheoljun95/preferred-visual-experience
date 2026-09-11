from pathlib import Path
from utils.configs import XFM_DIR
from scipy import sparse
import numpy as np

def to_fsavg6(data, subject, xfm_dir=XFM_DIR):
    xfm_dir = Path(xfm_dir)
    lh_xfm = sparse.load_npz(xfm_dir/f'{subject}_to_fsaverage_lh.npz')
    rh_xfm = sparse.load_npz(xfm_dir/f'{subject}_to_fsaverage_rh.npz')
    lh_fsavg_data = lh_xfm.dot(data.T).T
    rh_fsavg_data = rh_xfm.dot(data.T).T
    fsavg_data = np.concatenate([lh_fsavg_data[:,:40962], rh_fsavg_data[:,:40962]],-1)
    return fsavg_data


