import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgb
from pathlib import Path
from scipy import sparse

colors=["000000","a6cee3","85b9d8","63a3cc","418ec0","1f78b4","216490","234f6c","375f79",
        "b2df8a","93d073","73c05b","53b044","33a02c","358b2d","36762e","689762",
        "fb9a99","f57a7a","ef5a5b","e31a1c","c41617","a51212","ad2828","b43c3c",
        "fdbf6f","fbae54","f99d38","f78c1c","f57a00","ee6207","e7490d","e95a23",
        "cab2d6","b295c7","9a78b8","8f64b9","8350b9","694097","4e2f75","5e4282",
        "918fe7","826ff0","734ef9","6037f1","4c20e9","3a13c3","27069d","3b1da6",
        "b7efc5","93e7a8","6ede8a","2dc653","22953f","1d7d35","17642a","2c723d",
        "ec83a4","e07191","d45e7d","bd4c67","a63a50","873143","672835","753c47"]

colors = colors*10
colors = ['#'+c for c in colors]
colors_rgb= np.stack([to_rgb(c) for c in colors])

colors20 =["#A6CEE3", "#1F78B4","#63A3CC","#234F6C",
                    "#B2DF8A","#33A02C","#73C05B","#36762E",
                    "#FB9A99","#E31A1C","#F06A6A","#A51212",
                    "#FDBF6F","#F57A00","#FE9C34","#E7490D",
                    "#CAB2D6","#8350B9","#9A78B8","#4E2F75"]
colors32= ["#A6CEE3","#1F78B4","#63A3CC","#234F6C",
                    "#B2DF8A","#33A02C","#73C05B","#36762E",
                    "#FB9A99","#E31A1C","#F06A6A","#A51212",
                    "#FDBF6F","#F57A00","#FE9C34","#E7490D",
                    "#CAB2D6","#8350B9","#9A78B8","#4E2F75",
                    "#918FE7","#4C20E9","#734EF9","#27069D",
                    "#B7EFC5","#2DC653","#6EDE8A","#17642A",
                    "#EC83A4","#A63A50","#D45E7D","#672835"]
yeo18 = ["#DCDCDC", "#a251ac", "#fd372d", "#779ac0",
              "#43e6ce", "#72b86f", "#409832", "#e775ff", "#feb4e9",
              "#f7fdc9",  "#a7b16c", "#a1acd7", "#efb942","#b6687e",
              "#4066f6", "#383ba5", "#fef735", "#d9707c"]

yeo18_rgb = np.stack([to_rgb(c) for c in yeo18])
colors32_rgb= np.stack([to_rgb(c) for c in colors32])
colors16_rgb= np.stack([to_rgb(colors32[ci]) for ci in range(0,32,2)])
colors20_rgb= np.stack([to_rgb(c) for c in colors20])

def plot_hist2d(scores_1, scores_2, bins=100, cmin=1, vmin=None, vmax=None,
                ax=None, norm=LogNorm(), colorbar=True, **kwargs):
    vmin = min(scores_1.min(), scores_2.min()) if vmin is None else vmin
    vmax = max(scores_1.max(), scores_2.max()) if vmax is None else vmax
    bins = np.linspace(vmin, vmax, bins) if isinstance(bins, int) else bins
    ax = plt.gca() if ax is None else ax

    h = ax.hist2d(scores_1, scores_2, bins=bins, cmin=cmin, norm=norm,
                  **kwargs)
    if colorbar:
        cbar = ax.figure.colorbar(h[3], ax=ax)
        cbar.ax.set_ylabel('number of voxels')

    ax.plot([vmin, vmax], [vmin, vmax], color='k', linewidth=0.5)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid()
    return ax


def to_full_fsaverage(data,xfm_dir='/home/jlg/cheoljun/fsavg_xfm',fsres=5):
    xfm_dir = Path(xfm_dir)
    xfm_lh, xfm_rh = [sparse.load_npz(xfm_dir/f'fsaverage{fsres}_to_fsaverage_{hemi}.npz') for hemi in ['lh', 'rh']]
    data_lh, data_rh = data[:data.shape[0]//2],data[data.shape[0]//2:]
    fsavg_lh, fsavg_rh = xfm_lh.dot(data_lh.astype("float32")), xfm_rh.dot(data_rh.astype("float32"))
    data_fsavg = np.concatenate([fsavg_lh, fsavg_rh])
    return data_fsavg

def to_fsavg6(data, subject):
    xfm_dir = Path('/home/jlg/cheoljun/driving_surf2fsavg_xfm/')
    lh_xfm = sparse.load_npz(xfm_dir/f'{subject}_to_fsaverage_lh.npz')
    rh_xfm = sparse.load_npz(xfm_dir/f'{subject}_to_fsaverage_rh.npz')
    lh_fsavg_data = lh_xfm.dot(data.T).T
    rh_fsavg_data = rh_xfm.dot(data.T).T
    fsavg_data = np.concatenate([lh_fsavg_data[:,:40962], rh_fsavg_data[:,:40962]],-1)
    return fsavg_data