import numpy as np
from pathlib import Path
import tqdm 
from .filter import lanczosinterp2D
from .configs import RUNINFO, SUBJECT

def normalize(arr):
    assert len(arr) >0, arr.shape
    std = np.nanstd(arr,0)[None,:]
    std[std==0] = 1
    return (arr-np.nanmean(arr,0)[None,:])/std
    
def get_all_features(feature_dir, run_id):
    features = []
    frame_nums=[]
    files = [f for f in (feature_dir/run_id).glob('*.npy')]
    files.sort(key = lambda v: int(v.stem))
    for file in files:
        frame_num = int(file.stem)
        feature = np.load(file)
        features.append(feature)
        frame_nums.append(frame_num)
    features = np.array(features)
    return features,np.array(frame_nums)

def get_run_features(feature_dir, run_id, frame_nums):
    features = []
    valid_feature = None
    for frame_num in frame_nums:
        feature_path = feature_dir/run_id/f'{frame_num}.npy'
        if feature_path.exists():
            feature = np.load(feature_path).astype(np.float32)
            features.append(feature)
            valid_feature = feature
        else:
            features.append(None)
    assert valid_feature is not None
    
    features_ = []
    for feature in features:
        if feature is None:
            features_.append(np.zeros_like(valid_feature))
        else:
            features_.append(feature)
            
    features = features_
    return np.array(features)

def get_features(run_ids, TRs_list,feature_dir, time_diff=0, do_lanczos=True, recenter=True,
                demean_center=None,seq_len=1, frame_step=1,FB=0,EB=0, verbose=0):
    run_onsets = [0]
    total_features = []
    for run_id,TRs in zip(run_ids,TRs_list):
        frame_nums = (TRs+(time_diff*1)).astype(int)
        if recenter:
            frame_nums += 15
        if do_lanczos:
            features_old, frame_nums_old = get_all_features(feature_dir, run_id)
            
            features_seq = [lanczosinterp2D(features_old, frame_nums_old/15.0, (frame_nums)/15.0)]
            
            for si in range(1, seq_len):
                features = lanczosinterp2D(features_old, frame_nums_old/15.0, (frame_nums-frame_step*si)/15.0)
                features_seq = [features]+features_seq
            features = np.concatenate(features_seq,1)
        else:
            features_seq = []
            
            for si in range(seq_len):
                features = get_run_features(feature_dir, run_id, (frame_nums-frame_step*si))
                features_seq = [features]+features_seq
            features = np.concatenate(features_seq,1)
        if verbose==1:
            print(f"[{feature_dir.stem}] {run_id}: {features.shape}")
        features = features[FB:len(features)-EB]
        total_features.append(features)
        run_onsets.append(run_onsets[-1]+len(features))
    total_features = np.concatenate(total_features,0)
    total_features = total_features/((total_features.shape[1]*1.0)**.5)
    run_onsets = run_onsets[:-1]
    
    return total_features, run_onsets



def get_TRs_list(run_ids, data_dir):
    TRs_list = [np.load(data_dir/run_id/'Info'/'metainfo.npy', allow_pickle=True)[()]['TR'] for run_id in run_ids]
    return TRs_list

