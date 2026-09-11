import numpy as np
from pathlib import Path
import tqdm 
from utils.delayer import Delayer
from utils.vm_utils import generate_leave_one_run_out, permutation_test
from utils.feature_utils import get_TRs_list, get_features, normalize
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge
from himalaya.backend import set_backend
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from himalaya.scoring import r2_score, correlation_score
from utils.configs import RUNINFO, SUBJECT, FB, EB
import argparse
from scipy.stats import pearsonr
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--subject",type=str, default=None)
parser.add_argument("--save_path",type=str)
parser.add_argument("--bold_path",type=str)
parser.add_argument("--feature_path",type=str)
parser.add_argument("--info_data_path",type=str)
parser.add_argument("--feature_name", type=str, default='vae-2sec')
parser.add_argument('--save_coef', action='store_true', default=True)
parser.add_argument("--seq_len", type=int, default=1)
parser.add_argument("--frame_step", type=int, default=4)

if __name__ == '__main__': 
    args = parser.parse_args()
    info_data_path = Path(args.info_data_path)
    feature_path = Path(args.feature_path)
    feature_name = args.feature_name if args.feature_name is not None else feature_path.stem
    delays=[1,2,3,4]
    save_path =  args.save_path
    bold_dir = Path(args.bold_path)
    Path(save_path).mkdir(exist_ok=True)
    task_tag = ''
    if args.subject is None:
        subjects = SUBJECT
    else:
        subjects = [args.subject]
        
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir/f'{task_tag}{feature_name}'
    save_dir.mkdir(exist_ok=True)
    backend = set_backend("torch_cuda", on_error="warn")

    for subject in subjects:
    
        subj_save_dir = save_dir/subject
        subj_save_dir.mkdir(exist_ok=True)

        train_runs = RUNINFO[subject]['train_run_ids']
        test_runs = RUNINFO[subject]['test_run_ids']

        ### Data preparation
        print(f'loading BOLD responses...')

        Y_train = []
        run_onsets = [0]
        for run_id in train_runs:
            y = np.load(bold_dir/f'{run_id}_bold.npy').astype(np.float32)
            y = normalize(y)
            y[np.isnan(y)] = 0
            y = y[FB:-EB]
            Y_train.append(y)
            run_onsets.append(run_onsets[-1]+len(y))
        run_onsets = run_onsets[:-1]
        Y_train = np.concatenate(Y_train,0)
        X_train, run_onsets = get_features(train_runs, get_TRs_list(train_runs, info_data_path),
                                           feature_path, time_diff=0,recenter=args.recenter,
                                           do_lanczos=True, seq_len=args.seq_len, frame_step=args.frame_step,
                                           FB=FB,EB=EB)
        X_train = X_train.astype(np.float32)
        
        ### Model preparation
        cv = generate_leave_one_run_out(X_train.shape[0], run_onsets)
        alphas = np.logspace(2, 15, 50)
        kernel_ridge_cv = KernelRidgeCV(
                            alphas=alphas, cv=cv,
                            solver_params=dict(n_targets_batch=500,
                                               n_alphas_batch=5,
                                               n_targets_batch_refit=100),)
        delayer = Delayer(delays=delays)
        pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            delayer,
            kernel_ridge_cv,
        )
        
        ### Model fitting
        pipeline.fit(X_train, Y_train)
        
        
        ### Model testing
        X_test, _ = get_features(test_runs, get_TRs_list(test_runs, info_data_path),
                                 feature_path, time_diff=0, recenter=args.recenter,do_lanczos=True,
                                 seq_len=args.seq_len, frame_step=args.frame_step,
                                 FB=FB,EB=EB)
        X_test = X_test.astype(np.float32)

        Y_test = []
        for run_id in test_runs:
            y = np.load(bold_dir/f'{run_id}_bold.npy').astype(np.float32)
            y = normalize(y)
            y[np.isnan(y)] = 0
            y = y[FB:-EB]
            Y_test.append(y)
            
        Y_test = np.concatenate(Y_test,0)
        Y_test = backend.to_numpy(Y_test)
        pred_test = backend.to_numpy(pipeline.predict(X_test))
        test_r2scores = backend.to_numpy(r2_score(Y_test, pred_test))
        test_rscores = backend.to_numpy(correlation_score(Y_test, pred_test))
        r2pvals = backend.to_numpy(permutation_test(pred_test, Y_test, block_size=10, score_fn=r2_score))
        rpvals = backend.to_numpy(permutation_test(pred_test, Y_test, block_size=10, score_fn=correlation_score))
        
        ### Saving results
        np.save(subj_save_dir/f'Y_test.npy',Y_test)
        np.save(subj_save_dir/f'pred_test.npy',pred_test)
        np.save(subj_save_dir/f'r2pvals.npy',r2pvals)
        np.save(subj_save_dir/f'rpvals.npy',rpvals)
        np.save(subj_save_dir/f'r2scores.npy',test_r2scores)
        np.save(subj_save_dir/f'rscores.npy',test_rscores)
        best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
        np.save(subj_save_dir/f'best_alphas.npy', best_alphas)
        np.save(subj_save_dir/f'alphas.npy', alphas)
        if args.save_coef:
            primal_coef =pipeline[-1].get_primal_coef()
            primal_coef = backend.to_numpy(primal_coef)
            primal_coef = delayer.reshape_by_delays(primal_coef, axis=0)
            np.save(subj_save_dir/f'coef.npy', primal_coef)
