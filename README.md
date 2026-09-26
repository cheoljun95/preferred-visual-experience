# Preferred visual experiences provide intuitive descriptions of the functional properties of the cortical navigation network

Code base for paper titled "Preferred visual experiences provide intuitive descriptions of the functional properties of the cortical navigation network"

## System requirements

* **OS:** Linux (tested on Ubuntu). Other platforms have not been tested.
* **Python:** 3.9
* **Main dependencies (tested versions):** torch 1.13.1 (CUDA 11.7), torchvision 0.14.1, pytorch-lightning, hydra-core, himalaya, pycortex, opencv-python 4.5.3.56, pandas 1.3.2, Pillow 9.5.0. See `requirements.txt` for the full list.
* **Hardware:** An NVIDIA GPU with CUDA support is required for training the VAE and for fitting the voxelwise encoding models (`himalaya` with the `torch_cuda` backend). The VAE was trained on 3 GPUs (see `gpus` in `configs/vae-2sec.yaml`). Feature extraction and generating preferred visual experiences can run on a single GPU (or on the CPU, more slowly).

## Installation

* Set up a new Conda environment
```
conda create -n drivingvae python=3.9
conda activate drivingvae
```

* Install pytorch (tested on torch=1.13.1, CUDA==11.7, Linux)
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

* Install dependency requirements.
```
pip install -r requirements.txt
pip install himalaya pycortex
```

**Typical install time:** about 10–20 minutes on a normal desktop computer, most of it spent downloading PyTorch.

## Instructions for use

The pipeline has four steps. Each step's outputs are the next step's inputs.

### 1. Train model
Set `data_configs.data_dir` in `configs/vae-2sec.yaml` to your directory of driving video frames, then run:

```commandline
python train_vae.py --config-name=vae-2sec
```

The model logs and checkpoints will be saved under `outputs`.

**Expected run time:** about 5 weeks (~820 hours of wall-clock time, including one resume from checkpoint) on 3 GPUs.

### 2. Extract features

```commandline
python extract_feature.py --output_path MODEL_PATH --save_path FT_SAVE_PATH
```

**Expected output:** VAE latent features for each stimulus frame, saved under `FT_SAVE_PATH` (including `center.npy`, which the decoder uses in step 4).

**Expected run time:** under 1 hour on a single GPU.

### 3. Run voxelwise encoding models
Set the subject IDs, run IDs and pycortex transforms in `utils/configs.py`, then run:

```commandline
python encoding_analysis.py --save_path VEM_RESULTS_PATH --feature_path FT_SAVE_PATH --bold_path BOLD_PATH
```

**Expected output:** per-subject regression weights (`coef.npy`), held-out prediction scores (`r2scores.npy`) and selected regularization parameters (`best_alphas.npy`), saved under `VEM_RESULTS_PATH/<subject>`.

**Expected run time:** about half a day on a single GPU to process all participants.

### 4. Generate preferred visual experiences

Open `preferred_visual_experience.ipynb` and set `MODEL_PATH`, `FT_SAVE_PATH`, `VEM_RESULTS_PATH` and `NAVIGATION_NETWORK_FILE`. The notebook averages the encoding weights within each cluster of the navigation network and decodes them into video clips with the VAE decoder.

**Expected output:** for each cluster, a cortical flatmap of the cluster next to the decoded frames of its preferred visual experience.

**Expected run time:** a few minutes.

## Reproduction

Running steps 1–4 in order on the data used in the paper reproduces the encoding model performance and the preferred visual experiences reported in the manuscript.

## License

This project is licensed under the BSD 3-Clause License. See `LICENSE`.
