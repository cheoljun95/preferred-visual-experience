# Preferred visual experiences provide intuitive descriptions of the functional properties of the cortical navigation network

Code base for paper titled "Preferred visual experiences provide intuitive descriptions of the functional properties of the cortical navigation network"

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

* Install dependency requirments.
```
pip install -r requirements.txt
```

## Train model
* Run the following command.

```commandline
python train.py --config-name=vae-2sec
```

The model logs and checkpoints will be saved under `outputs`

## Extract features

```commandline
python extract_feature.py --output_path MODEL_PATH --save_path FT_SAVE_PATH
```

## Run voxelwise encoding models

```commandline
python encoding_analysis.py --save_path VEM_RESULTS_PATH --feature_path FT_SAVE_PATH --bold_path BOLD_PATH
```

## Generate preferred visual experiences

Check `preferred_visual_experience.ipynb`