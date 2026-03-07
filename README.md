# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos

## Repository Overview
- [System requirements and installation Guide](#installation)
- [Download dependency data](#installation)
- [Demo on a video clip](#running-harmoni-visual-mdoel-on-a-demo-video)
- [Demo on a audio clip](#running-harmoni-audio-model-on-example-data)
- [Code structure](#code-structure)
- [Related resources](#related-resources)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Contact](#contact)


## Installation

### Option A: Conda
Tested on Linux with NVIDIA GPUs. Requires CUDA 12.x.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then create the visual environment:
```bash
conda create -n harmoni_visual python=3.10
conda activate harmoni_visual
./install_visual.sh
```

2. For the audio pipeline:
```
cd audio
conda env create -f process_audio.yml
conda activate process_audio
```

Installation for either visual or audio model should be around 5 to 10 minutes.

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and version pinning notes.

### Option B: Docker
See [Docker](#docker) section below.

2. Download data folder that includes model checkpoints and other dependencies [here](https://drive.google.com/drive/u/2/folders/1vMZl8CTf1-LUv6x1J_yHpYWU-IhPLQQL).
Note (Sept26,2025): Due to licensing requirements we cannot provide SMPL models. Please download from their websites instead [SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html), [SMPL/SMPLX](https://www.google.com/search?client=safari&rls=en&q=smpl+body+model&ie=UTF-8&oe=UTF-8).
Note: to generate the `SMPLA_{gender}.pth` files follow the instructions [here](https://github.com/Arthur151/ROMP/blob/a8558aed480af850756f84e2a7c787e359bddbd0/docs/installation.md#3-preparing-smpl-model-files).

The expected `data/` folder structure:
```
data/
├── arial.ttf
├── body_models/
│   ├── gmm_08.pkl
│   ├── smil_packed_info.pth
│   ├── smpl_mean_params.npz
│   ├── SMPLA_{FEMALE,MALE,NEUTRAL}.pth
│   ├── smil/
│   │   └── SMPL_{FEMALE,MALE,NEUTRAL}.pkl
│   ├── smpl/
│   │   └── SMPL_{FEMALE,MALE,NEUTRAL}.pkl
│   └── smplx/
│       ├── SMPL_NEUTRAL.pkl
│       └── SMPLX_{FEMALE,MALE,NEUTRAL}.{pkl,npz}
├── cfgs/
│   └── harmoni.yaml
├── ckpts/
│   ├── body_pose_model.pth
│   ├── body_type_classifier_res101.pth.tar
│   ├── dapa_adult.pt
│   ├── dapa_infant.pt
│   ├── dpt_large-midas-2f21e586.pt
│   ├── hand_pose_model.pth
│   ├── hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt
│   └── panoptic_deeplab_...coco_dsconv.pth
├── demo/
│   └── giphy.gif
└── _DATA/                # PHALP tracker files (downloaded automatically on first run)
    ├── hmar_v2_weights.pth
    ├── hmmr_v2_weights.pt
    ├── J_regressor_h36m.npy
    ├── models/smpl/SMPL_NEUTRAL.pkl
    ├── smpl_mean_params.npz
    ├── SMPL_to_J19.pkl
    └── texture.npz
```

4. We provide the example output from public video clips. You could download them [here](https://drive.google.com/drive/u/2/folders/13B6j3Px0nfxt_CCMqGksEAGm4f_dRHGo).

Visualization of the example clip.
<p float="center">
  <img src="teasers/video_repeated.gif" width="50%" />
</p>
Please see below for instructions for reproducing the visual results.

## Running HARMONI visual model on a demo video
Here we show how to run HARMONI on a public video [clip](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzl4ZG10d3lhbGMxc2E1OTVrdHU1emo0YXYwcGtsbDV1NG5uaDdqdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5pK2Rs57ZCACAh8Fxs/giphy.gif). A basic command would be
```bash
python main.py --config data/cfgs/harmoni.yaml --video data/demo/giphy.gif --out_folder ./results/giphy --keep contains_only_both --save_gif
```
To reproduce the provided results, please use the below two commands instead.
### Some configurations that significantly improves the reconstruction quality:
1. Using the detected ground plane as additional constraint (`--ground_constraint`). 
As in [Ugrinovic et al.](https://github.com/nicolasugrinovic/size_depth_disambiguation/tree/d4787668131298de5bc47efaea9aad4f15f3f93d), ground normal is estimated by fitting a plane to the floor depth points, and then `--ground_anchor` ("child_bottom" | "adult_bottom") speficies whether we use the mean ankle positions of children or adults as the anchor point for the ground plane. Then, we run optimization on all humans and encourage their ankles to be on the ground plane.

2. Overwrite the classified tracks.
It is hard to have a model to accurately predict whether a detected human is adult or child, so we allow the user to overwrite the predicted body types. For example, we can first run the command with `--dryrun` to run the body type classifition for each track. The results are written to `./results/giphy/sampled_tracks`.
```bash
python main.py --config data/cfgs/harmoni.yaml --video data/demo/giphy.gif --out_folder ./results/giphy --dryrun
```
Then, we can run it again with the tracks we want to overwrite. e.g. `--track_overwrite "{2: 'infant', 11: 'infant'}"`.
```bash
python main.py --config data/cfgs/harmoni.yaml --video data/demo/giphy.gif --out_folder ./results/giphy --keep contains_only_both --ground_anchor child_bottom --save_gif --track_overwrite "{2: 'infant', 11: 'infant'}"
```
If turn on the `--add_downstream` flag, the downstream stats will be overlayed to the results. E.g. 

<p float="center">
  <img src="teasers/video_with_labels_repeated.gif" width="50%" />
</p>

### Run time
For this 60 frame video clip, the typical run time on a single NVIDIA TITAN RTX GPU is 20 seconds for the body pose estimation (excluding data preprocessing and rendering). 
Data preprocessing (i.e. runnign OpenPose, ground normal estimation, etc) took 2 minutes. Rendering took 1 sec/frame.

## Running HARMONI audio model on example data
Before you run this, make sure to follow the additional installation instructions in `audio/README.md` and rebuild the x-vector extractor file.

Here, we show the result on a publicly available demo [video](https://bergelsonlab.com/seedlings/). Please download and put it in `data/demo/seedlings.mp4`.
```
cd audio
python run.py ../data/demo/seedlings.mp4 ../results/seedlings/
```

## Docker
The Docker image contains two separate Python environments: a visual venv (Python 3.10) and an audio venv (Python 3.7). The `data/` folder is mounted at runtime to avoid bloating the image. The code auto-detects GPU/CPU, so the same image works on both GPU and CPU-only machines (CPU will be significantly slower).

### Pull the image
```bash
docker pull lmbravo/harmoni:latest
docker tag lmbravo/harmoni:latest harmoni
```

### Run — Visual model
With GPU (`--gpus all` requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
docker run --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  harmoni python main.py --config data/cfgs/harmoni.yaml \
    --video data/demo/giphy.gif \
    --out_folder /app/results/giphy \
    --keep all --save_gif
```

Without GPU (omit `--gpus all`):
```bash
docker run --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  harmoni python main.py --config data/cfgs/harmoni.yaml \
    --video data/demo/giphy.gif \
    --out_folder /app/results/giphy \
    --keep all --save_gif
```

### Run — Audio model
Use `/opt/venv_audio/bin/python` to run the audio pipeline (Python 3.7):
```bash
docker run --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  harmoni /opt/venv_audio/bin/python audio/run.py \
    data/demo/seedlings.mp4 /app/results/seedlings/
```

## Code structure
```bash
- preprocess # code for preprocessing: downsample, shot detection, ground plane estimation
- trackers # tracking
- detectors  # e.g. openpose, midas, body type classifier
- hps # human pose and shape models. e.g. DAPA
- postprocess  # code for refinement. e.g. SMPLify, One Euro Filter.
- visualization  # renderers and helpers for visualization
- downstream # code for downstream analysis
- audio # audio code
- data  # see Installation for full structure
```

Output folder structure
```bash
- openpose
- sampled_tracks
- render
- results.pkl
- dataset.pkl
- result.mp4  # if --save_video is on
- result.gif  # if --save_gif is on
```

## Related Resources
We borrowed code from the below amazing resources:
- [PARE](https://github.com/mkocabas/PARE) for HMR-related helpers.
- [PHALP](https://github.com/brjathu/PHALP) for tracking.
- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation.
- [Panoptic DeepLab](https://github.com/bowenc0221/panoptic-deeplab) for segmentation.
- [size_depth_disambiguation](https://github.com/nicolasugrinovic/size_depth_disambiguation) estimating ground normal.
- [OpenPose](https://github.com/Hzzone/pytorch-openpose) for 2D keypoint estimation.


## Contact
[Zhenzhen Weng](https://zzweng.github.io/) (zzweng AT stanford DOT edu)

## Citation
If you find this work useful, please consider citing:
```
@article{
weng2025harmoni,
author = {Zhenzhen Weng  and Laura Bravo-S\'anchez  and Zeyu Wang  and Christopher Howard  and Maria Xenochristou  and Nicole Meister  and Angjoo Kanazawa  and Arnold Milstein  and Elika Bergelson  and Kathryn L. Humphreys  and Lee M. Sanders  and Serena Yeung-Levy },
title = {Artificial intelligence–powered 3D analysis of video-based caregiver-child interactions},
journal = {Science Advances},
volume = {11},
number = {8},
pages = {eadp4422},
year = {2025},
doi = {10.1126/sciadv.adp4422},
URL = {https://www.science.org/doi/abs/10.1126/sciadv.adp4422}}
```
