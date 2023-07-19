# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos

## Repository Overview
- [System requirements and installation Guide](#installation)
- [Download dependency data](#installation)
- [Demo on a video clip](#running-harmoni-visual-mdoel-on-a-demo-video)
- [Demo on a audio clip](#running-harmoni-audio-model-on-example-data)
- [Code structure](#code-structure)
- [Related resources](#related-resources)
- [Contact](#contact)


## Installation
Tested on a linux machine with a single NVIDIA TITAN RTX GPU.

OS version is Ubuntu 20.04.4 LTS. CUDA version: 11.3. Python version 3.9.
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and then install the packages. 
```bash
conda create -n harmoni_visual python==3.9
conda activate harmoni_visual
./install_visual.sh
```
To install environment for the audio part, please do
```
cd audio
conda env create -f process_audio.yml
conda activate process_audio
```
Installation for either visual or audio model should be around 5 to 10 minutes.

2. Download data folder that includes model checkpoints and other dependencies [here](https://drive.google.com/drive/u/2/folders/1vMZl8CTf1-LUv6x1J_yHpYWU-IhPLQQL).

3. We provide the example output from public video clips. You could download them [here](https://drive.google.com/drive/u/2/folders/13B6j3Px0nfxt_CCMqGksEAGm4f_dRHGo).

Visualization of the example clip.
<p float="center">
  <img src="teasers/video_repeated.gif" width="50%" />
</p>
Please see below for instructions for reproducing the visual results.

## Running HARMONI visual mdoel on a demo video
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
- data
    - cfgs  # configurations
    - demo  # a short demo video
    - body_models
        - SMPL
        - SMIL
        - SMPLX
    - ckpts # model checkpoints
- _DATA # data for running PHALP. It should be downloaded automatically.
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
