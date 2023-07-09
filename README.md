# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos

## Current results
<p float="center">
  <img src="teasers/video_repeated.gif" width="50%" />
</p>


## Dev plan
Priority
- [x] Without preprocessing. Get the hps and visualization code working.
- [x] Get refinement with 2D keypoints working.
- [x] Get ground plane constraint working.
- Add downstream attributes extraction.
- [x] Add customizable filters. (e.g. Filter out frames without 1 child and 1 adult during visualization, etc.)
- [x] Add preprocessing: phalp tracking.
- Add preprocessing: shot detection.
- Allow for user correction.

Later
- audio
- Colab, Dockerfile, gradio
- Double check the license of dependencies and body models (e.g. SMPL)

## Installation
Note: if you are on SAIL. Just do this to set up the env and data.
```
ln -s /pasteur/u/zzweng/miniconda3/envs/harmoni path_to_your_cond_env
ln -s /pasteur/u/zzweng/projects/HARMONI/data data
```

1. Install conda environment.
```bash
./install.sh
```
2. Download data.
TODO.

## Running HARMONI on a demo video
Here we show how to run HARMONI on an example [clip](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzl4ZG10d3lhbGMxc2E1OTVrdHU1emo0YXYwcGtsbDV1NG5uaDdqdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5pK2Rs57ZCACAh8Fxs/giphy.gif). 
```bash
python main.py --config data/cfgs/harmoni.yaml --video data/demo/giphy.gif --out_folder ./results/giphy --keep contains_only_both --ground_anchor child_bottom --save_gif --save_mesh
```
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
python main.py --config data/cfgs/harmoni.yaml --video data/demo/giphy.gif --out_folder ./results/giphy --keep contains_only_both --ground_anchor child_bottom --save_gif --save_mesh --track_overwrite "{2: 'infant', 11: 'infant'}" 
```

## Compute downstream attributes
TODO

## Generate plots in the paper
TODO


## Code structure
```bash
- preprocess # code for preprocessing: downsample, shot detection, ground plane estimation
- trackers # tracking
- detectors  # e.g. openpose, midas
- hps # human pose and shape models. e.g. DAPA, CLIFF
- postprocess  # code for refinement code
- visualization  # renderers and helpers for visualization
- downstream # code for downstream analysis
- audio
- data
    - cfgs  # configurations
    - demo  # a short demo video
    - body_models
        - SMPL
        - SMIL
        - SMPLX
    - ckpts # model checkpoints
```

Output folder structure
```bash
- openpose
- sampled_tracks
- render
- results.pkl
- dataset.pkl
- result.mp4  # if --save_video
```

## Related Resources
We borrowed code from the below amazing resources:
- [PARE](https://github.com/mkocabas/PARE) for HMR-related helpers.
- MiDaS, Panoptic DeepLab, ControlNet, OpenPose, etc...