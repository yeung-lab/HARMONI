# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos

## Current results
<p float="center">
  <img src="teasers/video_repeated.gif" width="50%" />
</p>


## Installation
1. Install conda environment.
```bash
./install.sh
```
2. Download data.
```
gdown
unzip data.zip
rm data.zip
```
3. We provide the example output of one example public video clip. You could download it.
```
gdown
unzip results.zip
rm results.zip
```
Please see below for instructions for reproducing the results.

## Running HARMONI on a demo video
Here we show how to run HARMONI on a public video clip [clip](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYzl4ZG10d3lhbGMxc2E1OTVrdHU1emo0YXYwcGtsbDV1NG5uaDdqdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5pK2Rs57ZCACAh8Fxs/giphy.gif). 
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

## Code structure
```bash
- preprocess # code for preprocessing: downsample, shot detection, ground plane estimation
- trackers # tracking
- detectors  # e.g. openpose, midas
- hps # human pose and shape models. e.g. DAPA, CLIFF
- postprocess  # code for refinement code
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
```

Output folder structure
```bash
- openpose
- sampled_tracks
- render
- results.pkl
- dataset.pkl
- result.mp4  # if --save_video
- result.gif  # if --save_gif
```

## Related Resources
We borrowed code from the below amazing resources:
- [PARE](https://github.com/mkocabas/PARE) for HMR-related helpers.
- [MiDaS](https://github.com/isl-org/MiDaS) for depth estimation.
- [Panoptic DeepLab](https://github.com/bowenc0221/panoptic-deeplab) for segmentation.
- [size_depth_disambiguation](https://github.com/nicolasugrinovic/size_depth_disambiguation) estimating ground normal.
- [OpenPose](https://github.com/Hzzone/pytorch-openpose) for 2D keypoint estimation.


## Contact
[Zhenzhen Weng](https://zzweng.github.io/) (zzweng AT stanford DOT edu)

