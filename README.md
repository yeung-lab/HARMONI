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
Dependencies
- ffmpeg

Install conda environment.
```bash
./install.sh
```

## Running HARMONI on a folder of images
Default flags are in `data/cfgs/harmoni.yaml`.
```bash
python main.py --images data/demo/vid --out_folder ./results --render --use_cached_dataset --ground_constraint

Optional args:
[--ground_constraint]
[--top_view]

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
    - priors # vposer weights
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