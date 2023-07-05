# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos



## Dev plan
Priority
- [x] Without preprocessing. Get the hps and visualization code working.
- [x] Get refinement with 2D keypoints working.
- Get ground plane constraint working.
- Add preprocessing: tracking.
- Add interactive correction.

Later
- audio
- Colab, Dockerfile, gradio
- Double check the license

## Installation
Dependencies
- ffmpeg

```bash
./install.sh
```

## Running HARMONI on a sample video
```bash
python main.py --images data/demo/vid --hps dapa --out_folder ./results --render --use_cached_dataset --top_view

Optional args:
[--overlay_dyad_stats]
[--ground_constraint]
[--top_view]
[--refine_with_kp]
[--tracker]

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
ControlNet for openpose and midas api.
