# HARMONI: Using 3D Computer Vision and Audio Analysis to Quantify Caregiver–Child Behavior and Interaction from Videos



## Dev plan
Priority
- Without preprocessing. Get the hps and visualization code working.
- Get refinement with 2D keypoints working.
- Get ground plane constraint working.
- Add preprocessing.

Later
- audio
- Colab, Dockerfile, gradio
- Double check the license

## Installation
```bash
conda create -n harmoni python==3.9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/nghorbani/configer
pip install matplotlib opencv-python scikit-image
pip install loguru termcolor Pillow joblib tqdm configargparse
pip install smplx==0.1.28 trimesh==3.9.13 pyrender
# other dependencies
ffmpeg

```


## Running HARMONI on a sample video
```bash
python main.py --images data/demo/vid --hps dapa --out_folder ./results --render --use_cached_dataset

Optional args:
[--overlay_dyad_stats]
[--ground_constraint]
[--side_view]
[--top_view]
[--refine_with_kp]
[--tracker]

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
- detections
    - keypoints
    - depth
- rendered
- tracking_info.pkl
- results.pkl
- dataset.pkl
- result.mp4
```

## Related Resources
We borrowed code from the below amazing resources:
ControlNet for openpose and midas api.
