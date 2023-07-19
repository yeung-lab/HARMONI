conda activate harmoni
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/nghorbani/configer
pip install matplotlib opencv-python scikit-image
pip install loguru termcolor Pillow joblib tqdm configargparse
pip install smplx==0.1.28 trimesh==3.9.13 pyrender

pip install detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install open3d einops timm

# install dependencies for phalp
pip install gdown
pip install cython scikit-learn==0.22 scipy==1.9.0
pip install rich dill colordict scenedetect[opencv]
pip install hydra-core hydra-colorlog

# install dependencies for downstream
pip install scikit-spatial