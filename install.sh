conda create -n harmoni python==3.9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/nghorbani/configer
pip install matplotlib opencv-python scikit-image
pip install loguru termcolor Pillow joblib tqdm configargparse
pip install smplx==0.1.28 trimesh==3.9.13 pyrender