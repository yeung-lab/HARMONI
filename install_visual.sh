pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install "setuptools<70"
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install kornia

pip install git+https://github.com/nghorbani/configer
pip install git+https://github.com/cocodataset/panopticapi.git
pip install matplotlib opencv-python scikit-image
pip install loguru termcolor Pillow joblib tqdm configargparse
pip install smplx==0.1.28 trimesh==3.9.13 pyrender
pip install open3d einops timm
pip install PyOpenGL PyOpenGL-accelerate

# install dependencies for phalp
pip install gdown
pip install cython scikit-learn scipy
pip install rich dill colordict "scenedetect[opencv]"
pip install hydra-core hydra-colorlog
pip install pyransac3d pytube
pip install --no-build-isolation chumpy
# Patch chumpy for numpy 1.24+ (np.bool, np.int etc. were removed)
CHUMPY_INIT=$(python -c "import site; print(site.getsitepackages()[0])")/chumpy/__init__.py
sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/import numpy; bool, int, float, complex, object, str = numpy.bool_, numpy.int_, numpy.float64, numpy.complex128, numpy.object_, numpy.str_; unicode = str; from numpy import nan, inf/' "$CHUMPY_INIT"

# install dependencies for downstream
pip install scikit-spatial

# Re-pin numpy<2 at the end (later packages may have pulled numpy 2.x,
# which is incompatible with torch 2.1.2)
pip install "numpy<2"
