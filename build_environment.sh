module load gcc/6.3.0
module load cuda/10.0

python3.6 -m venv ampis_env

source ampis_env/bin/activate

CUDA='cpu' # tested with 'cu100' for cuda 10.0 and 'cpu' for cpu-only install

pip install torch=="1.4+${CUDA}" torchvision=="0.5+${CUDA}" -f https://download.pytorch.org/whl/torch_stable.html

pip install cython pyyaml==5.1

pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install scikit-image numpy matplotlib opencv-python jupyterlab scikit-learn

pip install detectron2 -f "https://dl.fbaipublicfiles.com/detectron2/wheels/${CUDA}/index.html"
