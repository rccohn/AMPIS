module load gcc/6.3.0
module load cuda/10.0

python3.6 -m venv ampis_env

source ampis_env/bin/activate

pip install torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install cython pyyaml==5.1

pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install scikit-image numy matplotlib opencv-python jupyterlab scikit-learn

pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html
