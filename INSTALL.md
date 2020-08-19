# Installation
AMPIS depends on several packages, including PyTorch and Detectron2. The PyTorch installation depends on the available CUDA environment. Detectron2 depends on both the PyTorch installation and available CUDA environment. Therefore, the process is as follows. First, clone the repository, create a virtual environment, and install the standard python packages used by AMPIS. Then, install PyTorch according to your system. Next, install the corresponding Detectron2 build. Finally, install AMPIS. 

## 1) Clone git repository
```bash
git clone https://github.com/rccohn/AMPIS.git
cd AMPIS
```
## 2) Set up and activate virtual environment
To create and activate an environment called **ampis_env:**
```bash
python3 -m venv ampis_env
source ampis_env/bin/activate
```
## 3) Install requirements file
```bash
pip install -r requirements.txt
```
Note- you may see some error messages pop up, but this is ok- pip should resolve the conflicts.

## 4) Install COCO Python API
Note ```pip install pycocotools``` does not work in some cases, I found this method to be better:
```bash 
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

## 5) Install PyTorch and TorchVision
The version of torch and torchvision will depend on your system and cuda installation. For more information, see https://pytorch.org/.

For example, for installing pytorch 1.5 and the compatible version of torchvision for CUDA 10.1:
```bash
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 6) Install Detectron2
The version of detectron2 will depend on the versions of pytorch and CUDA installed on your system.
There are different methods of installation. For Linux, the most straightforward method is to install a pre-built wheel. For other operating systems, detectron2 can be built from source, see their documentation for more info.

For pytorch 1.5 + CUDA 10.1:

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

More detailed information and additional builds can be found on the [detectron2 installation page](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

## 7) Install AMPIS
The recommended method for installing AMPIS is with pip's 'editable' mode.
After navigating to ` AMPIS/`:

```bash
pip install -e .
```

## 8) Verify installation
To verify everything is correctly installed, open a python terminal and enter the following:

```python
import pycocotools
import torch
import detectron2
import ampis
```

If the modules import without errors, you are good to go!
