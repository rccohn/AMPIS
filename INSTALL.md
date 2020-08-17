#Installation

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
## 1) Install requirements file
```bash
pip install -r requirements.txt
```

## 2) Install torch and torchvision
The version of torch and torchvision will depend on your system and cuda installation. For more information, see https://pytorch.org/.

For example, for installing pytorch 1.5 and the compatible version of torchvision for CUDA 10.1:
```bash
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 3) Install detectron2
The version of detectron2 will depend on the versions of pytorch and CUDA installed on your system.
There are different methods of installation. For Linux, the most straightforward method is to install a pre-built wheel. For other operating systems, detectron2 can be built from source, see their documentation for more info.

For pytorch 1.5 + CUDA 10.1:

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

More detailed information and additional builds can be found on the [detectron2 installation page](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). 

## 4) Install AMPIS
The recommended method for installing AMPIS is with pip's 'editable' mode.
After navigating to ` AMPIS/`:

```bash
pip install -e .
```