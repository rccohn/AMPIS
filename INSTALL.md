# Installation
Environment files are located in the docker directory of this project.
Installation of AMPIS takes 4 steps:

## Step 1: Create environment from conda yaml file
(yaml is located in docker/env.yml)
`conda env create --file env.yml`
Note: if you already have a conda env named "env", you can change the name
to anything else or override the default name by including `--name "my_env_name"`
in the above command.

## Step 2: activate environment
`conda activate env`

## Step 3: install AMPIS from source
`pip install -e git+https://github.com/rccohn/AMPIS@master#egg=ampis`

## Step 4: install detectron2
On linux, detectron2 binaries are avaiable:
```
pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

For other systems (or other versions,) please see the [Detectron2 docs](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
