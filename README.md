# AMPIS
Additive Manufacturing Powder Instance Segmentation (AMPIS) utilizing Mask R-CNN.

# Installation
Requires python3.6. List of required packages can be found in **requirements.txt**. gcc is required to build pytorch. Code was found to work with gcc versions 5.3.0 and 7.4.0, but some of the more recent versions can cause problems with the pytorch installation. Cuda 10.1 required for pytorch with gpu acceleration. To automatically build and configure the environment, verify that 'python3' points to python3.6:
```bash
which python3
>>>/usr/bin/python3.6
```
and then run:
```bash
./build_environment.sh
```
This will install all required packages in a virtual environment called **ampis_venv**. To test that the environment was installed correctly, activate the environment and run the test script:
```bash
source ampis_venv/bin/activate
python test_environment.py
```
