# create and activate virtual environment
python3.6 -m venv ampis_venv
source ampis_venv/bin/activate

# install dependencies
pip install -r requirements.txt

# build pycocotools from source
cd src/external/cocoapi/PythonAPI/
make
pip install -e .

# build detectron2 from source
cd ../../detectron2
pip install -e .
