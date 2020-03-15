<<<<<<< HEAD
=======
#!/usr/bin/env python
# coding: utf-8

# In[1]:


>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
##### Module imports

gui = False
if not gui:
    # make sure script doesn't break on non-gui jobs 
    # (ie batch job on computing cluster)
    import matplotlib 
    matplotlib.use('agg')

# regular module imports
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import re
import skimage.io

## detectron2
from detectron2 import model_zoo
from detectron2.checkpoint import Checkpointer, PeriodicCheckpointer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
<<<<<<< HEAD
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
=======
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
<<<<<<< HEAD
from detectron2.utils.visualizer import Visualizer

import dataval # custom module for getting loss stats # TODO figure out how to get more meaningful loss statistics
=======
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.modeling import build_model
import detectron2.utils.comm as comm
import dataval # custom module for getting loss stats
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
 
# verify cuda is installed and running correctly
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.cuda.is_available(), CUDA_HOME)


<<<<<<< HEAD
=======
# In[2]:


>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
##### Load data in a way that is consistent with detectron2 formatting

def get_data_dicts(json_path):
    """
    Loads data in format consistent with detectron2.
    Adapted from balloon example here:
    https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    
    Inputs: 
      json_path: string or pathlib path to json file containing relevant annotations
    
    Outputs:
      dataset_dicts: list(dic) of datasets compatible for detectron 2
                     More information can be found at:
                     https://detectron2.readthedocs.io/tutorials/datasets.html#
    """
    json_path = os.path.join(json_path) # needed for path manipulations
    with open(json_path) as f:
        via_data = json.load(f)
        
    # root directory of images is given by relative path in json file
    img_root = os.path.join(os.path.dirname(json_path), via_data['_via_settings']['core']['default_filepath'])
    imgs_anns = via_data['_via_img_metadata']
    
    
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_root, v["filename"])
        
        # inefficient for large sets of images, read from json?
        height, width = skimage.io.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["dataset_class"] = v['file_attributes']['Image Class']
        
        annos = v["regions"]
        objs = []
        for anno in annos:
            # not sure why this was here, commenting it out didn't seem to break anything
            #assert not anno["region_attributes"] 
            anno = anno["shape_attributes"]
            
            # polygon masks is list of polygon coordinates in format ([x0,y0,x1,y1...xn,yn]) as specified in
            # https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.PolygonMasks
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
                     
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS, # boxes are given in absolute coordinates (ie not corner+width+height)
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

<<<<<<< HEAD

# In my data I have the training/validation data in a single VIA json file.
# This function splits the data into distinct groups based on a specific attribute (specified by get_subset function)
=======
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
def split_data_dict(dataset_dicts, get_subset=None):
    """
    Splits data from json into subsets (ie training/validation/testing)
    
    inputs 
      dataset_dicts- list(dic) from get_data_dicts()
      get_subset- function that identifies 
                  class of each item  in dataset_dict.
                  For example, get_subset(dataset_dicts[0])
                  returns 'Training', 'Validation', 'Test', etc
                  If None, default function is used
    
    returns
      subs- dictionary where each key is the class of data
            determined from get_subset, and value is a list
            of dicts (same format of output of get_data_dicts())
            with data of that class
    """
    
    if get_subset is None:
        get_subset = lambda x: x['dataset_class']
    
    
    subsets = np.unique([get_subset(x) for x in dataset_dicts])

    datasets = dict(zip(subsets, [[] for _ in subsets]))
    
    for d in dataset_dicts:
        datasets[get_subset(d)].append(d)
    
    return datasets

json_path = '../data/raw/via_2.0.8/via_powder_particle_masks.json'
ddicts = get_data_dicts(json_path)

subs = split_data_dict(ddicts)

<<<<<<< HEAD

# Store the data so that detectron2 can work with it
=======
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
for key, value in subs.items():
    DatasetCatalog.register("powder_" + key, lambda key=key: subs.get(key))
    MetadataCatalog.get("powder_" + key).set(thing_classes=["Powder"])


<<<<<<< HEAD

##### Verify ground-truth masks are loaded correctly
# overlays ground truth instances on images and saves them. 
=======
# In[5]:


##### Verify ground-truth masks are loaded correctly

>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
for dataset in ['powder_Training', 'powder_Validation']:
    for d in DatasetCatalog.get(dataset):
        img_path = pathlib.Path(d['file_name'])
        img = cv2.imread(str(img_path))
        visualizer = Visualizer(img, metadata=MetadataCatalog.get('powder_Training'), scale=1)
        vis = visualizer.draw_dataset_dict(d)
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        ax.imshow(vis.get_image())
        ax.axis('off')
        ax.set_title('{}\n{}'.format(dataset, img_path.name))
        fig.tight_layout()
        fig.savefig('../figures/masks/ground_truth/{}_{}.png'.format(dataset, img_path.stem))
        if matplotlib.get_backend() is not 'agg': # if gui session is used, show images
            plt.show()
        plt.close('all')


<<<<<<< HEAD

##### Set up detectron2 configurations for Mask R-CNN model

cfg = get_cfg()  # initialize configuration
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask_rcnn with ResNet 50 backbone and feature pyramid network (FPN)

cfg.DATASETS.TRAIN = ("powder_Training",)  # specifies name of training dataset (must be registered)
cfg.DATASETS.TEST = ("powder_Validation",)  # specifies name of test dataset (must be registered)
cfg.DATALOADER.NUM_WORKERS = 2

## If the weights are locally available, use those. Otherwise, download them from model zoo.
## Needed when computer does not have network access/permission to download files.

weights_path = pathlib.Path('../','models','model_final_f10217.pkl') # path where downloaded weights is stored
if weights_path.is_file():
=======
# In[10]:


##### Set up configurations

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("powder_Training",)
cfg.DATASETS.TEST = ("powder_Validation",)
cfg.DATALOADER.NUM_WORKERS = 2
if pathlib.Path('../','models','model_final_f10217.pkl').is_file():
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
    print('Weights found')
    cfg.MODEL.WEIGHTS = '../models/model_final_f10217.pkl'
else:
    weights_source = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
    print('Weights not found, downloading from source: {}'.format(weights_source))
    cfg.MODEL.WEIGHTS = weights_source
    
cfg.SOLVER.IMS_PER_BATCH = 1
<<<<<<< HEAD
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate (default is 0.00025)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # default: 512

#  Number of different classes of instances that will be predicted.
=======
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 20000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (powder particle)

# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
<<<<<<< HEAD
cfg.TEST.DETECTIONS_PER_IMAGE = 300  ## UPPER LIMIT OF NUMBER INSTANCES THAT WILL BE RETURNED, ADJUST ACCORDINGLY

# Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
# Adjust this to determine how long the model trains.
cfg.SOLVER.MAX_ITER = 20000   
=======
cfg.TEST.DETECTIONS_PER_IMAGE = 300

# In[ ]:
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921


##### Train with model checkpointing

<<<<<<< HEAD
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # weights and tensorboard metrics will be stored here
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False) 

train = False # make True to retrain, False to skip training (ie when you only want to evaluate)
=======
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
train = False # make True to retrain
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
if train:
    print('training model')
    trainer.train() # uncomment to retrain model
else:
    print('skipping training, using results from previous training')


<<<<<<< HEAD
# get path of model weight at each checkpoint, 
# sorted in order of the number of training steps, followed by the last model
checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

=======

# In[70]:


##### 

# get model checkpoints, sorted in order of the number of training steps, followed by the last model
checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

# # temp directory
#checkpoint_paths = sorted(list(pathlib.Path('/media/ryan/TOSHIBA EXT/Research/Projects/AMPIS/output').glob('*model_*.pth')))

>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
## TODO get losses at each checkpoint
print('checkpoint paths found:\n\t{}'.format('\n\t'.join([x.name for x in checkpoint_paths])))


<<<<<<< HEAD
# Compute validation losses at each checkpoint
# TODO finish this, it does not appear to work yet

# create a separate directory for validation loss outputs (json and tensorboard)
val_dir = os.path.join(str(cfg.OUTPUT_DIR)+'_validation')
os.makedirs(val_dir, exist_ok=True)

# set up writers
writers = ([
 CommonMetricPrinter(cfg.SOLVER.MAX_ITER), # prints to terminal
            JSONWriter(os.path.join(val_dir, "metrics_validation.json")), # each line in this file is a separate JSON containing losses
            TensorboardXWriter(val_dir), # creates tensorboard log file with all constants
])

# set up evaluator for validation set
data_val = dataval.build_detection_val_loader(cfg, None, ['powder_Validation'])

# repeat for every checkpoint saved during training
for p in checkpoint_paths:

=======
# In[71]:


## visualize predictions of final model

val_dir = os.path.join(str(cfg.OUTPUT_DIR)+'_validation')
os.makedirs(val_dir, exist_ok=True)
writers = ([
 CommonMetricPrinter(cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(val_dir, "metrics_validation.json")),
            TensorboardXWriter(val_dir),
])

data_val = dataval.build_detection_val_loader(cfg, None, ['powder_Validation'])

for p in checkpoint_paths:
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
    cfg.MODEL.WEIGHTS = os.path.join(p)
    model = build_model(cfg)
    
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
    model, val_dir, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1
    )

    with EventStorage(start_iter) as storage:
<<<<<<< HEAD
        for data in data_val: 
              
            storage.step()
            loss_dict = model(data) # compute validation losses
=======
        for data in data_val:
              
            storage.step()
            loss_dict = model(data)
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
            losses = sum(loss_dict.values())

            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            
<<<<<<< HEAD
            for writer in writers: # write results to terminal, json, and tensorboard files
                writer.write()


        ### visualization of predicted masks on all images
=======
            for writer in writers:
                writer.write()


        ### visualization of masks
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
        # make directory for output mask predictions
        outdir = '../figures/masks/predictions/{}'.format(p.stem)
        os.makedirs(outdir, exist_ok=True)
        predictor = DefaultPredictor(cfg)
        for dataset in ['powder_Training', 'powder_Validation']:
<<<<<<< HEAD
            for d in DatasetCatalog.get(dataset): # TODO  replace with datasetloader to make this less hacky
=======
            for d in DatasetCatalog.get(dataset): # TODO  replace with datasetloader
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
                img_path = pathlib.Path(d['file_name'])
                print('image filename: {}'.format(img_path.name))
                img = cv2.imread(str(img_path))
            
<<<<<<< HEAD
            # overlay predicted masks on image
=======
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
            out = predictor(img)
            v = Visualizer(img, metadata=MetadataCatalog.get(dataset))
            draw = v.draw_instance_predictions(out['instances'].to('cpu'))
            
<<<<<<< HEAD
            # save images with standard formatting
=======
>>>>>>> b4c32bc862a2b2b282635eb29ef6a4915a8a4921
            fig, ax = plt.subplots(figsize=(10,5), dpi=300)
            ax.imshow(draw.get_image())
            ax.set_title(title)
            fig.tight_layout()
            print(outdir, title, '.png')
            
            fig.savefig(pathlib.Path(outdir, title.replace('\n','_')))
            plt.close('all')


