##### Module imports

gui = False
if not gui:
    # make sure script doesn't break on non-gui jobs 
    # (ie batch job on computing cluster)
    import matplotlib 
    matplotlib.use('agg')

# regular module imports
import cv2
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle
import re
import skimage
import skimage.io
import skimage.measure

import mrcnn_utils

## detectron2
from detectron2 import model_zoo
from detectron2.checkpoint import Checkpointer, PeriodicCheckpointer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.detection_utils import annotations_to_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.visualizer import Visualizer

#import dataval # custom module for getting loss stats # TODO figure out how to get more meaningful loss statistics
 
# verify cuda is installed and running correctly
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.cuda.is_available(), CUDA_HOME)


# use subset of images without excessive amounts of instances for this experiment
# filenames were randomly shuffled before saving
#with open('spheroidite-files.pickle', 'rb') as f:
with open('../data/raw/spheroidite/spheroidite-files.pickle', 'rb') as f:
    filename_subset = sorted(pickle.load(f))


EXPERIMENT_NAME = 'spheroidite'    
#data_root = pathlib.Path('/media/ryan/TOSHIBA EXT/Research/datasets/uhcs-segment/images/spheroidite/')
data_root = pathlib.Path('..','data','raw','spheroidite')
image_paths = {x.stem.replace('micrograph-','') : x for x in data_root.glob('micrograph*')}
annotation_paths = {x.stem.replace('annotation-','') : x for x in data_root.glob('annotation*')}

image_subset = [image_paths.get(x) for x in filename_subset if image_paths.get(x) is not None]
annotation_subset = [annotation_paths.get(x) for x in filename_subset if annotation_paths.get(x) is not None]

img_train = image_subset[:-2]
img_valid = image_subset[-2:]

ann_train = annotation_subset[:-2]
ann_valid = annotation_subset[-2:]


train_set = [img_train, ann_train, 'Training']
valid_set = [img_valid, ann_valid, 'Validation']
datasets_all = {'Training': train_set, 
       'Validation': valid_set}

    
##### Load data in a way that is consistent with detectron2 formatting

def get_ddicts(img_paths, label_paths, dclass):
    """
    img_paths: list of  path to original images
    label_paths: list of paths to annotation image (same order as img_paths )
    dclass:  'Training,' 'Validation', 'Testing', etc 
    """ 
    
    ddicts = []
    for ipath, lpath, d in zip(img_paths,label_paths, itertools.repeat(dclass)):

        im = skimage.io.imread(lpath)

        r, c = im.shape

        resized_im_path = pathlib.Path(ipath.parent, 'resized_'+ipath.name)
        if not resized_im_path.is_file():
            im_resize = skimage.io.imread(ipath, as_gray=True)[:r,:c]
            skimage.io.imsave(resized_im_path, im_resize)


	

        ddict = {'file_name': str(resized_im_path),
                 'annotation_file': str(lpath),
                 'height': r,
                 'width': c,
                 'dataset_class':d,
                 'mask_format': "bitmask"}
        
        ddicts.append(ddict)
    
    
    return ddicts

def mapper(ddict):
    """
    maps compressed ddict format to full format for training/inference
    input: ddict from list(dict) returned from get_ddicts() 
    returns: dataset_dict- data fully formatted for use with mask r-cnn
    """
    ext_mapper = {'.tiff':255,
                 '.png':0}

    im_path = pathlib.Path(ddict['file_name'])
    ann_path = pathlib.Path(ddict['annotation_file'])
    im_b = skimage.io.imread(ddict['annotation_file']) == ext_mapper[ann_path.suffix]


    img = skimage.io.imread(im_path, as_gray=True)
    img = skimage.color.gray2rgb(img)
    
    labels = skimage.measure.label(im_b)
    unique = np.unique(labels)[1:] # ignore 0 pixels 
    
    n = unique.shape[0]
    
    annotations=[]
    for i in range(1,n+1):
        mask = labels == i
        bbox = mrcnn_utils.extract_bboxes(mask[:,:,np.newaxis])[0]
        label = 0
        annotations.append({'bbox': bbox,
                            'bbox_mode': BoxMode.XYXY_ABS,
                            'segmentation': mask,
                            'category_id': label})
    
    instances = annotations_to_instances(annotations, img.shape[:2], mask_format='bitmask')
    
    dataset_dict = {}
    for k, v in ddict.items():
        dataset_dict[k] = v
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
    dataset_dict['annotations'] = annotations
    
    
    return dataset_dict



# Store the data so that detectron2 can work with it
dataset_names = []
for key, value in datasets_all.items():
    name = EXPERIMENT_NAME +'_'+key
    DatasetCatalog.register(name, lambda: get_ddicts(*value))
    MetadataCatalog.get(name).set(thing_classes=["Spheroidite"])
    dataset_names.append(name)

##### Verify ground-truth masks are loaded correctly
# overlays ground truth instances on images and saves them. 

for dataset in dataset_names:
    for d in DatasetCatalog.get(dataset):
        img_path = pathlib.Path(d['file_name'])
        img = cv2.imread(str(img_path))
        visualizer = Visualizer(img, metadata=MetadataCatalog.get(dataset), scale=1)
        vis = visualizer.draw_dataset_dict(mapper(d))
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        ax.imshow(vis.get_image())
        ax.axis('off')
        ax.set_title('{}\n{}'.format(dataset, img_path.name))
        fig.tight_layout()
        fig.savefig('../figures/masks/ground_truth/{}_{}.png'.format(dataset, img_path.stem))
        if matplotlib.get_backend() is not 'agg': # if gui session is used, show images
            plt.show()
        plt.close('all')



##### Set up detectron2 configurations for Mask R-CNN model

cfg = get_cfg()  # initialize configuration
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask_rcnn with ResNet 50 backbone and feature pyramid network (FPN)

cfg.DATASETS.TRAIN = ("spheroidite_Training",)  # specifies name of training dataset (must be registered)
cfg.DATASETS.TEST = ("spheroidite_Validation",)  # specifies name of test dataset (must be registered)
cfg.DATALOADER.NUM_WORKERS = 2

## If the weights are locally available, use those. Otherwise, download them from model zoo.
## Needed when computer does not have network access/permission to download files.

weights_path = pathlib.Path('../','models','model_final_f10217.pkl') # path where downloaded weights is stored
if weights_path.is_file():
    print('Weights found')
    cfg.MODEL.WEIGHTS = '../models/model_final_f10217.pkl'
else:
    weights_source = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo
    print('Weights not found, downloading from source: {}'.format(weights_source))
    cfg.MODEL.WEIGHTS = weights_source
    
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate (default is 0.00025)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000 
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # default: 512
cfg.TEST.EVAL_PERIOD=cfg.SOLVER.CHECKPOINT_PERIOD

#  Number of different classes of instances that will be predicted.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (powder particle)

# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
cfg.TEST.DETECTIONS_PER_IMAGE = 600  ## UPPER LIMIT OF NUMBER INSTANCES THAT WILL BE RETURNED, ADJUST ACCORDINGLY

# Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
# Adjust this to determine how long the model trains.
cfg.SOLVER.MAX_ITER = 20000   


##### Train with model checkpointing

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # weights and tensorboard metrics will be stored here
trainer = DefaultTrainer(cfg) 
#trainer.build_hooks()
trainer.resume_or_load(resume=False) 

train = True # make True to retrain, False to skip training (ie when you only want to evaluate)
if train:
    print('training model')
    trainer.train() # uncomment to retrain model
else:
    print('skipping training, using results from previous training')


# get path of model weight at each checkpoint, 
# sorted in order of the number of training steps, followed by the last model
checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

## TODO get losses at each checkpoint
print('checkpoint paths found:\n\t{}'.format('\n\t'.join([x.name for x in checkpoint_paths])))


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

    cfg.MODEL.WEIGHTS = os.path.join(p)
#     model = build_model(cfg)
    
#     optimizer = build_optimizer(cfg, model)
#     scheduler = build_lr_scheduler(cfg, optimizer)

#     checkpointer = DetectionCheckpointer(
#     model, val_dir, optimizer=optimizer, scheduler=scheduler
#     )
#     start_iter = (
#     checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1
#     )

#     with EventStorage(start_iter) as storage:
#         for data in data_val: 
              
#             storage.step()
#             loss_dict = model(data) # compute validation losses
#             losses = sum(loss_dict.values())

#             assert torch.isfinite(losses).all(), loss_dict

#             loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
#             losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#             if comm.is_main_process():
#                 storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            
#             for writer in writers: # write results to terminal, json, and tensorboard files
#                 writer.write()


        ### visualization of predicted masks on all images
        # make directory for output mask predictions
    outdir = '../figures/masks/predictions/{}'.format(p.stem)
    os.makedirs(outdir, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    for dataset in dataset_names:
        for d in DatasetCatalog.get(dataset): # TODO  replace with datasetloader to make this less hacky
            img_path = pathlib.Path(d['file_name'])
            print('image filename: {}'.format(img_path.name))
            img = cv2.imread(str(img_path))

        # overlay predicted masks on image
        out = predictor(img)
        v = Visualizer(img, metadata=MetadataCatalog.get(dataset))
        draw = v.draw_instance_predictions(out['instances'].to('cpu'))

        # save images with standard formatting
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        ax.imshow(draw.get_image())
        ax.set_title(title)
        fig.tight_layout()
        print(outdir, title, '.png')

        fig.savefig(pathlib.Path(outdir, title.replace('\n','_')))
        plt.close('all')


