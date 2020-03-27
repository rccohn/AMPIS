##### Module imports
gui = False
import matplotlib
if not gui:
    # make sure script doesn't break on non-gui jobs 
    # (ie batch job on computing cluster)
    # for non gui, this needs to be set before pyplot or other libraries that use pyplot (ie seaborn, detectron visualizer, etc)
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
import skimage
import skimage.io
import skimage.measure
from skimage.transform import resize as im_resize

import pycocotools.mask as RLE
import data_utils

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
from detectron2.structures import Boxes, BoxMode
import detectron2.utils.comm as comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.visualizer import Visualizer


# verify cuda is installed and running correctly
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(torch.cuda.is_available(), CUDA_HOME)


# use subset of images without excessive amounts of instances for this experiment
# filenames were randomly shuffled before saving
#with open('spheroidite-files.pickle', 'rb') as f:
debug_desktop = False # sets directories
if debug_desktop:
    #data_root = pathlib.Path('/media/ryan/TOSHIBA EXT/Research/datasets/uhcs-segment/images/spheroidite/')
    data_root = pathlib.Path('..','data','raw','spheroidite-images')
else:
    data_root = pathlib.Path('..', 'data', 'raw', 'spheroidite-images')

assert data_root.is_dir()
with open(pathlib.Path(data_root, 'spheroidite-files.pickle'), 'rb') as f:
    filename_subset = sorted(pickle.load(f))

EXPERIMENT_NAME = 'spheroidite'  # TODO set up json control file with all parameters


filename_selector = lambda pathname, pattern: pathname.stem.replace(pattern,'').split('_sizeRC')[0]
image_paths = {filename_selector(x, 'micrograph-'): x for x in data_root.glob('micrograph*')}
annotation_paths = {filename_selector(x, 'annotation-'): x for x in data_root.glob('annotation*')}

image_subset = [image_paths.get(x) for x in filename_subset if image_paths.get(x) is not None]
annotation_subset = [annotation_paths.get(x) for x in filename_subset if annotation_paths.get(x) is not None]


img_train = image_subset[:-2]
img_valid = image_subset[-2:]

ann_train = annotation_subset[:-2]
ann_valid = annotation_subset[-2:]

train_set = [img_train, ann_train, 'Training']
valid_set = [img_valid, ann_valid, 'Validation']

print('train_set: {}'.format([x.name for x in img_train]))
print('valid set: {}'.format([x.name for x in img_valid]))

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
    for idx, (ipath, lpath, d) in enumerate(zip(img_paths,label_paths, itertools.repeat(dclass))):

        get_size = lambda path: [int(x) for x in path.stem.split('sizeRC_')[-1].split('_')]
        imsize = get_size(ipath)
        assert imsize == get_size(lpath)

        ddict = {'file_name': str(ipath),
                 'annotation_file': str(lpath),
                 'height': imsize[0],
                 'width': imsize[1],
                 'dataset_class': d,
                 'mask_format': "bitmask",
                 'image_id': idx}

        ann = skimage.io.imread(str(lpath), as_gray=True)
        labels = skimage.measure.label(ann == 255)

        unique = np.unique(labels)
        unique = unique[unique > 0]

        annotations = []
        for i in range(1, unique.shape[0]+1):
            mask = labels == i
            bbox = data_utils.extract_boxes(mask)[0]
            mask = RLE.encode(np.asfortranarray(mask))

            annotations.append({
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': mask,
                'category_id': 0,
                })

        ddict['annotations'] = annotations
        ddict['num_instances'] = len(annotations)
        
        ddicts.append(ddict)

    return ddicts


# Store the data so that detectron2 can work with it
dataset_names = []
# USER: update thing_classes
for key in datasets_all.keys():
    name = EXPERIMENT_NAME +'_'+key
    DatasetCatalog.register(name, lambda k=key: get_ddicts(*datasets_all[k]),)
    MetadataCatalog.get(name).set(thing_classes=[""]) # can set to 'Spheroidite' but labels can
                                                        # overwhelm images with many instances
    dataset_names.append(name)



##### Set up detectron2 configurations for Mask R-CNN model
cfg = get_cfg()  # initialize configuration
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask_rcnn with ResNet 50 backbone and feature pyramid network (FPN)

cfg.INPUT.MASK_FORMAT = 'bitmask' # 'polygon' or 'bitmask'
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
cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD

#  Number of different classes of instances that will be predicted.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (powder particle)

# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
cfg.TEST.DETECTIONS_PER_IMAGE = 600  ## UPPER LIMIT OF NUMBER INSTANCES THAT WILL BE RETURNED, ADJUST ACCORDINGLY

# Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
# Adjust this to determine how long the model trains.
cfg.SOLVER.MAX_ITER = 20000   





##### Verify ground-truth masks are loaded correctly
# overlays ground truth instances on images and saves them.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # weights and tensorboard metrics will be stored here

outdir = pathlib.Path(str(cfg.OUTPUT_DIR))
figure_root = pathlib.Path(outdir, 'Figures', 'masks')
gt_figure_root = pathlib.Path(figure_root, 'ground_truth')
pred_figure_root = pathlib.Path(figure_root, 'predicted')

os.makedirs(gt_figure_root, exist_ok=True)
os.makekdirs(pred_figure_root, exist_ok=True)
assert gt_figure_root.is_dir(), pred_figure_root.is_dir()
for dataset in dataset_names:
    for d in DatasetCatalog.get(dataset):
        img_path = pathlib.Path(d['file_name'])
        visualizer = Visualizer(cv2.imread(str(img_path)), metadata=MetadataCatalog.get(dataset), scale=1)
        vis = visualizer.draw_dataset_dict(d)
        fig, ax = plt.subplots(figsize=(10,5), dpi=300)
        ax.imshow(vis.get_image())
        ax.axis('off')
        ax.set_title('{}\n{}'.format(dataset, img_path.name))
        fig.tight_layout()
        fig_path = pathlib.Path(gt_figure_root, '{}-n={}\n{}.png'.format(dataset,d['num_instances'],
                                                                         '{}'.format(img_path.stem)))
        fig.savefig(fig_path, bbox_inches='tight')
        if matplotlib.get_backend() is not 'agg':  # if gui session is used, show images
            plt.show()
        plt.close('all')


##### Train with model checkpointing

train = False  # make True to retrain, False to skip training (ie when you only want to evaluate)
if train:
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('training model')
    trainer.train() # uncomment to retrain model
else:
    print('skipping training, using results from previous training')


# get path of model weight at each checkpoint, 
# sorted in order of the number of training steps, followed by the last model
checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

print('checkpoint paths found:\n\t{}'.format('\n\t'.join([x.name for x in checkpoint_paths])))

last_only=True
if last only:
    checkpoint_paths = checkpoint_paths[-1:]

for p in checkpoint_paths:

    cfg.MODEL.WEIGHTS = os.path.join(p)

        ### visualization of predicted masks on all images
        # make directory for output mask predictions
    outdir = pathlib.Path(pred_figure_root,p.stem)
    os.makedirs(outdir, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    outputs = {}
    outputs_np = {}
    for dataset in dataset_names:
        for d in DatasetCatalog.get(dataset): # TODO  replace with datasetloader to make this less hacky
            img_path = pathlib.Path(d['file_name'])
            print('image filename: {}'.format(img_path.name))
            img = cv2.imread(str(img_path))

            # overlay predicted masks on image
            out = predictor(img)
            outputs[img_path.name] = {'outputs': out,'file_name': img_path.name, 'dataset': dataset} # store prediction outputs in dictionary
            outputs_np[img_path.name] = {'outputs': data_utils.instances_to_numpy(out['instances']),
                                         'file_name': img_path.name, 'dataset': dataset} # store outputs as numpy
                                        # for analysis on machines without detectron2/gpu

            v = Visualizer(img, metadata=MetadataCatalog.get(dataset))
            draw = v.draw_instance_predictions(out['instances'].to('cpu'))

            # save images with standard formatting
            fig, ax = plt.subplots(figsize=(10,5), dpi=300)
            ax.imshow(draw.get_image())
            ax.set_title(title)
            ax.axis('off')
            fig.tight_layout()
            print(outdir, title, '.png')

            fig.savefig(pathlib.Path(outdir, title.replace('\n','_')), bbox_inches='tight')
            plt.close('all')

    pickle_path = pathlib.Path(outdir, 'outputs.pickle')
    print('saving predictions to {}'.format(pickle_path))
    with open(pickle_path, 'wb') as f:
        pickle.dump(outputs, f)
    with open(pathlib.Path(outdir, 'outputs_np.pickle'), 'wb') as f:
        pickle.dump(outputs_np, f)


