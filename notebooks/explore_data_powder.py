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
import pickle
import skimage.io

## detectron2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

import data_utils

# verify cuda is installed and running correctly
import torch
from torch.utils.cpp_extension import CUDA_HOME


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
        record['num_instances'] = len(objs)
        dataset_dicts.append(record)
    return dataset_dicts


# In my data I have the training/validation data in a single VIA json file.
# This function splits the data into distinct groups based on a specific attribute (specified by get_subset function)
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


if __name__ == '__main__':
    print(torch.cuda.is_available(), CUDA_HOME)
    EXPERIMENT_NAME = 'particles' # can be 'particles' or 'satellites'

    json_dict = {'particles':'../data/raw/via_2.0.8/via_powder_particle_masks.json',
                 'satellites':'../data/raw/via_2.0.8/via_satellite_masks.json'}


    json_path = json_dict[EXPERIMENT_NAME]


    ddicts = get_data_dicts(json_path)
    subs = split_data_dict(ddicts)

    # Store the data so that detectron2 can work with it
    dataset_names = []
    # USER: update thing_classes
    # can use np.unique() on class labels to automatically generate

    for key, value in subs.items():
        name = EXPERIMENT_NAME + '_' + key
        DatasetCatalog.register(name, lambda key=key: subs.get(key))
        MetadataCatalog.get(name).set(thing_classes=[""])  # labels removed because they crowd images.
        dataset_names.append(name)

    ##### Set up detectron2 configurations for Mask R-CNN model
    cfg = get_cfg()  # initialize configuration
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask rcnn preset config

    cfg.INPUT.MASK_FORMAT = 'bitmask'  # 'polygon' or 'bitmask'.
    cfg.DATASETS.TRAIN = ("{}_Training".format(EXPERIMENT_NAME),)  # name of training dataset (must be registered)
    cfg.DATASETS.TEST = ("{}_Validation".format(EXPERIMENT_NAME),)  # name of test dataset (must be registered)

    cfg.SOLVER.IMS_PER_BATCH = 1  # Number of images per batch across all machines.
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # save checkpoint (model weights) after this many iterations
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD  # validation loss will be computed at every checkpoint

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (spheroidite)

    cfg.TEST.DETECTIONS_PER_IMAGE = 600  # maximum number of instances that will be detected by the model

    # Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
    cfg.SOLVER.MAX_ITER = 20000

    # If the weights are locally available, use those. Otherwise, download them from model zoo.
    # Needed when computer does not have network access/permission to download files.
    weights_path = pathlib.Path('../', 'models', 'model_final_f10217.pkl')  # path where downloaded weights is stored
    if weights_path.is_file():
        print('Weights found')
        cfg.MODEL.WEIGHTS = '../models/model_final_f10217.pkl'
    else:
        weights_source = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        print('Weights not found, downloading from source: {}'.format(weights_source))
        cfg.MODEL.WEIGHTS = weights_source

    #### Set up directory for storing model checkpoints, training metrics, and figures
    outdir = pathlib.Path(cfg.OUTPUT_DIR)  # root output directory
    figure_root = pathlib.Path(outdir, 'Figures', 'masks')  # directory for mask predictions
    gt_figure_root = pathlib.Path(figure_root, 'ground_truth')  # ground truth masks for verification
    pred_figure_root = pathlib.Path(figure_root, 'predicted')  # predicted masks

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(gt_figure_root, exist_ok=True)
    os.makedirs(pred_figure_root, exist_ok=True)

    # Visualize gt masks and save resulting figures
    for dataset in dataset_names:
        for d in DatasetCatalog.get(dataset):
            print('visualizing gt instances for {}'.format(d['file_name']))
            data_utils.quick_visualize_instances(d, gt_figure_root, dataset)

    # Train with model checkpointing
    train = True  # make True to retrain, False to skip training (ie when you only want to evaluate)
    if train:
        trainer = data_utils.AmpisTrainer(cfg)
        trainer.build_evaluator(cfg)
        trainer.resume_or_load(resume=False)
        print('training model')
        trainer.train()  # uncomment to retrain model
    else:
        print('skipping training, using results from previous training')


    # get path of model weight at each checkpoint,
    # sorted in order of the number of training steps, followed by the last model
    checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

    print('checkpoint paths found:\n\t{}'.format('\n\t'.join([x.name for x in checkpoint_paths])))

    last_only = True  # if True, only view masks for final model.
    #    Else, view predictions for all models.
    if last_only:
        checkpoint_paths = checkpoint_paths[-1:]

    for p in checkpoint_paths:
        cfg.MODEL.WEIGHTS = os.path.join(p)
        outdir = '../figures/masks/predictions/{}'.format(p.stem)
        os.makedirs(outdir, exist_ok=True)
        predictor = DefaultPredictor(cfg)
        outputs = {}  # outputs as detectron2 instances objects
        outputs_np = {}  # outputs as numpy arrays
        for dataset in dataset_names:
            for d in DatasetCatalog.get(dataset):  # TODO  replace with datasetloader to make this less hacky
                img_path = pathlib.Path(d['file_name'])
                print('image filename: {}'.format(img_path.name))
                img = cv2.imread(str(img_path))

                # overlay predicted masks on image
                out = predictor(img)
                outputs[img_path.name] = {'outputs': out, 'file_name': img_path.name,
                                          'dataset': dataset}  # store prediction outputs in dictionary
                outputs_np[img_path.name] = {'outputs': data_utils.instances_to_numpy(out['instances']),
                                             'file_name': img_path.name, 'dataset': dataset} # stores outputs as numpy
                data_utils.quick_visualize_instances(out['instances'].to('cpu'),
                                                     outdir, dataset, gt=False, img_path=img_path)

        pickle_out_path = pathlib.Path(outdir, 'outputs.pickle')
        print('saving predictions to {}'.format(pickle_out_path))
        with open(pickle_out_path, 'wb') as f:
            pickle.dump(outputs, f)
        with open(pathlib.Path(outdir, 'outputs_np.pickle'), 'wb') as f:
            pickle.dump(outputs_np, f)
