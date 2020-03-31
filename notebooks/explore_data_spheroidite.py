##### Module imports
gui = True
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


def get_files(pickle_file, n_test):
    """
    TODO document this
    Args:
        pickle_file:
        n_test:

    Returns:

    """
    pickle_file = pathlib.Path(pickle_file)
    data_root = pickle_file.parent

    with open(pickle_file, 'rb') as f:
        filename_subset = sorted(pickle.load(f))  # TODO remove sorting?

    filename_selector = lambda pathname, pattern: pathname.stem.replace(pattern,'').split('_sizeRC')[0]
    image_paths = {filename_selector(x, 'micrograph-'): x for x in data_root.glob('micrograph*')}
    annotation_paths = {filename_selector(x, 'annotation-'): x for x in data_root.glob('annotation*')}

    image_subset = [image_paths.get(x) for x in filename_subset if image_paths.get(x) is not None]
    annotation_subset = [annotation_paths.get(x) for x in filename_subset if annotation_paths.get(x) is not None]

    img_train = image_subset[:-n_test]
    img_valid = image_subset[-n_test:]

    ann_train = annotation_subset[:-n_test]
    ann_valid = annotation_subset[-n_test:]

    subset_dict = lambda ipath_, apath_, dclass_: {'image_paths': ipath_,
                                                   'annotation_paths': apath_,
                                                   'dclass': dclass_}
    train_set = subset_dict(img_train, ann_train, 'Training')
    valid_set = subset_dict(img_valid, ann_valid, 'Validation')

    file_subsets = {'Training': train_set,
                    'Validation': valid_set,}

    return file_subsets


##### Load data in a way that is consistent with detectron2 formatting

def get_ddicts(file_subset):
    """
    file_subset- dictionary containing img_paths, label_paths, and dclass.
    See output of get_files()
    img_paths: list of  path to original images
    label_paths: list of paths to annotation image (same order as img_paths )
    dclass:  'Training,' 'Validation', 'Testing', etc 
    """ 
    img_paths = file_subset['image_paths']
    label_paths = file_subset['annotation_paths']
    dclass = file_subset['dclass']
    metadata = get_metadata()

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


def get_metadata():
    """
    Returns metadata for the dataset.
    For the spheroidite dataset, metadata is hard-coded
    as it is not stored in any file.
    Returns:
        Metadata- dictionary containing metadata to register to MetadataCatalog
    """
    Metadata = {'thing_classes': ['spheroidite']}
    return Metadata


if __name__ == '__main__':
    print(torch.cuda.is_available(), CUDA_HOME)

    pickle_path = pathlib.Path('..', 'data', 'raw', 'spheroidite-images', 'spheroidite-files.pickle')
    assert pickle_path.is_file()
    datasets_all = get_files(pickle_path, 2)

    EXPERIMENT_NAME = 'spheroidite'  # TODO set up json control file with all parameters

    print('train_set: {}'.format([pathlib.Path(x).name for x in datasets_all['Training']['image_paths']]))
    print('valid set: {}'.format([pathlib.Path(x).name for x in datasets_all['Validation']['image_paths']]))

    # Store the data so that detectron2 can work with it
    dataset_names = []
    # USER: update thing_classes
    for key in datasets_all.keys():
        name = EXPERIMENT_NAME + '_' + key
        DatasetCatalog.register(name, lambda k=key: get_ddicts(datasets_all[k]))
        MetadataCatalog.get(name).set(**get_metadata())
        # overwhelm images with many instances
        dataset_names.append(name)

    ##### Set up detectron2 configurations for Mask R-CNN model
    cfg = get_cfg()  # initialize configuration
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask rcnn preset config

    cfg.INPUT.MASK_FORMAT = 'bitmask'  # 'polygon' or 'bitmask'.
    cfg.DATASETS.TRAIN = ("{}_Training".format(EXPERIMENT_NAME),)  #  name of training dataset (must be registered)
    cfg.DATASETS.TEST = ("{}_Validation".format(EXPERIMENT_NAME),)  # name of test dataset (must be registered)

    cfg.SOLVER.IMS_PER_BATCH = 1  # Number of images per batch across all machines.
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # save checkpoint (model weights) after this many iterations
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD  # validation loss will be computed at every checkpoint

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (spheroidite)

    cfg.TEST.DETECTIONS_PER_IMAGE = 600  # maximum number of instances that will be detected by the model

    # Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
    cfg.SOLVER.MAX_ITER = 20000

    # If the weights are locally available, use those. Otherwise, download them from model zoo.
    # Needed when computer does not have network access/permission to download files.
    weights_path = pathlib.Path('../', 'models', 'model_final_f10217.pkl')  # path where downloaded weights is stored
    if weights_path.is_file():
        print('Weights found: {}'.format(weights_path.relative_to('./')))
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
            data_utils.quick_visualize_instances(d, gt_figure_root, dataset, suppress_labels=True)

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

    last_only = True
    if last_only:
        checkpoint_paths = checkpoint_paths[-1:]

    for p in checkpoint_paths:
        cfg.MODEL.WEIGHTS = os.path.join(p)
        ### visualization of predicted masks on all images
        # make directory for output mask predictions
        outdir = pathlib.Path(pred_figure_root, p.stem)
        os.makedirs(outdir, exist_ok=True)
        predictor = DefaultPredictor(cfg)
        outputs = {} # raw outputs from model
        outputs_np = {} # outputs converted to numpy arrays for easier analysis

        for dataset in dataset_names:
            for d in DatasetCatalog.get(dataset):
                img_path = pathlib.Path(d['file_name'])
                print('image filename: {}'.format(img_path.name))
                img = cv2.imread(str(img_path))

                # overlay predicted masks on image
                out = predictor(img)
                outputs[img_path.name] = {'outputs': out, 'file_name': img_path.name,
                                          'dataset': dataset}  # store prediction outputs in dictionary
                outputs_np[img_path.name] = {'outputs': data_utils.instances_to_numpy(out['instances']),
                                             'file_name': img_path.name, 'dataset': dataset}  # store outputs as numpy
                data_utils.quick_visualize_instances(out['instances'].to('cpu'),
                                                     outdir, dataset, gt=False, img_path=img_path)

        pickle_out_path = pathlib.Path(outdir, 'outputs.pickle')
        print('saving predictions to {}'.format(pickle_out_path))
        with open(pickle_out_path, 'wb') as f:
            pickle.dump(outputs, f)
        with open(pathlib.Path(outdir, 'outputs_np.pickle'), 'wb') as f:
            pickle.dump(outputs_np, f)



