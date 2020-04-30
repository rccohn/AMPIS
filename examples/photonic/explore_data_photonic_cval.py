##### Module imports
import matplotlib
gui = False 
if __name__ == '__main__':
    if not gui:
        # make sure script doesn't break on non-gui jobs
        # (ie batch job on computing cluster)
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
import sys

## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode

ampis_root = pathlib.Path('../../src/')
sys.path.append(str(ampis_root))

from ampis import data_utils, visualize

# verify cuda is installed and running correctly
import torch
from torch.utils.cpp_extension import CUDA_HOME

##### Load data in a way that is consistent with detectron2 formatting

def get_data_dicts(data_root):
    png_files = sorted(data_root.glob('*.png'))
    pickle_files = sorted(data_root.glob('*.pickle'))
    dataset_dicts = []
    for i, (img_root, ann_root) in enumerate(zip(png_files, pickle_files)):
        assert img_root.stem == ann_root.stem.replace('label_', '')

        r1, r2, c1, c2 = [int(x) for x in img_root.stem.split('_')[-1].split('-')]

        record = {}
        record['file_name'] = str(img_root)
        record['height'] = r2 - r1
        record['width'] = c2 - c1
        record['image_id'] = i

        record['mask_format'] = 'bitmask'

        objs = []
        with open(ann_root, 'rb') as f:
            annos = pickle.load(f)

        for mask, box in zip(*annos.values()):
            obj = {'bbox': box,
                   'bbox_mode': BoxMode.XYXY_ABS,
                   'segmentation': mask,
                   'category_id': 0
                   }
            objs.append(obj)

        record['annotations'] = objs
        record['num_instances'] = len(objs)
        dataset_dicts.append(record)
    return dataset_dicts


def get_metadata():
    """
    For photonic crystals, there is only one class, so 
    metadata is hardcoded.
    Args:
        None

    Returns:
        metadata: dictionary of values to be registered to MetadataCatalog
    """
    metadata = {'thing_classes': ['particle']}

    return metadata


def main():
    print(torch.cuda.is_available(), CUDA_HOME)
    EXPERIMENT_NAME = 'photonic'

    data_root = pathlib.Path('../../data/raw/photonic')
    assert data_root.is_dir()

    ddicts = get_data_dicts(data_root)

    train_splits = []
    test_splits = []
    for i in range(len(ddicts)):
        train = ddicts[0:i]
        test = ddicts[i:i + 1]
        train += ddicts[i+1:]

        train_splits.append(train)
        test_splits.append(test)

    OUTPUT_ROOT = pathlib.Path('./output/')
    
    ddicts_all = []
    for i, (trainset, testset) in enumerate(zip(train_splits, test_splits)):
        # Store the data so that detectron2 can work with it
        dataset_names = []
        # USER: update thing_classes
        # can use np.unique() on class labels to automatically generate
        ddicts_all.append([])
        for j, (key, value) in enumerate( zip(['Training', 'Validation'], [trainset, testset])):
            ddicts_all[i].append(value)
            name = EXPERIMENT_NAME + '_' + key + 'cval_{}'.format(i)
            DatasetCatalog.register(name, lambda i1=i, j1=j: ddicts_all[i1][j1])
            MetadataCatalog.get(name).set(**get_metadata())  # labels removed because they crowd images.
            dataset_names.append(name)

        ##### Set up detectron2 configurations for Mask R-CNN model
        cfg = get_cfg()  # initialize configuration
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # use mask rcnn preset config

        cfg.INPUT.MASK_FORMAT = 'bitmask'  # 'polygon' or 'bitmask'.
        print(dataset_names)
        cfg.DATASETS.TRAIN = (dataset_names[0],)  # name of training dataset (must be registered)
        cfg.DATASETS.TEST = (dataset_names[1],)  # name of test dataset (must be registered)

        cfg.SOLVER.IMS_PER_BATCH = 1  # Number of images per batch across all machines.
        cfg.SOLVER.CHECKPOINT_PERIOD = 240  # save checkpoint (model weights) after this many iterations
        #cfg.TEST.EVAL_PERIOD = cfg.SOLVER.CHECKPOINT_PERIOD  # model evaluation (different from loss)

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (spheroidite)

        cfg.TEST.DETECTIONS_PER_IMAGE = 400  # maximum number of instances that will be detected by the model

        # Instead of training by epochs, detectron2 uses iterations. As far as I can tell, 1 iteration is a single instance.
        cfg.SOLVER.MAX_ITER = 3120

        # If the weights are locally available, use those. Otherwise, download them from model zoo.
        # Needed when computer does not have network access/permission to download files.
        weights_path = pathlib.Path('../','../', 'models', 'model_final_f10217.pkl')  # path where downloaded weights is stored
        if weights_path.is_file():
            print('Weights found')
            cfg.MODEL.WEIGHTS = str(weights_path)
        else:
            weights_source = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
            print('Weights not found, downloading from source: {}'.format(weights_source))
            cfg.MODEL.WEIGHTS = weights_source

        #### Set up directory for storing model checkpoints, training metrics, and figures
        cfg.OUTPUT_DIR = str(pathlib.Path(OUTPUT_ROOT, 'cval_{}'.format(i)))
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
                print('visualizing gt instances for {}'.format(pathlib.Path(d['file_name']).relative_to('./')))
                visualize.quick_visualize_ddicts(d, gt_figure_root, dataset, suppress_labels=True)

        # Train with model checkpointing
        train = True  # make True to retrain, False to skip training (ie when you only want to evaluate)
        if train:
            trainer = data_utils.AmpisTrainer(cfg)
            trainer.resume_or_load(resume=False)
            print('training model')
            trainer.train()  # uncomment to retrain model
        else:
            print('skipping training, using results from previous training')


        # get path of model weight at each checkpoint,
        # sorted in order of the number of training steps, followed by the last model
        checkpoint_paths = sorted(list(pathlib.Path(cfg.OUTPUT_DIR).glob('*model_*.pth')))

        print('checkpoint paths found:\n\t{}'.format('\n\t'.join([x.name for x in checkpoint_paths])))

        last_only = False  # if True, only view masks for final model.
        #    Else, view predictions for all models.
        if last_only:
            checkpoint_paths = checkpoint_paths[-1:]

        for p in checkpoint_paths:
            cfg.MODEL.WEIGHTS = os.path.join(p)
            outdir = '{}/{}'.format(pred_figure_root, p.stem)
            os.makedirs(outdir, exist_ok=True)
            predictor = DefaultPredictor(cfg)
            outputs = {}  # outputs as detectron2 instances objects
            for dataset in dataset_names:
                for d in DatasetCatalog.get(dataset):
                    img_path = pathlib.Path(d['file_name'])
                    print('image filename: {}'.format(img_path.name))
                    img = cv2.imread(str(img_path))
                    # overlay predicted masks on image
                    out = predictor(img)
                    outputs[img_path.name] = data_utils.format_outputs(img_path.name, dataset, out)
                    visualize.quick_visualize_ddicts(out['instances'],
                                                      outdir, dataset, gt=False, img_path=img_path)


            pickle_out_path = pathlib.Path(outdir, 'outputs.pickle')
            print('saving predictions to {}'.format(pickle_out_path))
            with open(pickle_out_path, 'wb') as f:
                pickle.dump(outputs, f)




if __name__ == '__main__':
    main()
