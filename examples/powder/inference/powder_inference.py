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
import os
from pathlib import Path
import pickle
import sys

## detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

ampis_root = Path('../../../src/')
sys.path.append(str(ampis_root))

from ampis.data_utils import format_outputs
from ampis.visualize import quick_visualize_ddicts


def get_model_cfg(which, weights):
    """
    Gets the configuration for the detectron model.

    Parameters
    ---------
    which: str
        either 'particles' or 'satellites', indicating which model this will be used for.
    weights: str or Path object
        path to model weights to be used

    Returns
    ---------
    cfg: detectron2 config objeect
        configuration which can be used for model inference.
    """

    if which not in ['particles', 'satellites']:
        raise(ValueError("must specify 'particles' or 'satellites' for configuration"))

    assert Path(weights).is_file()  # verify weight path exists

    # these config settings are the same for both models
    cfg = get_cfg()  # initialize config
    # both models are derived from mask rcnn + fpn base config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # each model only has 1 output class
    cfg.MODEL.WEIGHTS = str(weights)

    # these config settings are different for the two models
    if which == 'particles':
        cfg.TEST.DETECTIONS_PER_IMAGE = 600
    else:  # which == 'satellites'
        cfg.TEST.DETECTIONS_PER_IMAGE = 200

    return cfg


def main():
    # model weight paths
    model_root = Path('../../../models/2020_05_14_powder_inference_models/')
    particle_weights = model_root / '2020-04-23_13-13_15_particles_cval_4.pth'
    satellite_weights = model_root / '2020-04-23_17-11_22_satellites_cval_2.pth'

    # image paths
    data_root = Path('../../../data/raw/powder_inference/Sc1')
    files = sorted(data_root.glob('*'))

    # load model configurations for each model
    particle_cfg = get_model_cfg('particles', particle_weights)
    satellite_cfg = get_model_cfg('satellites', satellite_weights)

    outdir = Path('./output/')

    os.makedirs(outdir, exist_ok=True)

    particle_predictor = DefaultPredictor(particle_cfg)
    satellite_predictor = DefaultPredictor(satellite_cfg)

    outputs = dict()

    particle_dataset = 'particle_inference'
    satellite_dataset = 'satellite_inference'

    # run inference and overlay instances on image
    for img_path in files:
        img = cv2.imread(str(img_path))  # load image. detectron2 plays nicely with cv2
        particle_out = particle_predictor(img)  # particle inference
        satellite_out = satellite_predictor(img)  # satellite inference

        # visualize results and save image to outdir for each model
        quick_visualize_ddicts(particle_out['instances'], outdir, particle_dataset,
                               gt=False, img_path=img_path)
        quick_visualize_ddicts(satellite_out['instances'], outdir, satellite_dataset,
                               gt=False, img_path=img_path)

        # compress the outputs to RLE and store results
        outputs[img_path.name] = {
            'particles': format_outputs(particle_out, particle_dataset, particle_out),
            'satellites': format_outputs(satellite_out, satellite_dataset, satellite_out)
        }

    # write results for all files to disk
    with open(outdir / 'results.pickle', 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()
