"""
Miscellaneous tools, mostly used for model training.
Also provides the AmpisTrainer, which can get loss statistics on a validation set during training.
"""
import datetime
import json
import logging
import numpy as np
from pathlib import Path
import pycocotools.mask as RLE
import skimage
import skimage.io
import skimage.measure
import time
import torch

from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine.hooks import HookBase
from detectron2.engine.defaults import DefaultTrainer
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures import BoxMode


# HOOKS AND TRAINER NEEDED TO GET VALIDATION LOSS
# adapted from
# https://medium.com/@apofeniaco/
# training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
# and https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
class LossEvalHook(HookBase):
    """
    Adds validation loss statistics to AmpisTrainer class during training.
    """
    def __init__(self, eval_period, model, data_loader):
        """
        initializes LossEvalHook class

        Parameters
        ----------
        eval_period: int
            period on which to report validation metrics during training

        model: detectron2 model
            model that the hook is registered to

        data_loader: detectron2 data loader
            loads data in a format consistent with the model. See detectron2  documentation for more info.

        """
        super().__init__()
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        """
        Computes the loss on the validation set during training.
        """
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        metrics_dicts = []
        valid_losses_all = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch, metrics_dict = self._get_loss(inputs)
            losses.append(loss_batch)
            metrics_dicts.append(metrics_dict)
        mean_loss = np.mean(losses)
        
        for md in metrics_dicts:
            valid_losses_all.append(list(md.values()))
        valid_losses_all = np.asarray(valid_losses_all).mean(axis=0)

        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        for k, v in zip(md.keys(), valid_losses_all):
            self.trainer.storage.put_scalar('valid_'+k, v)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        """
        Computes loss from training loop.
        """
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced, metrics_dict

    def after_step(self):
        """
        method needed for hook to be automatically executed after training iterations.
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class AmpisTrainer(DefaultTrainer):
    """
    Extents detectron2 DefaultTrainer with validation loss metrics during training.

    If you do not have a validation set or do not need the validation metrics during training,
    just use the DefaultTrainer class.

    """
    def __init__(self, cfg, val_dataset=None):
        """
        initializes the trainer

        Parameters
        ----------
        cfg: detectron2 config
            configuration contains settings for the trainer

        val_dataset: str or None
            Name of validation dataset used. If None, dataset will be pulled from cfg.DATASETS.TEST[0]

        """

        if val_dataset is None:
            val_dataset = cfg.DATASETS.TEST[0]
        super().__init__(cfg)
        self.val_dataset = val_dataset

    def build_hooks(self):
        """
        Adds hooks to the trainer. This is needed to get the validation loss during training.

        """
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.SOLVER.CHECKPOINT_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


def extract_boxes(masks, mask_mode='detectron2', box_mode='detectron2'):
    """
    Extracts bounding boxes from boolean masks.

    For a boolean numpy array, bounding boxes are extracted from the min and max x and y coordinates containing
    pixels in each mask. Masks and boxes can be formatted for use with either  detectron2 (default) or
    the matterport visualizer.
    Detectron2 formatting:
        masks: n_mask x r x c boolean array
        boxes: (x0,y0,x1,y1) floating point box coordinates
    Matterport formatting:
        masks: r x c x n_mask boolean array
        boxes: [r1,r2,c1,c2] integer box indices

    Parameters
    ----------
        masks: ndarray
            boolean array of masks. Can be 2 dimensions for 1 mask or 3 dimensions for array of masks.
        mask_mode: str
            if 'detectron2,' masks are shape n_mask x r x c.
            if 'matterport,' masks are r x c x n_masks.
        box_mode: str
            if 'detectron2', boxes will be returned in [x1,y1,x2,y2] floating point format. (XYXY_ABS box mode)
            if 'matterport,' boxes will be returned in [y1,y2,x1,x2] integer format.

    Returns
    ---------
        boxes: ndarray
            n_mask x 4 array with bbox coordinates in the format and dtype and specified by box_mode.

    """

    if masks.ndim == 2:
        masks = masks[np.newaxis, :, :]

    else:
        if mask_mode == 'matterport':
            masks = masks.transpose((2, 0, 1))

    # by now, masks should be in n_mask x r x c format

    # matterport visualizer requires fixed

    dtype = np.float if box_mode == 'detectron2' else np.int
    boxes = np.zeros((masks.shape[0], 4), dtype=dtype)
    for i, m in enumerate(masks):

        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        if box_mode == 'detectron2':
            # boxes as absolute coordinates
            box = np.array([x1, y1, x2, y2], dtype=dtype)
        else:
            # boxes as array indices
            # offset x2 and y2 by 1 so they are included in indexing mask[y1:y2,x1:x2]
            # since indices are on the interval [i1, i2)
            box = np.array([y1, y2+1, x1, x2+1], dtype=dtype)

        boxes[i] = box

    return boxes


def compress_pred(pred):
    """
    Compresses predicted masks to RLE and converts other outputs to numpy arrays.

    Results in significantly smaller data structures that are easier to store and load into memory.

    Parameters
    -------
    pred: detectron2 Instances
        outputs that will be compressed.
 
    Returns
    -------
    pred_compressed: detectron2 Instances
        pred with masks compressed to RLE format and other outputs converted to numpy.

    """
    pred.pred_masks = [RLE.encode(np.asfortranarray(x.to('cpu').numpy())) for x in pred.pred_masks]
    pred.pred_boxes = pred.pred_boxes.tensor.to('cpu').numpy()
    pred.scores = pred.scores.to('cpu').numpy()
    pred.pred_classes = pred.pred_classes.to('cpu').numpy()
    return pred


def format_outputs(filename, dataset, pred):
    """
    Formats model outputs consistently to make analysis easier later

    Note that this function applies compress_pred() to pred, which modifies
    the instance predictions in-place to drastically reduce the space they take up. See compress_pred() documentation
    for more info.


    Parameters
    -----------
    filename:path or str
        path of image corresponding to outputs
    dataset: str
        'train' 'test' 'validation' etc that describes the class of the data
    pred: detectron2 Instances
        model outputs from detectron2 predictor.

    Returns
    ------------
        results- dictionary of outputs
    """

    compress_pred(pred['instances'])  # RLE encodes masks, converts tensors to numpy
    results = {'file_name': filename,
               'dataset': dataset,
               'pred': pred}

    return results


def get_ddicts(label_fmt, im_root, ann_root=None, pattern='*', dataset_class=None):
    """
    Reads images and their corresponding ground truth annotations in a format AMPIS can use.

    Data-dicts are read for single-class instance segmentation labels.
    Annotations can be stored in the following format, specified by *label_fmt*.
      * binary: Annotations are stored in separate images or numpy files. Annotation images must be the same size as
                the original images. In these images, background pixels are labeled 0, and pixels included in instances
                are labeled 1. skimage.measure.label() will be called to detect multiple instances included in the
                image, so it is assumed that 1) instances do not touch/overlap and 2) instances are not discontinuous.

      * label: Annotations are stored in separate images or numpy files, with the same size and naming requirements
               described above in 'binary'. In these images, background pixels are 0, and other pixel values indicate
               the unique instance ID that the pixel belongs to (similar to label images from skimage.measure.label().)

      * via2: Annotations are stored in JSON files generated from the VIA Image Annotator (version 2.) The exact format
              for the annotations can be found in the examples/documentation.

    Parameters
    ----------

    label_fmt: str
        Format of ground truth instance labels for each image. See above for detailed description. Options are:
          'binary': data_dicts will be read from binary images or npy files
          'label': data_dicts will be read from label images or npy files
          'via2': data_dicts will be read from VIA image annotator (version 2) JSON file

    im_root: str, Path object.
        path to directory containing images.
        If label_fmt == 'via2', im_root is the path to the JSON file generated by the VGG image annotator, which
        contains the paths to the images.
        Otherwise, im_root is the path to a folder containing images to be loaded in the dataset.

    ann_root: str, Path object, or None
        Only needed if label_fmt != 'via2'
        Path to directory containing annotations for each image in im_root. The annotation should have the same
        filename, except it may have a different extension. For example, the annotation for im_root/img1.png
        should be ann_root/img1.{png, tif, npy, etc...)

    pattern: str
        Glob pattern which can be used to select a subset files in im_root. By default, all files will be selected.

    dataset_class: str or None
        optional, can be used to indicate what set the data belongs to (ie 'Training','Testing','Validation',etc)


    Returns
    --------
    ddicts: list(dict)
        list of dictionaries containing the annotations for each image. Detailed information about the format is
        specified in the detectron2 documentation.

    """
    cwd = Path()
    im_root = Path(im_root)
    ann_root = Path(ann_root) if ann_root else None
    ddicts = []

    if label_fmt.lower() in ['binary', 'label']:
        img_paths = Path(im_root).glob(pattern)
        for idx, p in enumerate(img_paths):
            file_annotations = list(Path(ann_root).glob('*{}*'.format(p.stem)))  # find corresponding annotation file
            n = len(file_annotations)
            assert n == 1, f'There must be exactly 1 annotation file for, {p.name}, but {n} were found'
            ann_path = file_annotations[0].relative_to(ann_root)

            ann = np.load(str(ann_path)) if ann_path.suffix == '.npy' else skimage.io.imread(str(ann_path))
            height, width = ann.shape[:2]
            ddict = {'file_name': str(p.relative_to(cwd)),
                     'annotation_file':  str(ann_path),
                     'height': height,
                     'width': width,
                     'mask_format': 'bitmask',
                     'image_id':  idx,
                     'dataset_class': dataset_class}

            if label_fmt == 'binary':  # generate label image from binary image
                ann = skimage.measure.label(ann.astype(np.bool))

            unique = np.unique(ann)
            if unique[0] == 0:  # skip background pixels
                unique = unique[1:]

            annotations = []
            for u in unique:
                mask = ann == u
                bbox = extract_boxes(mask)[0]
                mask = RLE.encode(np.asfortranarray(mask))

                annotations.append({'bbox': bbox,
                                    'bbox_mode': BoxMode.XYXY_ABS,
                                    'segmentation': mask,
                                    'categroy_id': 0})

            ddict['annotations'] = annotations
            ddict['num_instances'] = len(annotations)

            ddicts.append(ddict)

    elif label_fmt.lower() == 'via2':
        with open(Path(im_root), 'rb') as f:
            j = json.load(f)

        img_dir = Path(im_root.parent, j['_via_settings']['core']['default_filepath'])

        for idx, annos in enumerate(j['_via_img_metadata'].values()):
            filename = Path(img_dir, annos['filename'])
            size = annos['file_attributes'].get('Size (width, height)', None)
            if size:
                width, height = tuple((int(x) for x in size.split(', ')))
            else:
                im = skimage.io.imread(filename, as_gray=True)
                height, width = im.shape
            hfw = annos['file_attributes'].get('HFW', None)

            ddict = {'file_name': str(filename.relative_to(cwd)),
                     'annotation_file':  im_root.name,
                     'height': height,
                     'width': width,
                     'mask_format': 'polygon',
                     'image_id':  idx,
                     'HFW': hfw,
                     'dataset_class': dataset_class}

            annotations = []
            for obj in annos['regions']:
                # polygon masks is list of polygon coordinates in format ([x0,y0,x1,y1...xn,yn]) as specified in
                # https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.PolygonMasks
                shape = obj['shape_attributes']
                px = shape['all_points_x']
                py = shape['all_points_y']
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                annotation = {
                    'bbox': np.asarray((np.min(px), np.min(py), np.max(px), np.max(py))),
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': [poly],
                    'category_id': 0,
                }
                annotations.append(annotation)
            ddict['annotations'] = annotations
            ddict['num_instances'] = len(annotations)

            ddicts.append(ddict)
    else:
        raise (ValueError("label_fmt must be 'binary','label', or 'via2'"))

    return ddicts

if __name__ == '__main__':
    # via ddict test
    print('test1')
    test = get_ddicts('via2',
                          '/home/ryan/Documents/School/Research/Projects/MatISeg/Projects/Powder/data/via_2.0.8/via_powder_particle_masks.json')
    print('test1 success')

    # binary ddict test with numpy arrays
    root = Path('/home/ryan/Documents/School/Research/Projects/MatISeg/Projects/Spheroidite/data/spheroidite-images')

    im_path = root  / 'images'
    ann_path = root / 'annotations'
    ann_npy = root / 'anno-npy'

    # binary ddict test with png images
    print('test2')
    test2 = get_ddicts('binary', im_path, ann_path)
    print('test2 success')

    # binary ddict test with npy images
    print('test3')
    test3 = get_ddicts('binary', im_path, ann_npy)
    print('test3 success')



