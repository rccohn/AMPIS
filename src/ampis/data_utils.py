import datetime
import logging
import numpy as np
import pycocotools.mask as RLE
import time
import torch

from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine.hooks import HookBase
from detectron2.engine.defaults import DefaultTrainer
import detectron2.utils.comm as comm
from detectron2.utils.logger import log_every_n_seconds


### HOOKS AND TRAINER NEEDED TO GET VALIDATION LOSS
# adapted from
# https://medium.com/@apofeniaco/
# training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
# and https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
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
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class AmpisTrainer(DefaultTrainer):
    def __init__(self, cfg, val_dataset=None):
        if val_dataset is None:
            val_dataset = cfg.DATASETS.TEST[0]
        super().__init__(cfg)
        self.val_dataset = val_dataset

    def build_hooks(self):
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
    Extracts bounding boxes from boolean masks. Can be formatted for use with
    either  detectron2 (default) or the matterport visualizer.
    Args:
        masks: boolean array of masks. Can be 2 dimensions for 1 mask or 3 dimensions for array of masks.
        mask shape specified by mask_mode.
        mask_mode: if 'detectron2,' masks are shape n_mask x r x c. if 'matterport,' masks are r x c x n_masks.
        box_mode: if 'detectron2', boxes will be returned in [x1,y1,x2,y2] floating point format. (XYXY_ABS box mode)
        if 'matterport,' boxes will be returned in [y1,y2,x1,x2] integer format.

    Returns:
        boxes: n_mask x 4 array with dtype and order of elements specified by box_mode input.
    """

    if masks.ndim == 2:
        masks = masks[np.newaxis, :, :]

    else:
        if mask_mode == 'matterport':
            masks = masks.transpose((2,0,1))

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
    Compresses predicted masks to RLE and converts outputs to numpy arrays. Results in
    massively smaller
    Args:
        pred: outputs of detectron2 predictor
 
    Returns:
        pred_compressed: pred with masks compressed to RLE format
    """
    pred.pred_masks = [RLE.encode(np.asfortranarray(x.to('cpu').numpy())) for x in pred.pred_masks]
    pred.pred_boxes = pred.pred_boxes.tensor.to('cpu').numpy()
    pred.scores = pred.scores.to('cpu').numpy()
    pred.pred_classes = pred.pred_classes.to('cpu').numpy()
    return pred

def format_outputs(filename, dataset, pred):
    """
    Formats model outputs consistently to make analysis easier later
    Args:
        filename: path of image corresponding to outputs
        dataset: 'train' 'test' 'validation' etc
        pred: model outputs from detectron2 predictor

    note that this function applies compress_pred() to pred, which modifies
    the instance predictions in-place to drastically reduce the space they take up
    Returns:
        results- dictionary of outputs
    """

    compress_pred(pred['instances'])  # RLE encodes masks, converts tensors to numpy
    results = {'file_name': filename,
               'dataset': dataset,
               'pred': pred}

    return results


