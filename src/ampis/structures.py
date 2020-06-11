import copy
import numpy as np
import pandas as pd
import pathlib
import pycocotools.mask as RLE
import skimage
from skimage.draw import polygon2mask
import skimage.measure
import torch
from typing import List, Union

from detectron2.structures import Boxes, BitMasks, PolygonMasks, Instances

from . import structures, visualize


class RLEMasks:
    """
    Class for adding RLE masks to Instances object.
    Supports advanced indexing (such as indexing with an array or another list).
    This allows for selecting a subset of masks, which is not possible with the
    standard list(dic) of RLE masks.
    """
    def __init__(self, rle):
        """

        Args:
            rle: n element list(dic) of RLE encoded masks
        """
        self.rle = rle

    def __getitem__(self, item: Union[int, slice, List[int], List[bool],
                                      torch.BoolTensor, np.ndarray]):
        idx_list = False
        if type(item) == int:
            return RLEMasks(self.rle[item])

        elif type(item) == torch.BoolTensor:
            return RLEMasks([mask for mask, bool_ in zip(self.rle, item) if bool_])

        elif type(item) == np.ndarray:
            if item.dtype == np.bool:
                assert item.shape[0] == len(self)
                return RLEMasks([mask for mask, bool_ in zip(self.rle, item) if bool_])
            else:
                idx_list = True

        elif type(item) == slice:
            return RLEMasks(self.rle[item])

        elif type(item) == list:
            if type(item[0]) == bool:
                assert len(item) == len(self)
                return RLEMasks([mask for mask, bool_ in zip(self.rle, item) if bool_])
            else:
                idx_list = True

        else:
            idx_list = True

        if idx_list:
            # list, (tuple, array, tensor, etc) of integer indices
            return RLEMasks([self.rle[idx] for idx in item])

        else:
            raise("invalid indices")

    def __len__(self):
        return(len(self.rle))


class instance_set(object):
    """
    Simple way to organize a set of instances for a single image to ensure
    that formatting is consistent.
    """

    def __init__(self, mask_format=None, bbox_mode=None, file_path=None, annotations=None, instances=None, img=None,
                 dataset_class=None, pred_or_gt=None, HFW=None, randomstate=None):

        self.mask_format = mask_format  # 'polygon' or 'bitmask'
        self.bbox_mode = bbox_mode  # from detectron2.structures.BoxMode
        self.img = img  # image r x c x 3
        self.filepath = file_path  # file name or path of image
        self.dataset_class = dataset_class  # 'Training', 'Validation', 'Test', etc
        self.pred_or_gt = pred_or_gt  # 'gt' for ground truth, 'pred' for model prediction
        self.HFW = HFW  # Horizontal Field Width of image. Can be float or string with value and units.
        self.rprops = None  # region props, placeholder for self.compute_regionprops()
        self.instances = instances
        self.annotations = annotations
        if randomstate is None:  # random state used for color assignment during visualization
            randomstate = np.random.randint(2 ** 32 - 1)
        self.randomstate = randomstate
        self.colors = None

    def read_from_ddict(self, ddict, return_=False):
        """
        Read ground-truth labels from ddict (see get_data_dicts)

        inputs:
        :param ddict: list(dic) from get_data_dicts
        :param return_: if True, function will return the instance_set object
        TODO update documentation for explore-data files to describe necessary fields
        or create ddict class?
        """

        # default values-always set
        self.pred_or_gt = 'gt'  # ddict assumed to be ground truth labels from get_ddict function

        # required values- function will error out if these are not set
        self.filepath = pathlib.Path(ddict['file_name'])
        self.mask_format = ddict['mask_format']
        image_size = (ddict['height'], ddict['width'])
        # instances_gt = annotations_to_instances(ddict['annotations'], image_size, self.mask_format)

        class_idx = np.asarray([anno['category_id'] for anno in ddict['annotations']], np.int)
        bbox = np.stack([anno['bbox'] for anno in ddict['annotations']])
        segs = [anno['segmentation'] for anno in ddict['annotations']]
        segtype = type(segs[0])
        if segtype == dict:
            # RLE encoded mask
            masks = RLEMasks(segs)

        elif segtype == np.ndarray:
            if segs[0].dtype == np.bool:
                #  bitmask
                masks = BitMasks(np.stack(segs))

        else:
            # list of (list or array) of coords in format [x0,y0,x1,y1,...xn,yn]
            masks = PolygonMasks(segs)

        instances = Instances(image_size, **{'masks': masks,
                                             'boxes': bbox,
                                             'class_idx': class_idx})
        self.instances = instances
        self.instances.colors = visualize.random_colors(len(instances), self.randomstate)

        # optional values- default to None if not in ddict
        self.dataset_class = ddict.get('dataset_class', None)
        self.HFW = ddict.get('HFW', None)
        if return_:
            return self
        return

    def read_from_model_out(self, outs, return_=False):
        """
        Read predicted labels from output of detectron2 predictor, formatted
        with data_utils.format_outputs() function.

        inputs:
        :param outs: dictionary with following structure:
        {'file_name': name of file, string or path object
        'dataset': string name of dataset, should end with
                    _Training, _Validation, _Test, etc
        'pred': dictionary of detectron2 predictor outputs}
        :param return_: if True, function will return the instance_set object
        """
        self.pred_or_gt = 'pred'
        self.mask_format = 'bitmask'  # model outs assumed to be RLE bitmasks

        self.filepath = outs['file_name']
        self.dataset_class = outs['dataset'].split('_')[1]  # Training, Validation, etc

        instances_pred = outs['pred']['instances']

        instances = Instances(instances_pred.image_size,
                              **{'masks': structures.RLEMasks(instances_pred.pred_masks),
                                 'boxes': instances_pred.pred_boxes,
                                 'class_idx': instances_pred.pred_classes,
                                 'scores': instances_pred.scores})
        self.instances = instances
        self.instances.colors = visualize.random_colors(len(self.instances), self.randomstate)

        if return_:
            return self

    def filter_mask_size(self, min_thresh=100, max_thresh=100000, to_rle=False):
        """
        Remove instances with mask areas outside of the interval (min_thresh, max_thresh.)

        inputs:
        :param instances:- instances object
        :param min_thresh: int- minimum mask size threshold in pixels, default 100, or None to not use this criteria
        :param max_thresh: int- maximum mask size threshold in pixels, default 100000, or None to not use this criteria
        :param to_rle: bool- if True, all masks will be converted to RLE before measuring
                        so the pixels can be counted directly
        returns:
            * instances_filtered-instances object which only includes instances within given size thresholds
        """
        masks = self.instances.masks
        if to_rle:
            masks = RLEMasks(masks_to_rle(masks, self.instances.image_size))
        masktype = type(masks)
        # determine which instances contain inlier masks
        areas = mask_areas(masks)

        if min_thresh is None:
            inlier_min = np.ones(areas.shape, np.bool)
        else:
            inlier_min = areas > min_thresh
        if max_thresh is None:
            inlier_max = np.ones(areas.shape, np.bool)
        else:
            inlier_max = areas < max_thresh

        inliers_bool = np.logical_and(inlier_min, inlier_max)

        new_instance_fields = {}
        # for key, value in self.instances._fields.items():
        #     print(key)
        #     print(type(value))
        #     print(value)
        #     temp = value[inliers_bool]
        #     new_instance_fields[key] = temp

        # can't iterate through polygonmasks properly, case must be handled separately
        if masktype == PolygonMasks:
            polygons = [p for p, b in zip(masks.polygons, inliers_bool) if b]
            masks = PolygonMasks(polygons)
        else:
            masks = masks[inliers_bool]

        new_instance_fields = {}
        for key, value in self.instances._fields.items():
            if key == 'masks':
                new_instance_fields[key] = masks
            else:
                new_instance_fields[key] = value[inliers_bool]

        instances_filtered = Instances(self.instances.image_size,
                                       **new_instance_fields)
        return instances_filtered

    # TODO decompress masks, this will not work in the current state
    def compute_rprops(self, keys=None, return_df=False):
        """Applies skimage.measure.regionprops_table to masks for analysis.
        :param keys: properties to be stored in dataframe. If None, default values will be used.
        Default values: 'area', 'equivalent_diameter','major_axis_length', 'perimeter','solidity','orientation.'
        For more info, see:
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops_table.
        :param return_df: bool, if True, function will return dataframe

        stored:
            self.rprops: pandas dataframe with desired properties, and an additional column for the class_idx.
        returns:
           self.rprops will be returned as a dataframe if return_df is True
        """

        if keys is None:  # use default values
            keys = ['area', 'equivalent_diameter', 'major_axis_length', 'perimeter', 'solidity', 'orientation']
        rprops = [skimage.measure.regionprops_table(masks_to_bitmask_array(mask, self.instances.image_size).squeeze()
                                                    .astype(np.int), properties=keys) for mask in self.instances.masks]
        df = pd.DataFrame(rprops)
        df['class_idx'] = self.instances.class_idx
        self.rprops = df

        if return_df:
            return self.rprops

    def copy(self):
        """
        returns a copy of the instance_set object
        """
        return copy.deepcopy(self)


def mask_areas(masks):
    """
    Computes area in pixels of each mask in masks
    Args:
        masks: bitmask, polygonmask, or array containing masks

    Returns: n_mask element array of mask areas

    TODO is this limited to 80 elements like IOU scores?

    TODO Test this
        For RLE masks, look into coco api   mask.area
        For polygon masks, look into
            def PolyArea(x,y):
                return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    """

    masktype = type(masks)

    if masktype == np.ndarray:
        # masks are already expanded, compute area directly
        return masks.sum(axis=(1, 2), dtype=np.uint)

    elif masktype == PolygonMasks:
        # polygon masks given as array of coordinates [x0,y0,x1,y1,...xn,yn]
        area = np.asarray([_shoelace_area(coords[0][::2], coords[0][1::2]) for coords in masks.polygons])
        return area

    elif masktype == list and type(masks[0]) == dict:
        # RLE encoded masks
        return RLE.area(masks)
    elif masktype == RLEMasks:
        return RLE.area(masks.rle)


def _shoelace_area(x, y):
    """
    Computes area of simple polygon from coordinates
    shoelace formula https://en.wikipedia.org/wiki/Shoelace_formula
    implementation from:
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Args:
        x: n element array of x coordinates
        y: n element array of y coordinates
    Returns: area- float- area of polygon in pixels
    # TODO verify this work compared to boolean mask areas
    """
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def boxes_to_array(boxes):
    """
    Helper function to convert any type of object representing bounding boxes to a n_box * 4 numpy array
    Args:
        boxes:

    Returns:

    """
    dtype = type(boxes)

    if dtype == np.ndarray:
        return boxes

    elif dtype == list:
        assert len(boxes[0]) == 4
        return np.asarray(boxes)

    elif dtype == Boxes:
        return boxes.tensor.to('cpu').numpy()


def masks_to_rle(masks, size=None):
    """

    Args:
        masks:
        size: only needed for polygonmasks- tuple(r, c) r, c are height and width of image in pixels

    Returns:

    """
    if type(masks) == list:
        if type(masks[0]) == dict:
            # assumed to already be in RLE format
            return masks
        elif type(masks[0]) == list:
            raise(NotImplementedError('):'))

    if type(masks) == RLEMasks:
        rle = masks.rle
        return rle

    elif type(masks) == PolygonMasks:
        assert size is not None
        rle = [RLE.frPyObjects(p, *size)[0] for p in masks.polygons]

        return rle
        # for mask in masks:
        #     cords = mask.polygon
        #     polygon2mask()


def _poly2mask(masks, size):
    """
    Helper function since polygon masks can be lists, arrays, or PolygonMask instances
    Args:
        masks:

    Returns:

    """
    return np.stack([polygon2mask(  # stack along axis 0 to get (n x r x c)
        size, np.stack((p[1::2],  # x coords
                        p[0::2]),  # y coords
                       axis=1))  # stack along axis 1 to get (n_p x 2)
        for p in masks])  # for every polygon


def masks_to_bitmask_array(masks, size=None):
    """

    Args:
        masks:

    Returns:

    """
    dtype = type(masks)


    if dtype == np.ndarray:
        # masks are already array
        assert masks.dtype == np.bool
        return masks

    elif dtype == PolygonMasks:
        assert size is not None
        polygons = [x[0] for x in masks.polygons]
        bitmasks = _poly2mask(polygons, size)
        return _poly2mask(polygons, size)




    elif dtype == list:
        if type(masks[0]) == dict:
            # RLE masks
            return RLE.decode(masks).astype(np.bool).transpose((2, 0, 1))
        elif type(masks[0]) == list or type(masks[0]) == np.ndarray:
            assert size is not None
            bitmasks = _poly2mask(masks, size)
            return _poly2mask(masks, size)
        else:
            raise NotImplementedError


    elif dtype == RLEMasks:
        bitmask = RLE.decode(masks.rle).astype(np.bool)
        if bitmask.ndim == 2:  # only 1 mask
            return bitmask[np.newaxis,:,:]
        else:  # multiple masks, reorder to (n_mask x r x c
            return bitmask.transpose((2, 0, 1))

    else:
        raise NotImplementedError