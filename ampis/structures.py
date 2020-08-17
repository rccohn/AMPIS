# Copyright (c) 2020 Ryan Cohn and Elizabeth Holm. All rights reserved.
# Licensed under the MIT License (see LICENSE for details)
# Written by Ryan Cohn
"""
Provides convenient data structures and methods for working with different masks
and collections of instances.
"""
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import pycocotools.mask as RLE
import skimage
from skimage.draw import polygon2mask
import skimage.measure
import torch
from typing import List, Union

from detectron2.structures import Boxes, BitMasks, PolygonMasks, Instances

from . import structures, visualize


class RLEMasks(object):
    """
    Class for adding RLE masks to Instances object.
    Supports advanced indexing (such as indexing with an array or another list).
    This allows for selecting a subset of masks, which is not possible with the
    standard list(dic) of RLE masks.

    """

    def __init__(self, rle):
        """
        Initialize the class instance.

        Parameters
        -------
        rle: list(dic)
            n element list(dic) of RLE encoded masks

        Attributes
        ----------
        rle: list(dic)
            stores the rle list so it can be retrieved.

        """
        super().__init__()
        self.rle = rle

    def __getitem__(self, item: Union[int, slice, List[int], List[bool],
                                      torch.BoolTensor, np.ndarray]):
        """
        Fancy indexing which allows for convenient selection of subsets of data from the RLEMasks object.
        """
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
        """
        get the length of self.rle. For convenience and also so it can be included in
        detectron2 Instances object.
        """
        return len(self.rle)


class InstanceSet(object):
    """
    Simple way to organize a set of instances for a single image.

    Ensures that formatting is consistent for convenient analysis.

    Attributes
    ---------
    mask_format: str
        can be 'polygon' or 'bitmask'. Indicates the format in which the masks are stored consistent
        with the formats used in detectron2.

    bbox_mode: detectron2.structures.BoxMode
        Indicates how the boxes are stored. For this package it should pretty much always be
        BoxMode.XYXY_ABS.

    img: ndarray
        r x c {x 3} array of pixel values of the image for which the InstanceSet contains instances for.

    filepath: string or Path object
        path to the image for which the InstanceSet containts instances for.

    dataset_class: str
        Describes which dataset the data in the InstanceSet belongs to (ie 'training','validation', etc)

    pred_or_gt: str
        'pred' indicates InstanceSet contains model predictions, 'gt' indicates
        it contains ground truth instance annotations.

    HFW: str or None
        Horizontal field width of *img.* Contains the value and units separated by a space, eg '100 um'

    rprops: None or list of skimage.measure.RegionProperties
        if defined, contains region properties (ie area, perimeter, convex_hull, etc) of each mask.

    instances: detectron2 Instances
        contains the instances (segmentation masks, bboxes, class_idx, scores, etc) for the instances.

    annotations: list(dic)
        contains the annotations for the image

    randomstate: None or int
        if None, a random state is determined from a random integer.
        If it is an int, this is the seed used to generate random colors for displaying the instances.

    colors: ndarray
        Array containing (randomly generated) colors used for visualizing the instances.

    """

    def __init__(self, mask_format=None, bbox_mode=None, filepath=None, annotations=None, instances=None, img=None,
                 dataset_class=None, pred_or_gt=None, HFW=None, HFW_units=None, randomstate=None):
        """
        initializes the InstanceSet instance.

        Parameters
        ---------
        mask_format: str
            can be 'polygon' or 'bitmask'. Indicates the format in which the masks are stored consistent
            with the formats used in detectron2.

        bbox_mode: detectron2.structures.BoxMode
            Indicates how the boxes are stored. For this package it should pretty much always be
            BoxMode.XYXY_ABS.

        filepath: string or Path object
            path to the image for which the InstanceSet containts instances for.

        dataset_class: str
            Describes which dataset the data in the InstanceSet belongs to (ie 'training','validation', etc)

        pred_or_gt: str
            'pred' indicates InstanceSet contains model predictions, 'gt' indicates
            it contains ground truth instance annotations.

        instances: detectron2 Instances
            contains the instances (segmentation masks, bboxes, class_idx, scores, etc) for the instances.

        annotations: list(dic)
            contains the annotations for the image

        HFW: float or None
            Horizontal field width of *img.*

        HFW_units: str or None
            units of length for HFW (ie 'um')

        randomstate: None or int
            if None, a random state is determined from a random integer.
            If it is an int, this is the seed used to generate random colors for displaying the instances.

        Attributes
        -----------

        img: None or ndarray
            Initialized to None.
            When image is loaded, it is stored as a r x c {x 3} array of pixel values of the
            image for which the InstanceSet contains instances for.

        rprops: None or list of skimage.measure.RegionProperties
            if defined, contains region properties (ie area, perimeter, convex_hull, etc) of each mask.

        colors: None or ndarray
            Array containing (randomly generated) colors used for visualizing the instances.

        """
        super().__init__()
        self.mask_format = mask_format  # 'polygon' or 'bitmask'
        self.bbox_mode = bbox_mode  # from detectron2.structures.BoxMode
        self.img = img  # image r x c x 3
        self.filepath = filepath  # file name or path of image
        self.dataset_class = dataset_class  # 'Training', 'Validation', 'Test', etc
        self.pred_or_gt = pred_or_gt  # 'gt' for ground truth, 'pred' for model prediction
        self.HFW = HFW  # Horizontal Field Width of image, float.
        self.HFW_units = HFW_units  # units associated with HFW, str.
        self.rprops = None  # region props, placeholder for self.compute_regionprops()
        self.instances = instances
        self.annotations = annotations
        if randomstate is None:  # random state used for color assignment during visualization
            randomstate = np.random.randint(2 ** 32 - 1)
        self.randomstate = randomstate
        self.colors = None

    def read_from_ddict(self, ddict, inplace=True):
        """
        Read ground-truth annotations from data dicts.

        Reads data dicts and stores the information as attributes of the InstanceSet object.
        The descriptions of the attributes are provided in the documentation for self.__init__().

        Parameters
        -----------
        ddict: list
            List of data dicts in format described below in Notes.

        inplace: bool
            If True, the object is modified in-place. Else, the InstanceSet object is returned.

        Returns
        -----------
        self (optinal): InstanceSet
            only returned if return_ == True

        Attributes
        -----------
        pred_or_gt: str
            set to 'gt' (it is assumed these are ground truth instances)

        filepath: Path object
            path to file described by annotations

        mask_format: str
            read from ddict['mask_format'], either 'bitmask' or 'polygonmask'

        instances: detectron2.structures.Instances object
            contains information about class label, bbox, and segmentation mask for each instance.
            Also assigns random colors for each instance for visualization.

        dataset_class: str or None
            read from ddict['dataset_class'], describes if the image is in the training, validation,
            or test set.

        HFW: float or None
            Horizontal field width of *img.*

        HFW_units: str or None
            units of length for HFW (ie 'um')


        Notes
        ------
                Data dicts should have the following format:
            {
            'file_name': str or Path object
                        path to image corresponding to annotations
            'mask_format': str
                          'polygonmask' if segmentation masks are lists of XY coordinates, or
                          'bitmask'  if segmentation masks are RLE encoded segmentation masks
            'height': int
                    image height in pixels
            'width': int
                    image width in pixels
            'annotations': list(dic)
                            list of annotations. See the annotation format below.
            'num_instances': int
                        equal to len(annotations)- number of instances present in the image
            }

        The dictionary format for the annotation dictionaries is as follows:
            {
            'category_id': int
                            numeric class label for the instance.
            'bbox_mode': detectron2.structures.BoxMode object
                        describes the format of the bounding box coordinates.
                        The default is BoxMode.XYXY_ABS.
            'bbox':  list(int)
                    4-element list of bbox coordinates
            'segmentation': list
                            list containing:
                              * a list of polygon coordinates (mask format is polygonmasks)
                              * dictionaries  of RLE mask encodings (mask format is bitmasks)
            }
        """

        # default values-always set
        self.pred_or_gt = 'gt'  # ddict assumed to be ground truth labels from get_ddict function

        # required values- function will error out if these are not set
        self.filepath = Path(ddict['file_name'])
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
        HFW = ddict.get('HFW', None)
        HFW_units = None
        if HFW is not None:
            try:
                HFW = float(HFW)
            except ValueError:
                split = HFW.split(' ')
                if len(split) == 2:
                    HFW = float(split[0])
                    HFW_units = split[1]
        self.HFW = HFW
        self.HFW_units = HFW_units

        if not inplace:
            return self
        return

    def read_from_model_out(self, outs, inplace=True):
        """
        Read model predictions formatted with data_utils.format_outputs() function.

        The relevant entries are stored as attributes in the InstanceSet object.

        Parameters
        ----------
        outs: dict
            dictionary of formatted model outputs with the following format:
            {
            'file_name': str or Path
                filename of image corresponding to predictions

            'dataset': str
                string describing dataset image is in (ie training, validation , test, etc)

            'pred': detectron2.structures.Instances object
                model outputs formatted with format_outputs(). Should have the following fields:
                    pred_masks: list
                            list of dictionaries of RLE encodings for the predicted masks

                    pred_boxes: ndarray
                            nx4 array of bounding box coordinates

                    scores: ndarray
                        n-element array of confidence scores for each instance (output from softmax of class label)

                    pred_classes: ndarray
                        n-element array of integer class indices for each predicted index

            }

        inplace: bool
            if True, the InstanceSet object is modified in place.
            Otherwise, the object is returned.

        Attributes
        -----------
        pred_or_gt: str
            set to 'gt' (it is assumed these are ground truth instances)

        filepath: Path object
            path to file described by annotations.

        mask_format: str
            Assumed to be 'bitmask' for model predictions.

        dataset_class: str
            string describing if the image was in the training, validation, test set.

        instances: detectron2.structures.Instances object
            contains information about class label, bbox, and segmentation mask for each instance.
            Also assigns random colors for each instance for visualization.

        """
        self.pred_or_gt = 'pred'
        self.mask_format = 'bitmask'  # model outs assumed to be RLE bitmasks

        self.filepath = outs['file_name']
        ## if dataset name contains '_', assume that last portion of name indicates dataset class (ie powder_Training)
        split = outs['dataset'].split('_')
        if len(split) > 1:
            self.dataset_class = outs['dataset'].split('_')[-1]  # Training, Validation, etc
        else:
            self.dataset_class = outs['dataset']

        instances_pred = outs['pred']['instances']

        instances = Instances(instances_pred.image_size,
                              **{'masks': structures.RLEMasks(instances_pred.pred_masks),
                                 'boxes': instances_pred.pred_boxes,
                                 'class_idx': instances_pred.pred_classes,
                                 'scores': instances_pred.scores})
        self.instances = instances
        self.instances.colors = visualize.random_colors(len(self.instances), self.randomstate)

        if not inplace:
            return self
        return

    # TODO add in_place argument for in place modification of self.instances?
    def filter_mask_size(self, min_thresh=100, max_thresh=100000, to_rle=False):
        """
        Remove instances with mask areas outside of the interval (min_thresh, max_thresh.)

        Useful for removing small instances (ie 1 or even 0 pixels in segmentation mask) or
        abnormally large outliers (ie many instances combined in a giant blob.) Note that
        this does not modify the InstanceSet in place and returns an Instances object.

        Parameters
        -----------
        min_thresh, max_thresh: int, float or None
            only instances with mask areas greater than min thresh and smaller than max_thresh are kept.
            If either threshold is None, it is not applied (ie if both min_thresh and max_thresh are None
            then all masks are kept.)

        to_rle: bool
            if True, masks are converted to RLE before filtering. The inlier masks will be returned as RLE.
            Otherwise, mask format is preserved.

        Returns
        ----------
        instances_filtered: detectron2.structures.Instances object
            Instances object only containing instances with mask areas in the threshold range.

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

    def compute_rprops(self, keys=None, return_df=False):
        """
        Computes region properties of segmentation masks (ie perimeter, convex area, etc) in self.instances.


        Applies skimage.measure.regionprops_table to self.instances.masks for analysis. Allows for convenient
        measurements of many properties of the segmentation masks. A list of available parameters is available in
        the skimage documentation (see link below.)

        Parameters
        ------------
        keys: list(str) or None
            Properties to measure. Passed to skimage.measure.regionprops_table().

        return_df: bool
            if True, returns the region properties as a pandas dataframe

        Returns
        ------------
        rprops (optional): DataFrame object
            Pandas dataframe containing region properties. Only returned if input argument return_df=True

        Attributes
        -------------
        rprops: DataFrame object
            dataframe of region properties

        See Also
        ---------------
        `skimage.measure.regionprops_table <https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops_table>`_

        """

        if keys is None:  # use default values
            keys = ['area', 'equivalent_diameter', 'major_axis_length', 'perimeter', 'solidity', 'orientation']
        # TODO do this with multiprocessing or joblib.parallel to speed up
        rprops = [skimage.measure.regionprops_table(masks_to_bitmask_array(mask, self.instances.image_size).squeeze()
                                                    .astype(np.int), properties=keys) for mask in self.instances.masks]
        df = pd.DataFrame(rprops)
        df['class_idx'] = self.instances.class_idx
        self.rprops = df

        if return_df:
            return self.rprops

    def copy(self):
        """
        Returns a copy of the InstanceSet object.

        Returns a deep copy (ie everything is copied in memory, does not create pointers to the
        original class instance.)

        Parameters
        ----------
        None

        Returns
        -------
        self: InstanceSet object
            returns a copy of InstanceSet.

        """
        return copy.deepcopy(self)

#  todo depreciated??
def mask_areas(masks):
    """
    Computes area in pixels of each mask in masks.

    Mask areas are computed from counting pixels (bitmasks) or the shoelace formula
    (polygon masks.)

    Parameters
    ----------
    masks: bitmask, polygonmask, or ndarray
        Masks for which the areas will be calculated.

    Returns
    ----------
    areas: ndarray
        n_mask element array where each element contains the Farea of the corresponding mask in masks


    """

    masktype = type(masks)

    if masktype == np.ndarray:
        # masks are already expanded, compute area directly
        return masks.sum(axis=(1, 2), dtype=np.uint)

    elif masktype == PolygonMasks:
        # polygon masks given as array of coordinates [x0,y0,x1,y1,...xn,yn]
        area = np.asarray([_shoelace_area(coords[0][::2], coords[0][1::2]) for coords in masks.polygons])
        return area

    elif masktype == list and type(masks[0]) == dict:  # RLE encoded masks
        return RLE.area(masks)

    elif masktype == RLEMasks:
        return RLE.area(masks.rle)

    elif masktype == Instances:
        return mask_areas(masks.masks)

    elif masktype == InstanceSet:
        return mask_areas(masks.instances)

    elif masktype == list:  # assumed to be list of objects containing masks
        return [mask_areas(x) for x in masks]

    else:
        raise NotImplementedError('Not implemented for type {}'.format(masktype))


def _shoelace_area(x, y):
    """
    Computes area of simple polygon from polygon coordinates.

    Parameters
    -----------
    x, y: ndarray
        n-element array containing x and y coordinates of the vertices of the polygon

    Returns
    --------
    area: float
        mask area in pixels


    Notes
    --------------
    `Shoelace formula <https://en.wikipedia.org/wiki/Shoelace_formula>`_

    `Implementation <https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates>`_

    """

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

# todo depreciated??
def boxes_to_array(boxes):
    """
    Helper function to convert any type of object representing bounding boxes to a n_box * 4 numpy array

    Parameters
    ---------------
    boxes: list, array, or detectron2.structures.Boxes object
        contains the bounding boxes which will be converted

    Returns
    ---------
    box_array: ndarray
        n x 4 array of bounding box coordinates

    """

    dtype = type(boxes)

    if dtype == np.ndarray:
        return boxes

    elif dtype == list:
        assert len(boxes[0]) == 4
        return np.asarray(boxes)

    elif dtype == Boxes:
        return boxes.tensor.to('cpu').numpy()


# TODO depreciated?
def masks_to_rle(masks, size=None):
    """
    Converts various objects to RLE masks

    Parameters
    ----------
    masks: list, RLEMasks, or PolygonMasks object
        masks to convert to RLE.

    size: tuple(int, int) or None
        Only needed to be specified for polygon masks.
        Tuple containing the image height and width in pixels needed to convert polygon coordinates
        into full segmentation masks.

    Returns
    --------
    rle_masks: list(dic)
        list of dictionaries with RLE encodings for each mask

    """
    dtype = type(masks)
    if dtype == list:
        if type(masks[0]) == dict:
            # assumed to already be in RLE format
            return masks
        elif type(masks[0]) == list:
            raise(NotImplementedError('):'))

    if dtype == RLEMasks:
        rle = masks.rle
        return rle

    elif dtype == PolygonMasks:
        assert size is not None
        rle = [RLE.frPyObjects(p, *size)[0] for p in masks.polygons]
        return rle
        # for mask in masks:
        #     cords = mask.polygon
        #     polygon2mask
    elif dtype == InstanceSet:
        return masks_to_rle(masks.instances.masks, masks.instances.image_size)

    elif dtype == Instances:
        return masks_to_rle(masks.masks, masks.image_size)

    else:
        raise NotImplementedError('cannot convert mask type {} to RLE'
                                  .format(masks))


def _poly2mask(masks, size):
    """
    Helper function to convert polygon masks since they can be lists, arrays, or PolygonMask instances
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
    Converts various mask types to an n_mask x r x c boolean array of masks.

    Parameters
    -----------
    masks: list,ndarray, RLEMasks, or PolygonMasks object
        masks to convert to RLE.

    size: tuple(int, int) or None
        Only needed to be specified for polygon masks.
        Tuple containing the image height and width in pixels needed to convert polygon coordinates
        into full segmentation masks.

    Returns
    ----------
    mask_array: ndarray
        n_mask x r x c boolean array of pixel values

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

    elif dtype == InstanceSet:
        return masks_to_bitmask_array(masks.instances.masks, masks.instances.image_size)

    elif dtype == Instances:
        return masks_to_bitmask_array(masks.masks, masks.image_size)

    else:
        raise NotImplementedError