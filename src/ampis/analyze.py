import colorsys
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pycocotools.mask as RLE
import skimage.io 
import skimage.measure

#import mrcnn_utils
from . import evaluate
from . import structures

from detectron2.data.detection_utils import annotations_to_instances
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.structures.masks import PolygonMasks
from detectron2.utils.visualizer import Visualizer

def align_instance_sets(A, B):
    """
    For lists of instance set objects A and B, rearranges
    B to match the order of A. Matching is performed on the
    basis of filenames. Only instance sets from A and B that have
    corresponding filenames are kept.

    Args:
        A: List of instance set objects.
        B: List of instance set objects.

    Returns:

        A_ordered: List of matched instance set objects in A.
        B_ordered: List of matched instance set objects in B.
    """
    Bdict = {pathlib.Path(item.filepath).name: item for item in B}

    A_ordered = []
    B_ordered = []
    for item in A:
        x = Bdict.get(pathlib.Path(item.filepath).name, None)
        if x is not None:
            A_ordered.append(item)
            B_ordered.append(x)
    return A_ordered, B_ordered


class instance_set(object):
    """
    Simple way to organize a set of instances for a single image to ensure
    that formatting is consistent.
    """
    def __init__(self, mask_format=None, bbox_mode=None, file_path=None, annotations=None, instances=None, img=None,
                 dataset_class=None, pred_or_gt=None, HFW=None, randomstate=None):
        
        self.mask_format = mask_format # 'polygon' or 'bitmask'
        self.bbox_mode = bbox_mode  # from detectron2.structures.BoxMode
        self.img = img  # image r x c x 3
        self.filepath = file_path  # file name or path of image
        self.dataset_class = dataset_class  # 'Training', 'Validation', 'Test', etc
        self.pred_or_gt = pred_or_gt  # 'gt' for ground truth, 'pred' for model prediction
        self.HFW = HFW  # Horizontal Field Width of image. Can be float or string with value and units. ##TODO automatically read this
        self.rprops=None  # region props, placeholder for self.compute_regionprops()
        self.instances = instances
        self.annotations = annotations
        if randomstate is None:  # random state used for color assignment during visualization
            randomstate=np.random.randint(2**32-1)
        self.randomstate = randomstate
        self.colors = None
    ##TODO __repr__?

    
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
        #instances_gt = annotations_to_instances(ddict['annotations'], image_size, self.mask_format)

        class_idx = np.asarray([anno['category_id'] for anno in ddict['annotations']], np.int)
        bbox = Boxes([anno['bbox'] for anno in ddict['annotations']])
        segs = [anno['segmentation'] for anno in ddict['annotations']]
        segtype = type(segs[0])
        if segtype == dict:
            # RLE encoded mask
            masks = structures.RLEMasks(segs)

        elif segtype == np.ndarray:
            if anno0.dtype == np.bool:
                #  bitmask
                masks = BitMasks(np.stack(segs))

        else:
            # list of (list or array) of coords in format [x0,y0,x1,y1,...xn,yn]
            masks = PolygonMasks(segs)


        instances = Instances(image_size, **{'masks': masks,
                                             'boxes': bbox,
                                             'class_idx': class_idx})
        self.instances = instances
        self.instances.colors = random_colors(len(instances), self.randomstate)




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
        self.dataset_type = outs['dataset'].split('_')[1]  # Training, Validation, etc

        instances_pred = outs['pred']['instances']

        instances = Instances(instances_pred.image_size,
                              **{'masks': instances_pred.pred_masks,
                                 'boxes': instances_pred.pred_boxes,
                                 'class_idx': instances_pred.pred_classes,
                                 'scores': instances_pred.scores})
        self.instances = instances
        self.instances.colors = random_colors(len(self.instances), self.randomstate)

        if return_:
            return self

    def filter_mask_size(self, min_thresh=100, max_thresh=100000):
        """
        Remove instances with mask areas outside of the interval (min_thresh, max_thresh.)

        inputs:
        :param instances:- instances object
        :param min_thresh: int- minimum mask size threshold in pixels, default 100, or None to not use this criteria
        :param max_thresh: int- maximum mask size threshold in pixels, default 100000, or None to not use this criteria

        returns:
            * instances_filtered-instances object which only includes instances within given size thresholds
        """

        masks = self.instances.masks
        # determine which instances contain inlier masks
        areas = _mask_areas(masks)

        if min_thresh is None:
            inlier_min = np.ones(area.shape, np.bool)
        else:
            inlier_min = areas > min_thresh
        if max_thresh is None:
            inlier_max = np.ones(area.shape, np.bool)
        else:
            inlier_max = areas < max_thresh

        inliers_bool = np.logical_and(inlier_min, inlier_max)

        new_instance_fields = {key: value[inliers_bool] for key, value
                               in self.instances._fields.items()}

        instances_filtered = Instances(self.instances.image_size,
                                       **new_instance_fields)

        return instances_filtered

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
        
        if keys is None: # use default values
            keys = ['area', 'equivalent_diameter','major_axis_length', 'perimeter','solidity','orientation']
        rprops = [skimage.measure.regionprops_table(mask.astype(np.int),properties=keys)  
                                                    for mask in np.transpose(self.masks, (2,0,1))]
        df = pd.DataFrame(rprops)
        df['class_idx'] = self.class_idx
        self.rprops = df
        
        if return_df:
            return self.rprops
        
    
    def copy(self):
        """
        returns a copy of the instance_set object
        """
        return copy.deepcopy(self)


def random_colors(N, seed, bright=True):  # controls randomstate for plotting consistentcy
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """

    rs = np.random.RandomState(seed=seed)

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    rs.shuffle(colors)
    colors = np.asarray(colors)
    return colors


def quick_display_instances(img, metadata, iset, show_class_idx=False, show_scores=False,  ax=None, colors=None):

    # by default, colors will be extracted from instances. Otherwise, custom colors can be supplied.
    if colors is None:
        colors = iset.instances.colors

    V = Visualizer(img, metadata, scale=1)

    if show_class_idx:
        if show_scores:
            extra = ': '
        else:
            extra = ''
        class_idx = ['{}{}'.format(metadata['thing_classes'][idx], extra) for idx in iset.instances.class_idx]
    else:
        class_idx = ['' for x in iset.instances.class_idx]

    if show_scores:
        scores = ['{:.3f}'.format(x) for x in iset.instances.scores]
    else:
        scores = ['' for x in iset.instances.class_idx] # gt do not have scores,
        # so must iterate through a field that exists

    labels = ['{}{}'.format(idx, score) for idx, score in zip(class_idx, scores)]

    masktype = type(iset.instances.masks)
    if masktype == structures.RLEMasks:
        masks = iset.instances.masks.masks
    else:
        masks = iset.instances.masks

    vis = V.overlay_instances(boxes=iset.instances.boxes, masks=masks, labels=labels,
                              assigned_colors=colors)
    vis_img = vis.get_image()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7), dpi=150)
        ax.imshow(vis_img)
        ax.axis('off')
        plt.show()

    else:
        ax.imshow(vis_img)
        ax.axis('off')



def project_masks(masks):
    """
    Project an array of boolean masks (r x c x n_masks) into a single integer label image (r x c.) 
    This is similar to the output of skimage.label() when instance segmentation is trivial (ie no overlap.)
    Works when instance masks do not have significant overlap.
    inputs:
    :param masks: r x c x n_masks array of boolean masks
    returns:
    labels: r x c array where each pixel has an integer label corresponding to the
             index of the mask in masks. Background pixels (no masks) are assigned values of -1
    """
    r, c, n = masks.shape
    
    labels = np.zeros((r,c), np.int) - 1 # initialize and set background to -1
    for i, mask in enumerate(np.transpose(masks, (2,0,1))):
        labels[mask] = i
    
    return labels


def IOU(true_mask, predicted_mask):
    """
    Computes Intersection Over Union score for binary segmentation masks true_mask
    and predicted_mask. Note that both masks must have the same shape for them to be compared.
    IOU is defined as (A and B).sum()/(A or B).sum().
    If the union is 0 (both masks are empty), the IOU score is assumed to be 0
    in order to prevent divide by 0 error.
    Note that because intersection and union are symmetric, the function will
    still work correctly if true_mask and predicted_mask are switched with each other.
    :param true_mask: numpy array containing ground truth binary segmentation mask
    :param predicted mask: numpy array containing predicted binary segmentation mask
    """
    assert true_mask.shape == predicted_mask.shape  # masks must be same shape for comparison
    union = np.logical_or(true_mask, predicted_mask).sum()
    if union == 0:  # both masks are empty
        return 0.
    intersection = np.logical_and(true_mask, predicted_mask).sum()
    return intersection / union


def fast_instance_match(gt_masks, pred_masks, gt_bbox=None, pred_bbox=None, IOU_thresh=0.5):
    """
    instance matching based on projected mask labels (see project_masks() function.)
    Label images are two-dimensional arrays (size r x c) where background pixels are
    -1 and all other pixels have integer values corresponding to their index in the
    original r x c x n_masks arrays used to compute the label images.
    
    Note that gt_masks and pred_masks must contain only instances from a single class. For multi-class instance segmentation, the instances can be divided into subsets based on their class labels. TODO make helper function to split instance.
    
    Instances are matched on the basis of Intersection over Union (IOU,) or ratio of areas of overlap between 2 masks to the total area occupied by both masks.
    IOU (A,B) = sum(A & B)/sum(A | B)
    
    This should be much faster than brute-force matching, where IOU is computed for all
    possible pairs of masks, even ones that are very far from each other with IOU=0.
    inputs:
    :param gt_masks: r x c x n_mask_gt boolean numpy array of of ground truth masks
    :param pred_masks: r x c x n_mask_pred boolean numpy array  of predicted masks
    :param gt_bbox: n_mask_gt x 4 array of ground truth bounding box coordinates for 
                    each mask. If None, bounding boxes are extracted from labels_gt.
    :param gt_bbox: n_mask_gt x 4 array of bounding box coordinates for each predicted 
                    mask. If None, bounding boxes are extracted from labels_gt.
    :param IOU_thresh: float between 0 and 1 (inclusive,) if the max IOU for mask pairs
                       is less than or equal to the threshold, the mask will be considered 
                       to not have a valid match. Values above 0.5 ensure that masks do not 
                       have multiple matches.

    returns:
      :param results: dictionary with the following structure:
        {
        'gt_tp': n_match x 2 array of indices of matches. The first element corresponds to the index of the gt instance for match i. The second element corresponds to the index of the pred index for match i.
        'gt_fn': n_fn element array where each element is a ground truth instance that was not matched (false negative)
        'pred_fp': n_fp element array where each element is a predicted instance that was not matched (false positive)
        'IOU_match': n_match element array of IOU scores for each match.
        }
    """
    ## TODO consider using np.unique[r1:r2,c1:c2,:].max((0,1)) with indexing array instead of projecting instances onto 2d image to handle case of overlapping instances
    n = gt_masks.shape[2] # number of ground truth instances
    
    # get label images for each set of masks 
    #gt_labels = project_masks(gt_masks)
    pred_labels = project_masks(pred_masks)
    
    # get bboxes
    if gt_bbox is None:
        gt_bbox = mrcnn_utils.extract_bboxes(gt_masks)
    else: # TODO handle non-integer bboxes (with rounding and limits at edge of images)
        gt_bbox = gt_bbox.astype(np.int) if gt_bbox.dtype is not np.int else gt_bbox
    if pred_bbox is None:
        pred_bbox = mrcnn_utils.extract_bboxes(pred_masks)
    else:
        pred_bbox = pred_bbox.astype(np.int) if pred_bbox.dtype is not np.int else pred_bbox
    
    gt_tp = []  # true positive  [[gt_idx, pred_idx]]
    gt_fn = []  # false negatives [gt_idx]
    IOU_match = []
    
    pred_matched = np.zeros(pred_masks.shape[2], np.bool)  # values will be set to True when 
    
    for gt_idx, (mask, box) in enumerate(zip(np.transpose(gt_masks, (2,0,1)), 
                                                     gt_bbox)):
        # find predicted masks in the same neighborhood as
        # the ground truth mask in question
        g_r1, g_c1, g_r2, g_c2 = box
        neighbor_idx = np.unique(pred_labels[g_r1:g_r2, 
                                          g_c1:g_c2])
        if neighbor_idx[0] == -1:
            neighbor_idx = neighbor_idx[1:]
        
        if len(neighbor_idx) > 0:
            mask_candidates = pred_masks[...,neighbor_idx]
            bbox_candidates = pred_bbox[neighbor_idx,...]

            IOU_ = np.zeros(bbox_candidates.shape[0], np.float)

            for i, (pmask, pbox) in enumerate(zip(np.transpose(mask_candidates, (2,0,1)), 
                                                  bbox_candidates)):
                p_r1, p_c1, p_r2, p_c2 = pbox

                # extract the smallest window indices [r1:r2, c1:c2]
                # that fully includes both the gt and predicted masks
                r1 = min(g_r1, p_r1)
                c1 = min(g_c1, p_c1)
                r2 = max(g_r2, p_r2)
                c2 = max(g_c2, p_c2)

                IOU_[i] = IOU(*[x[r1:r2,c1:c2] for x in [mask, pmask]])

            IOU_amax = IOU_.argmax()
            IOU_max = IOU_[IOU_amax]

            if IOU_max > IOU_thresh:
                gt_tp.append([gt_idx, neighbor_idx[IOU_amax]])
                pred_matched[neighbor_idx[IOU_amax]] = True
                IOU_match.append(IOU_max)
            else:
                gt_fn.append(gt_idx)
        
    pred_fp = np.asarray([x for x, matched in enumerate(pred_matched) if not matched])
    results = {'gt_tp': np.asarray(gt_tp, np.int),
              'gt_fn': np.asarray(gt_fn, np.int),
              'pred_fp': pred_fp,
              'IOU_match': IOU_match}
    return results


def expand_masks(masks):
    """
    decompresses masks to n_mask x r x c boolean array for characterization and computation
    Args:
        masks: masks

    Returns:
        bitmasks- n_mask x r x c boolean array
    """

    dtype = type(masks)

    if dtype == np.ndarray: # masks are already bitmasks
        bitmasks = masks

    elif dtype == list:
        if type(masks[0]) == dict:  # RLE mask
            pass # bitmasks =

    return bitmasks


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


def _mask_areas(masks):
    """
    Computes area in pixels of each mask in masks
    Args:
        masks: bitmask, polygonmask, or array containing masks

    Returns: n_mask element array of mask areas

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
        return np.asarray([_shoelace_area(coords[0][::2], coords[0][1::2]) for coords in masks.polygons])
    elif masktype == list and type(masks[0]) == dict:
        # RLE encoded masks
        return RLE.area(masks)
    elif masktype == structures.RLEMasks:
        return RLE.area(masks.masks)




    
def mask_match_stats(gt, pred, IOU_thresh=0.5):
    """
        Computes match and mask statistics for a give pair of masks (of the same class.) Match statistics describe the number of instances that were correctly matched with IOU above the threshold. Mask statistics describe how well the matched masks agree with each other. For each set of tests, the precision and recall are reported. 
    
    Inputs:
    :param gt: r x c x n_mask boolean array of ground truth masks
    :param pred: r x c x n_mask boolean array of predicted masks
    :param IOU_thresh: IOU threshold for matching (see fast_instance_match())
    
    Returns:
    output: dictionary with the following key:value pairs:
      'match_precision': float between 0 and 1, match precision for instances 
      'match_recall': float between 0 and 1, match recall for instances
      'mask_precision': n_match element array containing match precision for
                        each matched pair of instances
      'mask_recall': n_match element array containing match recall for
                     each matched pair of instances
    """
    ## match scoring
    match_results_ = fast_instance_match(gt, pred, IOU_thresh=IOU_thresh)
    matches_ = np.asarray(match_results_['gt_tp'])
    TP_match_ = len(matches_) #  true positive
    FN_match_ = len(match_results_['gt_fn']) #  false negative
    FP_match_ = len(match_results_['pred_fp']) #  false positive

    match_precision = TP_match_ / (TP_match_ + FP_match_)
    match_recall = TP_match_ / (TP_match_ + FN_match_)

    ## mask scoring
    
    # only include masks that were correctly matched

    matched_masks_gt_ = gt[:,:,matches_[:,0]]
    matched_masks_pred_ = pred[:,:,matches_[:,1]]
    
    TP_mask_ = np.logical_and(matched_masks_gt_,
                             matched_masks_pred_,).sum((0,1))  # true positive
    FN_mask_ =  np.logical_and(matched_masks_gt_,
                              np.logical_not(matched_masks_pred_)).sum((0,1))  # false negative
    FP_mask_ =  np.logical_and(np.logical_not(matched_masks_gt_),
                               matched_masks_pred_).sum((0,1))  # false positive
    
    mask_precision = TP_mask_ / (TP_mask_ + FP_mask_)
    mask_recall = TP_mask_ / (TP_mask_ + FN_mask_)
    
    return {'match_precision': match_precision,
           'match_recall': match_recall,
           'mask_precision': mask_precision,
           'mask_recall': mask_recall}


def match_visualizer(gt_masks, gt_bbox, pred_masks, pred_bbox, colormap=None, match_results=None, TP_gt=False):
    """
    Computes matches between gt and pred masks. Returns the masks, boxes, and colors in a format that is convenient for visualizing the match performance number of correctly matched instances.
    inputs:
    :param gt_masks: r x c x n_mask_gt boolean array of ground truth masks
    :param gt_bbox: n_mask_gt x 4 array of bbox coordinates for each ground truth mask
    :param pred_masks: r x c x n_mask_pred boolean array of predicted masks
    :param pred_bbox: n_mask_pred x 4 array of bbox coordinates for each predicted mask
    :param colormap: dictionary with keys 'TP', 'FP', 'FN'. The value corresponding to each key is a 1x3 float array of RGB color values.
    If colormap is None, default colors will be used.
    :param match_results: dictionary of match indices (see fast_instance_match()) with keys 'gt_tp' for match indices (ground truth and predicted), 'pred_fp' for false positive predicted indices, and 'gt_fn' for ground truth false negative indices.
    :param TP_gt: bool. If True, true positives will be displayed from ground truth instances. If False, true positives will be displayed from predicted instances.
    If match_results is None, they will be computed using fast_instance_match().
    
    returns:
    masks: r x c x n_mask_match boolean array containing true positive and false positive predicted masks, as well as false negative ground truth masks
    bbox: n_mask_match x 4 array of bbox coordinates for each mask in masks
    colors: n_mask_match x 3 array of RGB colors for each mask. Colors can be used to visually distinguish true positives, false positives, and false negatives. 
    colormap: only returned if colormap=None. Returns the default colormap.
    """
    
    return_colormap = colormap == None
    
    #TODO pick prettier values!
    if colormap is None:  # default values
        colormap = {'TP': np.asarray([1,0,0],np.float),
                    'FP': np.asarray([0,1,0],np.float),
                    'FN': np.asarray([0,0,1], np.float)}
    
    if match_results is None:  # default
        match_results = fast_instance_match(gt_masks, pred_masks)
    
    if TP_gt:
        TP_idx = match_results['gt_tp'][:,0]
        TP_masks = gt_masks[:,:,TP_idx]
        TP_bbox = gt_bbox[TP_idx]
    else:
        TP_idx = match_results['gt_tp'][:,1]
        TP_masks = pred_masks[:,:,TP_idx]
        TP_bbox = pred_bbox[TP_idx]

    TP_colors = np.tile(colormap['TP'], (TP_masks.shape[2],1))
    
    FP_idx = match_results['pred_fp']
    FP_masks = pred_masks[:,:,FP_idx]
    FP_bbox = pred_bbox[FP_idx]
    FP_colors = np.tile(colormap['FP'], (FP_masks.shape[2],1))
    
    FN_idx = match_results['gt_fn']
    FN_masks = gt_masks[:,:,FN_idx]
    FN_bbox = gt_bbox[FN_idx]
    FN_colors = np.tile(colormap['FN'], (FN_masks.shape[2],1))
    
    masks = np.concatenate((TP_masks, FP_masks, FN_masks), axis=2)
    bbox = np.concatenate((TP_bbox, FP_bbox, FN_bbox), axis=0)
    colors = np.concatenate((TP_colors, FP_colors, FN_colors), axis=0)
    
    outs = [masks, bbox, colors]
    if return_colormap:
        outs.append(colormap)
    
    return tuple(outs)


def mask_visualizer(gt_masks, pred_masks, match_results=None):
    """
    Computes matches between gt and pred masks. Returns a mask image where each pixel describes if the pixel in the masks is a true positive false positive, false negative, or a combinaton of these.
    inputs:
    :param gt_masks: r x c x n_mask_gt boolean array of ground truth masks
    :param pred_masks: r x c x n_mask_pred boolean array of predicted masks
    :param match_results: dictionary of match indices (see fast_instance_match()) with keys 'gt_tp' for match indices (ground truth and predicted), 'pred_fp' for false positive predicted indices, and 'gt_fn' for ground truth false negative indices.
    If match_results is None, they will be computed using fast_instance_match().
    
    returns:
    mask_img: r x c x 3 RGB image that maps mask results to original image.
    mapper: dictionary mapping colors in mask_img to descriptions 
    """
    # defaults
    if match_results is None:
        match_results = fast_instance_match(gt_masks, pred_masks)
    
    tp_idx = match_results['gt_tp']
    matched_gt = gt_masks[:,:,tp_idx[:,0]]
    matched_pred = pred_masks[:,:,tp_idx[:,1]]
    
    TP_mask_ = np.logical_and(matched_gt,
                             matched_pred,)  # true positive
    FN_mask_ =  np.logical_and(matched_gt,
                              np.logical_not(matched_pred))  # false negative
    FP_mask_ =  np.logical_and(np.logical_not(matched_gt),
                               matched_pred)  # false positive
    
    
    project = lambda masks: np.logical_or.reduce(masks, axis=2)
    
    TP_reduced = project(TP_mask_).astype(np.uint)
    FN_reduced = project(FN_mask_).astype(np.uint) * 2
    FP_reduced = project(FP_mask_).astype(np.uint) * 4
    
    # Code | TP   FN   FP 
    #------+-------------
    # 0    |  F    F   F
    # 1    |  T    F   F
    # 2    |  F    T   F    
    # 3    |  T    T   F
    # 4    |  F    F   T
    # 5    |  T    F   T
    # 6    |  F    T   T
    # 7    |  T    T   T
    pixel_map = TP_reduced + FN_reduced + FP_reduced
    
    masks = np.zeros((*pixel_map.shape[:2], 7), np.bool)
    for i in range(1,8):
        masks[:,:,i-1] = pixel_map == i
    
    # maps index to colors
    color_mapper = np.array([
       [0.   , 0.   , 0.   ],
       [0.150, 0.   , 0.100],
       [0.286, 1.   , 0.   ],
       [1.   , 0.857, 0.   ],
       [1.   , 0.   , 0.   ],
       [0.   , 0.571, 1.   ],
       [0.   , 1.   , 0.571],
       [0.285, 0.   , 1.   ]])
    
    #mask_img_ravel = color_mapper[pixel_map.ravel(),:]
    
    #mask_img = np.reshape(mask_img_ravel, pixel_map.shape)
    
    colors = [color_mapper[1:], 
              ['Other', 'TP','FN','TP+FN','FP','TP+FP','FN+FP','TP+FN+FP']]
    
    return masks, colors
    
