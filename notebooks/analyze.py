import copy
import json
import numpy as np
import os
import pandas as pd
import pathlib
import skimage.io 
import skimage.measure

import mrcnn_utils
import evaluate


class instance_set(object):
    """
    Simple way to organize a set of instances for a single image to ensure
    that formatting is consistent.
    """
    def __init__(self, masks=None, boxes=None, class_idx=None, class_idx_map=None, scores=None, 
                 img=None, filepath=None, dataset_type=None, pred_or_gt=None, HFW=None,):
        
        self.masks = masks # array of masks n_mask x r x c
        self.boxes = boxes # array of boxes n_mask x 4 
        self.class_idx = class_idx # array of int class indices  n_mask
        self.class_idx_map = class_idx_map # maps int class to string label/descriptor dict
        self.scores = scores # array of confidence scores n_mask 
        self.img = img # image r x c x 3
        self.filepath = filepath # file name or path of image
        self.dataset_type = dataset_type # 'Training', 'Validation', 'Test', etc
        self.pred_or_gt = pred_or_gt # 'gt' for ground truth, 'pred' for model prediction
        self.HFW = HFW  # Horizontal Field Width of image. Can be float or string with value and units. ##TODO automatically read this
        self.rprops=None # region props, placeholder for self.compute_regionprops()
    
    ##TODO __repr__?
    
    
    def read_from_ddict(self, ddict, return_=False):
        """
        Read ground-truth labels from ddict (see get_data_dicts)
        
        inputs:
        :param ddict: list(dic) from get_data_dicts
        :param return_: if True, function will return the instance_set object
        """
        
        self.pred_or_gt = 'gt' # ddict assumed to be ground truth labels from via
        self.filepath = pathlib.Path(ddict['file_name'])
        self.dataset_type = ddict['dataset_class']
        gt_ = extract_gt(ddict)
        self.img = gt_['img']
        self.masks = gt_['masks']
        self.boxes = gt_['bboxes']
        self.class_idx = gt_['class_idx']
        self.class_idx_map = ddict['label_idx_mapper']
        self.HFW = ddict['HFW']
        if return_:
            return self
        return
        
       
    def read_from_numpy_out(self, filename, outs, return_=False):
        """
        Read predicted labels from output of detectron2 predictior
        
        inputs:
        :param filename: filename of prediction (from dictionary) 
        :param outs: predictions for a single image
        :param return_: if True, function will return the instance_set object
        """
        self.filename = filename,
        self.dataset_type = outs[1].split('_')[1] # Training or Validation
        ann = outs[0]
        self.masks = np.transpose(ann['masks'], (1,2,0)) # convert from n_mask x r x c to r x c x n_mask
        self.boxes = ann['boxes'][:,(1,0,3,2)] # order bbox to have (x0,y0, x1, y1)
        self.scores = ann['scores']
        self.class_idx = ann['class']
        
        if return_:
            return self
    
    
    def apply_mask(self, idx_mask):
        """
        Selects a subset of instances based on idx_mask, which can either be a boolean mask or an integer array of indices to select from.
        
        inputs:
        :param idx_mask: n_mask boolean array or n_mask_inlier integer index array array to select instances from.
        """
        
        # masks are formatted differently than other attributes
        # filter must be applied along axis 2
        self.masks = self.masks[:,:,idx_mask]
        
        # other attributes filtered by applying mask along axis 0
        for key in ['boxes', 'class_idx', 'scores',]:
            att = self.__dict__[key]
            if att is not None:
                self.__dict__[key] = att[idx_mask]
    
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
        
        width, height = [int(x) for x in v['file_attributes']
                         ['Size (width, height)'].split(', ')]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["dataset_class"] = v['file_attributes']['Image Class']
        record['HFW'] = v['file_attributes']['HFW']

        annos = v["regions"]
        objs = []
        for anno in annos:
            # not sure why this was here, commenting it out didn't seem to break anything
            #assert not anno["region_attributes"] 
            reg = anno['region_attributes']
            anno = anno["shape_attributes"]
            
            # polygon masks is list of polygon coordinates in format ([x0,y0,x1,y1...xn,yn]) as specified in
            # https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.PolygonMasks
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": 'BoxMode.XYXY_ABS', # boxes are given in absolute coordinates (ie not corner+width+height)
                "segmentation": [poly],
                "category_id": 0,
                "Labels": reg
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    labels_all = set()
    for record in dataset_dicts:
        for ann in record['annotations']:
            labels_all.add(ann['Labels']['Label'])
    label_mapper = {idx: label for idx,label in enumerate(labels_all)}
    for ddict in dataset_dicts:
        ddict['label_idx_mapper'] = label_mapper
    return dataset_dicts


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


# TODO setup 'thing_classes' to read from data-- later, this requires a lot of changes
def extract_gt(ddict):
    """
    Extracts image and ground truth instance segmentation labels from an input ddict.
    :param ddict: dictionary of dataset compatible for detectron 2 (see get_data_dicts() function)
    returns:
    gt: dictionary containing ground truth image and labels.
        gt contains the following keys and corresponding values:
          'img': r x c x 3 RGB image,
          'masks': n_instance x r x c array of boolean instance masks
          'bboxes': n_instance x 4 array of coordinates for each bbox
          'class_idx': n_instance element array of integers corresponding to
                        the class label of each 
    """
    path = os.path.join(ddict['file_name'])
    img = skimage.io.imread(path)
    r, c = img.shape
    img = skimage.color.gray2rgb(img)
    n = len(ddict['annotations'])
    masks = np.zeros((r, c, n), dtype=np.bool)
    bboxes = np.zeros((n,4), dtype=np.float)
    class_idx = np.zeros(n, np.int)
    for i, ann in enumerate(ddict['annotations']):
        # class idx can be read directly
        class_idx[i] = ann['category_id']
        # bbox must be reordered 
        bboxes[i,:] = [ann['bbox'][i] for i in [1,0,3,2]] #  TODO see if order is correct
        
        # masks must be computed
        mask_xycords = ann['segmentation'][0]
        mask = np.asarray(list(zip(mask_xycords[1::2], 
                                   mask_xycords[::2])), np.float)
        masks[:,:,i] = skimage.draw.polygon2mask((r,c), mask)
        
    gt = {'img': img,
              'masks': masks,
              'bboxes': bboxes,
              'class_idx': class_idx}
    return gt


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


def filter_mask_size(masks, min_thresh=100, max_thresh=100000,
                     bbox=None, class_idx=None, scores=None, colors=None,
                     return_mask=False):
    """
    Remove instances with mask areas outside of the interval (min_thresh, max_thresh.)
    
    inputs:
    :param masks: r x c x n_mask boolean array of masks
    :param min_thresh: int- minimum mask size threshold in pixels, default 100
    :param max_thresh: int- maximum mask size threshold in pixels, default 100000
    :param bbox: n_mask x 4 array of bbox coordinates (optional)
    :param class_idx: n_mask element array of class labels (optional)
    :param scores: n_mask element array of scores (optional)
    :param colors: array of colors for visualization (optional)
    :param return_mask: if True, function only returns n_mask element boolean array indicating which instances are inliers

    
    returns:
    outs- tuple containing (masks, bbox(optional), class_idx(optional), scores(optional), colors, and map(optional)) 
    containing inlier instances. Optional outputs are only returned if their corresponding input is not None
    (or True for mapper)
    """
    
    # determine which instances contain inlier masks
    mask_areas = masks.sum((0,1))
    inlier_min = mask_areas > min_thresh
    inlier_max = mask_areas < max_thresh
    inlier_mask = np.logical_and(inlier_min, inlier_max)
    
    new_masks = masks[:,:,inlier_mask]

    # if return_mask, we only want the boolean mask array, not the filtered components
    if return_mask:
        return inlier_mask
    
    outs = [new_masks]
    

    # include optional arguments
    for optional_arg in [bbox, class_idx, scores, colors]:
        if optional_arg is not None:
            outs.append(optional_arg[inlier_mask])


    if len(outs) == 1:
        return outs[0]
    
    else:
        return tuple(outs)

    
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
    
