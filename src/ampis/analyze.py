import numpy as np
import pathlib
import pycocotools.mask as RLE

from detectron2.structures import Instances

from . import data_utils
from .structures import instance_set, RLEMasks, masks_to_rle, masks_to_bitmask_array


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
        gt_bbox = data_utils.extract_bboxes(gt_masks)
    else: # TODO handle non-integer bboxes (with rounding and limits at edge of images)
        gt_bbox = gt_bbox.astype(np.int) if gt_bbox.dtype is not np.int else gt_bbox
    if pred_bbox is None:
        pred_bbox = data_utils.extract_bboxes(pred_masks)
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


def _piecewise_iou(a, b, interval=80):
    """
    helper function for computing iou since
    this function apparently can only handle a max of 80 x 80
    instances
    Args:
        a, b: full list (ie can be more than 80 elements)
        interval: int, max length of list that can be input to function

    Returns:

    """
    imax = len(a)
    jmax = len(b)
    target = np.zeros((imax, jmax))

    n_seg_a = imax // interval + int(imax % interval > 0)
    n_seg_b = jmax // interval + int(jmax % interval > 0)



    for i in range(n_seg_a):
        i_int = interval * np.array([i, i+1], np.int)
        for j in range(n_seg_b):
            j_int = interval * np.array([j, j+1], np.int)
            a_args = a[i_int[0]:i_int[1]]
            b_args = b[j_int[0]:j_int[1]]

            target[i_int[0]:i_int[1], j_int[0]:j_int[1]] = RLE.iou(a_args, b_args, [False for _ in range(len(b_args))])

    return target


def _piecewise_rle_match(gt, pred, iou_thresh=0.5, interval=80):
    """
    helper function for computing iou since
    this function apparently can only handle a max of 80 x 80
    instances
    Args:
        gt, pred: full list of RLE instances (ie can be more than 80 elements)
        iou_thresh: float between 0 and 1, minimim IOU must be above this for instances to match
        interval: int, max length of list that can be input to function

    Returns:

    """
    jmax = len(pred)


    tp = []
    fn = []
    iou = []
    pred_matched = np.zeros(len(pred), np.bool)
    n_seg_pred = jmax // interval + int(jmax % interval > 0)
    for gt_idx, gt_mask in enumerate(gt):
        IOU_max = 0.
        IOU_argmax = -1
        for j in range(n_seg_pred):
            j0, j1 = interval * np.array([j, j+1], np.int)
            pred_args = pred[j0:j1]

            iou_scores_ = RLE.iou([gt_mask], pred_args, [False for _ in range(j1-j0)])[0]
            iou_amax_j = np.argmax(iou_scores_)
            iou_max_j = iou_scores_[iou_amax_j] # max is computed with index relative to subset of data


            if iou_max_j > IOU_max:
                IOU_max = iou_max_j
                IOU_argmax = iou_amax_j + j0  # offset by j0 so argmax is indexed to predictions instead of subset


        if IOU_max > iou_thresh:
            tp.append([gt_idx, IOU_argmax])
            iou.append(IOU_max)
            pred_matched[IOU_argmax] = True
        else:
            fn.append(gt_idx)

    fp = np.array([x for x, matched in enumerate(pred_matched) if not matched], np.int)

    results = {'tp': np.asarray(tp, np.int),
               'fn': np.asarray(fn, np.int),
               'fp': np.asarray(fp, np.int),
               'iou': np.asarray(iou)}

    return results



def rle_instance_matcher(gt, pred, iou_thresh=0.5, size=None):
    """
    Performs instance matching (single class) to determine true positives,
    false positives, and false negative instance predictions.

    Instances are matched on the basis of Intersection over Union (IOU,)
    or ratio of areas of overlap between 2 masks to the total area occupied by both masks.
    IOU (A,B) = sum(A & B)/sum(A | B)

    Args:
        gt, pred: ampis.structures.RLEMasks or detectron2.structures.polygonmasks
        size: None if bitmasks or RLEmasks, tuple(r, c) image height, width in pixels if gt or pred is polygonmasks

    Returns:
        results: dictionary with the following structure:
        {
        'tp': n_match x 2 array of indices of matches. The first element corresponds to the index of the gt instance for match i. The second element corresponds to the index of the pred index for match i.
        'gt_fn': n_fn element array where each element is a ground truth instance that was not matched (false negative)
        'pred_fp': n_fp element array where each element is a predicted instance that was not matched (false positive)
        'IOU_match': n_match element array of IOU scores for each match.
        }


    """

    # convert masks if needed
    gt = masks_to_rle(gt, size)
    pred = masks_to_rle(pred, size)
    return _piecewise_rle_match(gt, pred, iou_thresh)


def mask_match_stats(gt, pred, IOU_thresh=0.5, size=None):
    """
        Computes match and mask statistics for a give pair of masks (of the same class.) Match statistics describe the number of instances that were correctly matched with IOU above the threshold. Mask statistics describe how well the matched masks agree with each other. For each set of tests, the precision and recall are reported. 
    TODO update docs
    Inputs:
    :param gt: r x c x n_mask boolean array of ground truth masks
    :param pred: r x c x n_mask boolean array of predicted masks
    :param IOU_thresh: IOU threshold for matching (see fast_instance_match())
    :param size: size passed to masks_to_rle. Can be None for rle masks or bitmasks. For polygonmasks, must be
                tuple (r, c) for the image height and width in pixels, respectively.
    
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
    gtmasks = masks_to_rle(gt, size)
    predmasks = masks_to_rle(pred, size)


    match_results_ = rle_instance_matcher(gtmasks, predmasks, iou_thresh=IOU_thresh, size=size)
    matches_ = np.asarray(match_results_['tp'])
    TP_match_ = len(matches_) #  true positive
    FN_match_ = len(match_results_['fn']) #  false negative
    FP_match_ = len(match_results_['fp']) #  false positive

    match_precision = TP_match_ / (TP_match_ + FP_match_)
    match_recall = TP_match_ / (TP_match_ + FN_match_)

    gtmasks_tp = [gtmasks[i[0]] for i in matches_]
    predmasks_tp = [predmasks[i[1]] for i in matches_]
    mask_true_positive = np.array([RLE.area(RLE.merge([m1,m2], intersect=True)) for m1, m2 in zip(gtmasks_tp, predmasks_tp)],
                            np.int)
    tp_gt_area = np.array([RLE.area(m) for m in gtmasks_tp], np.int)
    tp_pred_area = np.array([RLE.area(m) for m in predmasks_tp], np.int)

    mask_false_positive = tp_pred_area - mask_true_positive
    mask_false_negative = tp_gt_area - mask_true_positive

    mask_precision = mask_true_positive /(mask_true_positive + mask_false_positive)
    mask_recall = mask_true_positive /(mask_true_positive + mask_false_negative)

    return {'match_precision': match_precision,
           'match_recall': match_recall,
           'mask_precision': mask_precision,
           'mask_recall': mask_recall}

# TODO move to visualize or rename?
def match_visualizer(gt, pred, match_results=None, colormap=None, TP_gt=False):
    """
    Computes matches between gt and pred masks. Returns the masks, boxes, and colors in a format that is convenient for visualizing the match performance number of correctly matched instances.
    inputs:
    TODO update documentation
    :param gt: r x c x n_mask_gt boolean array of ground truth masks
    :param pred: r x c x n_mask_pred boolean array of predicted masks
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
    if match_results is None:
        match_results = rle_instance_matcher(gt, pred)

    return_colormap = colormap is None

    size = gt.instances.image_size
    gt_masks = masks_to_rle(gt.instances.masks, size)
    pred_masks = masks_to_rle(pred.instances.masks, size)

    gt_bbox = gt.instances.boxes if type(gt.instances.boxes) == np.ndarray else gt.instances.boxes.tensor.numpy()
    pred_bbox = pred.instances.boxes if type(pred.instances.boxes) == np.ndarray \
        else pred.instances.boxes.tensor.numpy()

    #TODO pick prettier values!
    if colormap is None:  # default values
        colormap = {'TP': np.asarray([1,0,0],np.float),
                    'FP': np.asarray([0,1,0],np.float),
                    'FN': np.asarray([0,0,1], np.float)}
    
    if match_results is None:  # default
        match_results = fast_instance_match(gt_masks, pred_masks)
    
    if TP_gt:
        TP_idx = match_results['tp'][:, 0]
        TP_masks = [gt_masks[i] for i in TP_idx]
        TP_bbox = gt_bbox[TP_idx]
    else:
        TP_idx = match_results['tp'][:, 1]
        TP_masks = [pred_masks[i] for i in TP_idx]
        TP_bbox = pred_bbox[TP_idx]

    TP_colors = np.tile(colormap['TP'], (len(TP_masks), 1))
    
    FP_idx = match_results['fp']
    FP_masks = [pred_masks[i] for i in FP_idx]
    FP_bbox = pred_bbox[FP_idx]
    FP_colors = np.tile(colormap['FP'], (len(FP_masks), 1))
    
    FN_idx = match_results['fn']
    FN_masks = [gt_masks[i] for i in FN_idx]
    FN_bbox = gt_bbox[FN_idx]
    FN_colors = np.tile(colormap['FN'], (len(FN_masks), 1))
    
    masks = RLEMasks(TP_masks +  FP_masks + FN_masks)
    bbox = np.concatenate((TP_bbox, FP_bbox, FN_bbox), axis=0)
    colors = np.concatenate((TP_colors, FP_colors, FN_colors), axis=0)

    i = instance_set()
    i.instances = Instances(image_size=masks.masks[0]['size'], **{'masks': masks, 'boxes': bbox, 'colors': colors})

    if return_colormap:
        return i, colormap
    return i

# TODO move to visualize or rename?
def mask_visualizer(gt_masks, pred_masks, match_results=None, size=None):
    """
    Computes matches between gt and pred masks. Returns a mask image where each pixel describes if the pixel in the masks is a true positive false positive, false negative, or a combinaton of these.
    inputs:
    :param gt_masks: r x c x n_mask_gt boolean array of ground truth masks
    :param pred_masks: r x c x n_mask_pred boolean array of predicted masks
    :param match_results: dictionary of match indices (see fast_instance_match()) with keys 'gt_tp' for match indices (ground truth and predicted), 'pred_fp' for false positive predicted indices, and 'gt_fn' for ground truth false negative indices.
    :param size: None if gt and pred are RLE masks, otherwise tuple (r, c) image height, image width
    If match_results is None, they will be computed using fast_instance_match().
    
    returns:
    mask_img: r x c x 3 RGB image that maps mask results to original image.
    mapper: dictionary mapping colors in mask_img to descriptions 
    """
    # defaults
    if match_results is None:
        match_results = rle_instance_matcher(gt_masks, pred_masks, size=size)
    gt_masks = masks_to_bitmask_array(gt_masks, size)
    pred_masks = masks_to_bitmask_array(pred_masks, size)

    tp_idx = match_results['tp']
    matched_gt = gt_masks[tp_idx[:, 0]]
    matched_pred = pred_masks[tp_idx[:, 1]]
    
    TP_mask_ = np.logical_and(matched_gt, matched_pred,)  # true positive
    FN_mask_ =  np.logical_and(matched_gt, np.logical_not(matched_pred))  # false negative
    FP_mask_ =  np.logical_and(np.logical_not(matched_gt), matched_pred)  # false positive

    project = lambda masks: np.logical_or.reduce(masks, axis=0)

    
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

    masks = np.asfortranarray(masks)#.transpose((0,1,2)))
    masks = RLE.encode(masks)
    masks = RLEMasks(masks)

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
    

    i = instance_set()
    i.instances = Instances(image_size=masks.masks[0]['size'], **{'masks': masks, 'colors': colors[0],
                                                                  'boxes': np.zeros((len(masks), 4))})

    return i, colors[1]


def RLE_numpy_encode(mask):
    """
    Encodes RLE with numpy arrays. Can accomodate bitmasks (boolean array)
    or label images (integer array where 0 indicates no mask, and after 0 each integer
    indicates a unique, separate mask)

    Args:
        mask: bitmask or label image

    Returns:
    """
    if mask.dtype == np.bool:
        assert mask.ndim == (2 or 3)
    else:
        assert mask.squeeze().ndim < 3

    r0 = np.reshape(mask, mask.size)
    r1 = np.where(r0[:-1] != r0[1:])[0]
    r2 = r1[1:] - r1[:-1]

    last = np.array([mask.size - r1[-1] - 1])

    if r1[0] != 0:
        first = np.array([0, r1[0] + 1])
    else:
        first = np.array([r1[0] + 1])

    np_RLE = np.concatenate((first, r2, last), axis=0)
    values = r0[r1]
    values = values[values != 0]
    results = {'size': mask.shape,  # TODO this doesn't work for bitmasks
               'encoding': np_RLE,
               'values': values}  # TODO debug values
