"""
Provides tools mostly for analyzing and evaluating model predictions after training and inference have
been run. Perhaps most importantly, gives tools for matching ground truth to predicted instances for
the same image, and some methods to quantify the performance of the predictions.
"""
import numpy as np
from pathlib import Path
import pycocotools.mask as rle
import torch

from detectron2.structures import Instances

from .structures import InstanceSet, RLEMasks, masks_to_rle, masks_to_bitmask_array


def align_instance_sets(a, b):
    """

    Reorders lists of instance sets so they are consistent with each other.

    For lists of instance set objects *a* and *b*, rearranges *b* to match the order of *a*.
    Matching is performed on the basis of filenames. Only instance sets from *a* and *b* that have
    corresponding filenames are kept. This is useful for tasks like comparing the performance of
    instance predictions to their ground truth labels for multiple images.

    Parameters
    ----------
        a, b: List of instance set objects.
            Lists that will be matched to each other.

    Returns
    -----------
        a_ordered, b_ordered: List of instance set objects.
            Lists of matched instance set objects with consistent order by filenames.

    """
    # TODO examples in docstring

    bdict = {Path(item.filepath).name: item for item in b}

    a_ordered = []
    b_ordered = []
    for item in a:
        x = bdict.get(Path(item.filepath).name, None)
        if x is not None:
            a_ordered.append(item)
            b_ordered.append(x)
    return a_ordered, b_ordered


def _piecewise_iou(a, b, interval=80):
    """
    helper function for computing iou scores for rle masks *a* and *b*.

    rle.iou() apparently can only handle a max of 80 x 80
    instances. This splits up the inputs so they can be handled by the function,
    computes the IOU scores piece by piece, and then puts them back together to
    the original input sizes.

    Parameters
    ---------
        a, b: list(dic) of rle masks
            rle masks for which iou scores should be computed
        interval: int
            max length of list that can be input to IOU function. Default 80 to match rle.iou().

    Returns
    ---------
        target: ndarray
            len(a) x len(b) array of IOU scores for masks in a and b.

    """

    # TODO example for this function

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

            target[i_int[0]:i_int[1], j_int[0]:j_int[1]] = rle.iou(a_args, b_args, [False for _ in range(len(b_args))])

    return target


def _piecewise_rle_match(gt, pred, iou_thresh=0.5, interval=80):
    """
    helper function for matching rle ground truth (*gt*) and predicted (*pred*) masks on the basis of IOU score.

    rle.iou() can only handle a maximum of 80x80 masks, so this function splits the inputs up to be processed
    in batches, and recombines the results to match the original shape of the inputs.

    Parameters
    -----------
        gt, pred: list(dic)
            full list of RLE instances for ground truth and gpredicted instances to be matched
            to each other (ie can be more than 80 elements.)
        iou_thresh: float
            Threshold between 0 and 1, minimum IOU must be above this for instances to be considered a match.

        interval: int
            interval size for inputs to be split into during computation of IOU scores.

    Returns
    -----------
    results: dic
        dictionary with the following format:
            {'tp': n_match x 2 ndarray of matches. results['tp'][i] contains the index of the gt   

    """
    jmax = len(pred)

    tp = []
    fn = []
    iou = []
    pred_matched = np.zeros(len(pred), np.bool)
    n_seg_pred = jmax // interval + int(jmax % interval > 0)
    for gt_idx, gt_mask in enumerate(gt):
        iou_max = 0.
        iou_argmax = -1
        for j in range(n_seg_pred):
            j0, j1 = interval * np.array([j, j+1], np.int)
            pred_args = pred[j0:j1]

            iou_scores_ = rle.iou([gt_mask], pred_args, [False for _ in range(j1 - j0)])[0]
            iou_amax_j = np.argmax(iou_scores_)
            iou_max_j = iou_scores_[iou_amax_j]  # max is computed with index relative to subset of data

            if iou_max_j > iou_max:
                iou_max = iou_max_j
                iou_argmax = iou_amax_j + j0  # offset by j0 so argmax is indexed to predictions instead of subset

        if iou_max > iou_thresh:
            tp.append([gt_idx, iou_argmax])
            iou.append(iou_max)
            pred_matched[iou_argmax] = True
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
    IOU (A,B) = sum(A & B)/sum(A | B). This function wraps _piecewise_rle_match().

    Parameters
    ------------
    gt, pred: ampis.structures.RLEMasks or detectron2.structures.polygonmasks
        ground truth (gt) or predicted (pred) instances that will be matched.

    iou_thresh: float
        Minimum IOU threshold for a pair of masks to be considered a match.

    size: None or tuple(int, int)
        Not needed if gt and pred are RLEMasks. If either mask is a polygonmasks object,
        size needs to be specified to convert masks to RLE.
        Size is specified as image (height, width) in pixels.

    Returns
    ----------
        results: dictionary with the following structure:
        {
        'tp': n_match x 2 array of indices of matches. The first element corresponds to the index of the gt instance
        for match i. The second element corresponds to the index of the pred index for match i.
        'fn': n_fn element array where each element is a ground truth instance that was not matched (false negative)
        'fp': n_fp element array where each element is a predicted instance that was not matched (false positive)
        'iou': n_match element array of IOU scores for each match.
        }


    """

    # convert masks if needed
    gt = masks_to_rle(gt, size)
    pred = masks_to_rle(pred, size)
    return _piecewise_rle_match(gt, pred, iou_thresh)


def mask_match_scores(gt, pred, iou_thresh=0.5, size=None):
    """
        Computes match and mask statistics for a give pair of sets of masks.

        Masks are matched on the basis of IOU score using rle_instance_matcher(). Then the scores
        are computed on the results.

        Match scores describe the number of instances that were correctly
        matched with IOU above the threshold. True positives are matched masks, false positives
        are unmatched predicted masks, and false negatives are unmatched ground truth masks.
        Mask statistics describe how well the matched masks agree with each other. In each matched pair of masks,
        true positives are pixels included in both the ground truth masks, false positives are pixels only included
        in the predicted mask, and false negatives are pixels only included in the ground truth mask.

        For each set of tests, the precision and recall are reported. Precision is the ratio of true positives
        to true positives + false positives. Recall is the ratio of true positives to true positives + false negatives.


    Parameters
    ________________
    gt, pred: RLEMasks or polygonmasks
        ground truth (gt) or predicted (pred) masks that will be matched to each other to compute the match scores.

    iou_thresh: float
        IOU threshold for matching (see rle_instance_matcher())
    size: None or tuple(int, int)
        size passed to masks_to_rle. Can be None for rle masks or bitmasks. For polygonmasks, must be
                tuple (r, c) for the image height and width in pixels, respectively.
    
    Returns
    ---------------
    output: dictionary
        Contains the following key:value pairs:
          'match_precision': float between 0 and 1, match precision for instances
          'match_recall': float between 0 and 1, match recall for instances
          'mask_precision': n_match element array containing match precision for
                            each matched pair of instances
            'mask_recall': n_match element array containing match recall for
                        each matched pair of instances
            'match_tp' : n_match x 2 array of indices of the ground truth and predicted instances that were matched,
                         respectively. For example, match_tp[i] gives [gt_idx, pred_idx] for the i'th match.
            'match_fn' : n_fn element array of unmatched ground-truth instances, which are false negatives
            'match_fp' : n_fp element array of unmatched predicted instances, which are false positives
            'mask_tp': n_match element array of true positive pixel counts for each matched mask
            'mask_fn': n_match element array of false negative pixel counts for each matched gt mask
            'mask_fp': n_match element array of of false postiive pixel counts for each matched pred mask}
            'match_tp_iou': n_match element array of IOU scores for each match
    """
    ## match scoring
    gtmasks = masks_to_rle(gt, size)
    predmasks = masks_to_rle(pred, size)


    match_results_ = rle_instance_matcher(gtmasks, predmasks, iou_thresh=iou_thresh, size=size)
    matches_ = np.asarray(match_results_['tp'])
    TP_match_ = len(matches_) #  true positive
    FN_match_ = len(match_results_['fn']) #  false negative
    FP_match_ = len(match_results_['fp']) #  false positive

    match_precision = TP_match_ / (TP_match_ + FP_match_)
    match_recall = TP_match_ / (TP_match_ + FN_match_)

    gtmasks_tp = [gtmasks[i[0]] for i in matches_]
    predmasks_tp = [predmasks[i[1]] for i in matches_]
    mask_true_positive = np.array([rle.area(rle.merge([m1, m2], intersect=True))
                                   for m1, m2 in zip(gtmasks_tp, predmasks_tp)],
                            np.int)
    tp_gt_area = np.array([rle.area(m) for m in gtmasks_tp], np.int)
    tp_pred_area = np.array([rle.area(m) for m in predmasks_tp], np.int)

    mask_false_positive = tp_pred_area - mask_true_positive
    mask_false_negative = tp_gt_area - mask_true_positive

    mask_precision = mask_true_positive /(mask_true_positive + mask_false_positive)
    mask_recall = mask_true_positive /(mask_true_positive + mask_false_negative)

    return {'match_precision': match_precision,
           'match_recall': match_recall,
           'mask_precision': mask_precision,
           'mask_recall': mask_recall,
            'match_tp': matches_,
            'match_fn': match_results_['fn'],
            'match_fp': match_results_['fp'],
            'mask_tp': mask_true_positive,
            'mask_fn': mask_false_negative,
            'mask_fp': mask_false_positive,
            'match_tp_iou': match_results_['iou']}


def merge_boxes(box1, box2):
    """
    Finds the smallest bounding box that fully encloses box1 and box2.


    Boxes are of the form [r1, r2, c1, c2] (indices, not coordinates). The region of image *im* in the box
    can be accessed by im[r1:r2,c1:c2].


    Paramaters
    -----------
        box1, box2: ndarray
            4-element array of the form [r1, r2, c1, c2], the indices of the
            top left and bottom right corners of the box

    Returns
    ----------
        bbox_merge: ndarray
            4-element array containing the combined boxes.

    Examples
    ----------
    TODO example


    """
    r11, r12, c11, c12 = box1
    r21, r22, c21, c22 = box2

    bbox_merge = np.array([min(r11, r21),  # min first row index
                           max(r12, r22),  # max last row index
                           min(c11, c21),  # min first col index
                           max(c12, c22)])  # max last col index

    return bbox_merge


def _min_euclid(a, b):
    """
    Minimum euclidean distance in pixels from tensors *a* to pixels in tensor *b*.

    Used for computing the distance of false positive and false negative pixels to gt
    and pred pixels in mask_edge_distance()

    Parameters
    ------------
    a: tensor
        n x 2 tensor where each row corresponds to one set of (x,y) coords
    b: tensor
        m x 2 tensor where each row corresponds to one set of (x,y) coords

    Returns
    ---------
    min_distances: tensor
        n element tensor of minimum euclidean distances from elements in *a* to *b*.


    Examples
    --------------
    TODO example

    """

    a = a.unsqueeze(axis=1)

    square_diffs = torch.pow(a.double() - b.double(), 2)

    distances = torch.sqrt(square_diffs.sum(axis=2))

    min_distances = distances.min(axis=1)[0]

    return min_distances


def mask_edge_distance(gt_mask, pred_mask, gt_box, pred_box, matches, force_cpu=False):
    """
    Investigate the disagreement between the boundaries of predicted and ground truth masks.

    For every matched pair of masks in pred and gt, determine false positive and false negative pixels.
    For every false positive pixel, compute the distance to the nearest ground truth pixel.
    For every false negative pixel, compute the distance to the nearest predicted pixel.

    Parameters
    -------------
    gt_mask, pred_mask: list or RLEMasks
        ground truth and predicted masks, RLE format
    gt_box, pred_box: array
        array of bbox coordinates
    matches: array
        n_match x 2 element array where matches[i] gives the index of the ground truth and predicted masks in
        gt and pred corresponding to match i. This can be obtained from mask_match_stats (results['match_tp'])
    force_cpu: bool
        if True, prevents running computations on gpu. I added this because my gpu is too small which causes problems ):


    Returns
    -------------
    FP_distances, FN_distances: list(torch.tensor)
        List of results for each match in matches. Each element is a tensor containing the euclidean distances
        (in pixels) from each false positive to its nearest ground truth pixel(FP_distances)
        or the distance from each false negative to the nearest predicted pixel(FN_distances).
    """

    if torch.cuda.is_available() and not force_cpu:
        device = 'cuda'
    else:
        device = 'cpu'

    if type(gt_mask) == RLEMasks:
        gt_mask = gt_mask.rle
    if type(pred_mask) == RLEMasks:
        pred_mask = pred_mask.rle

    gt_masks = [gt_mask[i] for i in matches[:, 0]]
    gt_boxes = [gt_box[i] for i in matches[:, 0]]

    pred_masks = [pred_mask[i] for i in matches[:, 1]]
    pred_boxes = [pred_box[i] for i in matches[:, 1]]


    FP_distances = []
    FN_distances = []
    for gm, pm, gb, pb in zip(gt_masks, pred_masks, gt_boxes, pred_boxes):
        # masks are same size as whole image, but we only need to look in the region containing the masks.
        # combine the bboxes to get region containing both masks
        r1, r2, c1, c2 = merge_boxes(gb, pb)

        # decode RLE, select subset of masks included in box, and cast to torch tensor
        gm = torch.tensor(rle.decode(gm)[r1:r2, c1:c2], dtype=torch.bool).to(device)
        pm = torch.tensor(rle.decode(pm)[r1:r2, c1:c2], dtype=torch.bool).to(device)

        # indices of pixels included in ground truth and predicted masks
        gt_where = torch.stack(torch.where(gm), axis=1)
        pred_where = torch.stack(torch.where(pm), axis=1)

        # indices of false positive (pred and not gt) and false negative (gt and not pred) pixels
        FP_where = torch.stack(torch.where(pm & torch.logical_not(gm)), axis=1)
        FN_where = torch.stack(torch.where(gm & torch.logical_not(pm)), axis=1)

        # distance from false positives to nearest ground truth pixels
        if FP_where.numel():
            FP_dist = _min_euclid(FP_where, gt_where)

        else:
            FP_dist = torch.tensor([], dtype=torch.double)

        # distance from false negatives to nearest predicted pixels
        if FN_where.numel():
            FN_dist = _min_euclid(FN_where, pred_where)
        else:
            FN_dist = torch.tensor([], dtype=torch.double)

        FP_distances.append(FP_dist)
        FN_distances.append(FN_dist)

    return FP_distances, FN_distances


def match_perf_iset(gt, pred, match_results=None, colormap=None, tp_gt=False):
    """
    Stores match true positives, false positives, and false negatives in an instance set for visualization.

    Computes matches between gt and pred masks (unless match_results is supplied).
    Returns an instance set object containing the masks, boxes, and colors for true positive (matched,)
    false positive (unmatched pred) and false negative (unmatched gt) instances. Colors correspond to each
    class of instance. The instance set can be visualized using visualize.quick_visualize_iset() to evaluate
    the performance of a model on matching instances.

    Parameters
    ---------------
    gt, pred: masks object (RLEMasks, list(dic), etc)
        ground truth (gt) and predicted (predicted) masks that will be matched

    match_results: dict or None
        if dict: output from rle_instance_matcher() containing match scores and indices
        if None: rle_instance_matcher() will be called to compute scores with default iou_thresh=0.5

    colormap: dict or None
        if dict: dictionary with keys in ['TP','FP','FN'] for true positive, false positiev, and false negative,
        respectively, and values of RGB or RGBA colors (ie 3 or 4 element array of floats between 0 and 1)

    tp_gt: bool.
        If True, true positives will be displayed from ground truth instances. If False,
        true positives will be displayed from predicted instances.

    Returns
    ----------
    iset: InstanceSet
        instance set whose instances contain the tp, fp, fn instances (the type is indicated by the color)

    colormap (optional): dict
        only returned if the colormap input is None. Dictionary with keys in ['TP','FP','FN'] and values
        are the default colors corresponding to each type of instance.
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

    if colormap is None:  # default values
        colormap = {'TP': np.asarray([0.5,0.,1.],np.float),
                    'FP': np.asarray([0.,1.,1.],np.float),
                    'FN': np.asarray([1.,0.,0.], np.float)}
    
    if tp_gt:
        tp_idx = match_results['tp'][:, 0]
        tp_masks = [gt_masks[i] for i in tp_idx]
        tp_bbox = gt_bbox[tp_idx]
    else:
        tp_idx = match_results['tp'][:, 1]
        tp_masks = [pred_masks[i] for i in tp_idx]
        tp_bbox = pred_bbox[tp_idx]

    tp_colors = np.tile(colormap['TP'], (len(tp_masks), 1))
    
    fp_idx = match_results['fp']
    fp_masks = [pred_masks[i] for i in fp_idx]
    fp_bbox = pred_bbox[fp_idx]
    fp_colors = np.tile(colormap['FP'], (len(fp_masks), 1))
    
    fn_idx = match_results['fn']
    fn_masks = [gt_masks[i] for i in fn_idx]
    fn_bbox = gt_bbox[fn_idx]
    fn_colors = np.tile(colormap['FN'], (len(fn_masks), 1))
    
    masks = RLEMasks(tp_masks +  fp_masks + fn_masks)
    bbox = np.concatenate((tp_bbox, fp_bbox, fn_bbox), axis=0)
    colors = np.concatenate((tp_colors, fp_colors, fn_colors), axis=0)

    iset = InstanceSet()
    iset.instances = Instances(image_size=masks.rle[0]['size'], **{'masks': masks, 'boxes': bbox, 'colors': colors})

    if return_colormap:
        return iset, colormap
    return iset


def mask_perf_iset(gt_masks, pred_masks, match_results=None, mode='reduced'):
    """
    Stores mask true positives, false positives, and false negatives in an instance set for visualization.

    Computes matches between gt and pred masks (unless match_results is supplied).
    Returns an instance set object containing the masks, boxes, and colors for true positive (included in
    both gt and pred masks that are matched), false positive (included in pred but not gt) and false negative
    (included in gt but not pred), and other(the same pixel is included in multiple instances due to overlap).
    Colors correspond to each class of instance.
    The instance set can be visualized using visualize.quick_visualize_iset() to evaluate the quality of matched
    instances.

    Parameters
    ---------------
    gt, pred: masks object (RLEMasks, list(dic), etc)
        ground truth (gt) and predicted (predicted) masks that will be matched

    match_results: dict or None
        if dict: output from rle_instance_matcher() containing match scores and indices
        if None: rle_instance_matcher() will be called to compute scores with default iou_thresh=0.5

    mode: str
        'all' or 'reduced'
        if 'all', there will be 8 colors corresponding to all possible combinations of pixels (described in
        colormap keys)
        if 'reduced', only pixels are only classified as tp, fp, fn, or'other'
    Returns
    ----------
    iset: InstanceSet
        instance set whose instances contain the tp, fp, fn instances (the type is indicated by the color)
    colormap: dict
        Dictionary with keys describing the pixel class and values are the default colors corresponding to each type of
        instance.

    """

    # defaults
    if match_results is None:
        match_results = rle_instance_matcher(gt_masks, pred_masks,)
    gt_masks = masks_to_bitmask_array(gt_masks)
    pred_masks = masks_to_bitmask_array(pred_masks)

    tp_idx = match_results['tp']
    matched_gt = gt_masks[tp_idx[:, 0]]
    matched_pred = pred_masks[tp_idx[:, 1]]
    
    tp_mask_ = np.logical_and(matched_gt, matched_pred,)  # true positive
    fn_mask_ =  np.logical_and(matched_gt, np.logical_not(matched_pred))  # false negative
    fp_mask_ =  np.logical_and(np.logical_not(matched_gt), matched_pred)  # false positive

    def project(masks_): return np.logical_or.reduce(masks_, axis=0)

    tp_reduced = project(tp_mask_).astype(np.uint)
    fn_reduced = project(fn_mask_).astype(np.uint) * 2
    fp_reduced = project(fp_mask_).astype(np.uint) * 4


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
    pixel_map = tp_reduced + fn_reduced + fp_reduced

    if mode == 'all':
        masks = np.zeros((*pixel_map.shape[:2], 7), np.bool)
        for i in range(1,8):
            masks[:,:,i-1] = pixel_map == i

        # maps index to colors
        color_mapper = np.array([
            [0., 0., 0.],
            [0.153, 0.153, 0.000],
            [0.286, 1., 0.],
            [1., 0.857, 0.],
            [1., 0., 0.],
            [0., 0.571, 1.],
            [0., 1., 0.571],
            [0.285, 0., 1.]])

        colors = [color_mapper[1:],
                  ['Other', 'TP', 'FN', 'TP+FN', 'FP', 'TP+FP', 'FN+FP', 'TP+FN+FP']]

    else:
        masks = np.zeros((*pixel_map.shape[:2], 4), np.bool)
        for i, idx in enumerate([1, 2, 4]):
            masks[:, :, i] = pixel_map == idx  # idx for tp, fn, fp
        # idx for pixels in multiple overlapping masks
        masks[:, :, 3] = np.logical_or.reduce([pixel_map == i for i in [3, 5, 6, 7]], axis=0)
        # color-blind friendly palette from https://venngage.com/blog/color-blind-friendly-palette/
        # np.array([[169, 90, 161],[153, 153, 0],[15,32,128],[133, 192, 249]]) / 255
        color_mapper = np.array([[0.5, 0., 1.],
                                 [1.,0.,0.],
                                 [0., 1., 1.],
                                 [1., 1., 0.]])
        colors = [color_mapper, ['TP', 'FN', 'FP', 'other']]

    masks = np.asfortranarray(masks)
    masks = rle.encode(masks)
    masks = RLEMasks(masks)

    i = InstanceSet()
    i.instances = Instances(image_size=masks.rle[0]['size'], **{'masks': masks, 'colors': colors[0],
                                                                  'boxes': np.zeros((len(masks), 4))})

    return i, colors
