"""
Old functions that have been replaced by updated versions, but I did not want to delete quite yet.
"""

############### ANALYZE.PY ###############################################################################
import numpy as np


def fast_instance_match(gt_masks, pred_masks, gt_bbox=None, pred_bbox=None, IOU_thresh=0.5):
    """
    instance matching based on projected mask labels (see project_masks() function.)
    Label images are two-dimensional arrays (size r x c) where background pixels are
    -1 and all other pixels have integer values corresponding to their index in the
    original r x c x n_masks arrays used to compute the label images.

    Note that gt_masks and pred_masks must contain only instances from a single class. For multi-class instance
    segmentation, the instances can be divided into subsets based on their class labels.
    TODO make helper function to split instance.

    Instances are matched on the basis of Intersection over Union (IOU,) or ratio of areas of overlap between
    2 masks to the total area occupied by both masks.
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
        'gt_tp': n_match x 2 array of indices of matches. The first element corresponds to the index of the gt instance
            for match i. The second element corresponds to the index of the pred index for match i.
        'gt_fn': n_fn element array where each element is a ground truth instance that was not matched (false negative)
        'pred_fp': n_fp element array where each element is a predicted instance that was not matched (false positive)
        'IOU_match': n_match element array of IOU scores for each match.
        }
    """
    ## TODO consider using np.unique[r1:r2,c1:c2,:].max((0,1)) with indexing array instead of projecting instances
    # # onto 2d image to handle case of overlapping instances
    n = gt_masks.shape[2]  # number of ground truth instances

    # get label images for each set of masks
    # gt_labels = project_masks(gt_masks)
    pred_labels = project_masks(pred_masks)

    # get bboxes
    if gt_bbox is None:
        gt_bbox = data_utils.extract_bboxes(gt_masks)
    else:  # TODO handle non-integer bboxes (with rounding and limits at edge of images)
        gt_bbox = gt_bbox.astype(np.int) if gt_bbox.dtype is not np.int else gt_bbox
    if pred_bbox is None:
        pred_bbox = data_utils.extract_bboxes(pred_masks)
    else:
        pred_bbox = pred_bbox.astype(np.int) if pred_bbox.dtype is not np.int else pred_bbox

    gt_tp = []  # true positive  [[gt_idx, pred_idx]]
    gt_fn = []  # false negatives [gt_idx]
    IOU_match = []

    pred_matched = np.zeros(pred_masks.shape[2], np.bool)  # values will be set to True when

    for gt_idx, (mask, box) in enumerate(zip(np.transpose(gt_masks, (2, 0, 1)),
                                             gt_bbox)):
        # find predicted masks in the same neighborhood as
        # the ground truth mask in question
        g_r1, g_c1, g_r2, g_c2 = box
        neighbor_idx = np.unique(pred_labels[g_r1:g_r2,
                                 g_c1:g_c2])
        if neighbor_idx[0] == -1:
            neighbor_idx = neighbor_idx[1:]

        if len(neighbor_idx) > 0:
            mask_candidates = pred_masks[..., neighbor_idx]
            bbox_candidates = pred_bbox[neighbor_idx, ...]

            IOU_ = np.zeros(bbox_candidates.shape[0], np.float)

            for i, (pmask, pbox) in enumerate(zip(np.transpose(mask_candidates, (2, 0, 1)),
                                                  bbox_candidates)):
                p_r1, p_c1, p_r2, p_c2 = pbox

                # extract the smallest window indices [r1:r2, c1:c2]
                # that fully includes both the gt and predicted masks
                r1 = min(g_r1, p_r1)
                c1 = min(g_c1, p_c1)
                r2 = max(g_r2, p_r2)
                c2 = max(g_c2, p_c2)

                IOU_[i] = IOU(*[x[r1:r2, c1:c2] for x in [mask, pmask]])

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

    labels = np.zeros((r, c), np.int) - 1  # initialize and set background to -1
    for i, mask in enumerate(np.transpose(masks, (2, 0, 1))):
        labels[mask] = i

    return labels


def expand_masks(masks):
    """
    decompresses masks to n_mask x r x c boolean array for characterization and computation
    Args:
        masks: masks

    Returns:
        bitmasks- n_mask x r x c boolean array
    """

    dtype = type(masks)

    if dtype == np.ndarray:  # masks are already bitmasks
        bitmasks = masks

    elif dtype == list:
        if type(masks[0]) == dict:  # RLE mask
            pass  # bitmasks =

    return bitmasks

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

    return results


def ordinal_hist_distance(A, B, normalize=True):
    """
    Computes the ordinal distance between 2 histograms.
    Normalizing allows for comparison of histograms with different numbers of samples.

    The distance describes the number of moves required to transform one histogram into the other,
    where a 'move' indicates shifting one sample to the next or previous bin.

    The un-normalized distance describes the number of blocks moved * number of bins moved

    The normalized distance describes the average number of bins moved for each block in the normalized histogram.

    Reference: Cha, Srihari, On measuring the distance between histograms, Pattern Recognition 35 (2002) 1355-1370

    Parameters:
        A, B,: 1 dimensional numpy arrays of the histograms being compared
        normalize: if True, the normalized distance will be computed.

    Returns:
        h_dist: int or float, distance measure between histograms A and B.
    """

    assert A.ndim == 1 and B.ndim == 1
    assert len(A) == len(B)

    if normalize:
        nA = A.sum()
        nB = B.sum()
        N = nA * nB
        A = nB * A
        B = nA * B

    h_dist = np.abs((A - B).cumsum().sum())

    if normalize:
        h_dist /= N

    return h_dist