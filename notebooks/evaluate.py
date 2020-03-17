import colorsys
import numpy as np
import skimage
import skimage.filters
import skimage.morphology



def mask_dict(masks, ids):
    """
    helper function for evaluate_masks()
    masks: rxcxn numpy array of masks
    ids: n-element numpy array of class labels corresponding to each mask
    returns: results- dictionary where each key is a unique class label and
    the value is a list of masks corresponding to that class
    """
    r, c, _ = masks.shape

    unique, counts = np.unique(ids, return_counts=True)

    results = {}
    temp = {}
    for key, value in zip(unique, counts):
        results[key] = [np.zeros((r, c, value), dtype=np.bool), np.zeros(value, np.int)]
        temp[key] = 0

    for i, idx in enumerate(ids):
        results[idx][0][:, :, temp[idx]] = masks[:, :, i]  # stores mask
        results[idx][1][temp[idx]] = i  # stores index of mask in original results
        temp[idx] += 1

    return results


# Todo use overlap and centroid distance (and possibly size if needed) to get corresponding masks
def compare_masks(gt_masks, pred_masks):
    """
    helper function for evaluate_masks()
    gt_masks: [A, B] rxcxn numpy array of ground truth masks of some class i
    pred_masks: [A, B] rxcxn numpy array of predicted masks of some class i
    returns-
    """
    gt_r, gt_c, gt_n = gt_masks[0].shape
    # TODO fix error if pred_masks[1] is empty
    pred_r, pred_c, pred_n = pred_masks[0].shape

    gt_matches = np.zeros(gt_n, dtype=np.bool)
    pred_matches = np.zeros(pred_n, dtype=np.bool)

    match_pairs = []

    for i in range(gt_n):
        mask = gt_masks[0][:, :, i]
        print('test')
        print(mask.shape)
        print(pred_masks[0][:,:,0].shape)
        IOUs = np.asarray([IOU(mask, pred_masks[0][:, :, j]) for j in range(pred_n) if bool(pred_matches[j]) is False])

        if IOU.max() == 0:  # no good match
            continue
        j = np.argmax(IOU)

        gt_matches[i] = True
        pred_matches[j] = True

        match_pairs.append((gt_masks[1][i], pred_masks[1][j]))


    return match_pairs

def evaluate_masks(gt, gt_id, pred):
    """
    Determine IOU, precision, recall, F1 score for mask predictions.

    gt- ground truth- AmesDataset object
    gt_id- index in gt.image_info list
    pred- predictions results (output of model.detect()[0] for one image)
    pred_i- of predictions output. Defaults to 0 for the case where there is only one image.

    estimates corresponding masks between ground truth and predictions and computes statistics for each one.

    Ground truths with no corresponding predictions and vice versa score 0
    """

    gt_masks, gt_classes = gt.load_mask(gt_id)

    pred_masks, pred_classes = tuple([pred[x] for x in ['masks', 'class_ids']])

    gt_results = mask_dict(gt_masks, gt_classes)
    pred_results = mask_dict(pred_masks, pred_classes)

    # TODO handle case where key is present in pred_results but not gt_results
    # This could likely happen on images of single powder particles with
    # false predictions of satellites
    for key, value in gt_results.items():
        matches = compare_masks(value, pred_results.get(key, [np.array([]), None]))


# Todo-
def remove_outliers(gt_masks, gt_classes, pred, lower_thresh=2, upper_thresh=2, return_mask=False, colors=None):
    """
    remove outliers based on threshold critera
    gt_masks, gt_classes from AmesDataset.load_masks()
    pred- predictions results (output of model.detect()[0] for single image)

    lower_thresh- float- lower threshold for pixel ratios
    upper_thresh- float- upper threshold for pixel ratios

    thresholds are compared to the ratio of pixels in each predicted mask with the biggest or smallest
    ground truth mask of the same class


    lower_thresh- if n_pixels(pred)/n_pixels(GT_min) < lower_thresh, predicted mask is remved
    upper_thresh- if n_pixels(pred)/n_pixels(GT_max) > upper_thresh, predicted mask is removed

    return_mask- if True, returns a vector with number of elements equal to original number of
    elements in pred without removing any outliers. Vector is a boolean mask used to exclude outliers.
    Can be used to track which elements are indicated as outliers and thrown out.
    colors- list of colors used for display purposes only. N_elements x 3 
    """

    pred_masks, pred_classes = tuple([pred[x] for x in ['masks', 'class_ids']])

    gt_masks_dict = mask_dict(gt_masks, gt_classes)
    pred_masks_dict = mask_dict(pred_masks, pred_classes)

    final = []

    for classLabel, gt_masks in gt_masks_dict.items():
        pred_masks = pred_masks_dict.get(classLabel)
        if pred_masks is None:
            continue
        gt_areas = gt_masks[0].sum((0, 1))
        pred_areas = pred_masks[0].sum((0, 1))
        gt_max = gt_areas.max()
        gt_min = gt_areas.min()

        selection = np.logical_and(pred_areas / gt_max < upper_thresh, pred_areas / gt_min > lower_thresh)

        final.extend((pred_masks[1], selection))
    final = np.asarray(sorted(final, key=lambda x: x[0]), np.int)
    final = final[1].astype(np.bool)

    # results = [x[final] for x in pred[pred_i].keys()]

    newresults = {'masks': pred['masks'][:, :, final],
                  'scores': pred['scores'][final],
                  'class_ids': pred['class_ids'][final]}

    if pred.get('scores', None) is not None:
        newresults['scores'] = pred['scores'][final]
    if pred.get('bbox', None) is not None:
        newresults['bbox'] = pred['bbox'][final]
    if pred.get('rois', None) is not None:
        newresults['rois'] = pred['rois'][final]
    if colors is not None:
        newcolors = np.asarray([x for x, y in zip(colors, final) if y])
        newresults['colors'] = newcolors
    
    if return_mask:
        return newresults, final

    return newresults

def remove_outlier_dict(gt, pred, lower_thresh=2, upper_thresh=2, colors=None):
    """
    Wrapper function for remove_otuliers when gt and pred are dictionaries (instead of passing all parameters manually)
    :param gt: ground truth dictionary of the form {'class_ids' : (array of class ids), 'bbox' : (array of bboxes), 'masks' : (array of masks)}
    :param pred: prediction mask of the form {'class_ids' : (array of class ids), 'bboxes' : (array of bboxes), 'masks' : (array of masks), 'scores' : (array of scores)}
    """
    results = remove_outliers(gt['masks'], gt['class_ids'], pred, lower_thresh=lower_thresh, upper_thresh=upper_thresh, colors=colors, return_mask=False)
    
    return results

def refine_masks_watershed_li(im, masks, bclose=False, remove_bg=False, new_bgcolor=None, merge=False):
    """
    Thresholds background of image and uses masks as starting points for watershed,
    runs watershed to update masks, ensuring smoother edges on masks and including
    regions which may have been cut off during initial segmentation.
    r x c- size of image and each individual mask
    n_mask- number of mask
    :param im: r x c numpy array of image
    :param masks: numpy array r x c x n_masks of boolean values corresponding to binary masks
    :param bclase: bool- if True, binary closing is applied to each mask after thresholding to prevent
                           pixels in the mask to be classified as background 
    :param remove_bg: bool- if False, only masks are returned. If True, masks and image with
                            background subtracted are returned.
    :param new_bgcolor: float between 0 and 1 or int between 0 and 255 GRAYSCALE value of 
    background after removal. Currently, only grayscale is supported.
    :param merge: bool- if True, union of watershed masks with original masks is taken.
                   This can prevent small groups of pixels from being left out after thresholding.
    :return:
    new_masks: numpy array r x c x n_masks of boolean values corresponding to updated masks
    im_no_bg: image with background subtracted and set to new_bgcolor, array of same size of images
    """
    newmasks = np.zeros_like(masks) # initialize newmasks

    r, c, nmask = masks.shape  # get dimensions of mask
    imshape = im.shape
    assert r, c == imshape[:2]  # masks must be same size as image

    # make sure image is only 1 channel
    if len(imshape) == 3:
        im_gray = skimage.color.rgb2gray(im)
    else:
        im_gray = im


    li_thresh = skimage.filters.threshold_li(im_gray)
    im_mask = im_gray > li_thresh
    
    #TODO distance transform (I think this doesn't change results much)
    
    #TODO see if this is useful
    #for i in range(nmask):
    #    im_mask = np.logical_or(im_mask, masks[:, :, i]) # ensures all masks are included

    # combine array of binary masks into single label image (integers)
    watershed_start = np.zeros((r, c), np.int)
    for i in range(1, nmask+1):
        watershed_start[masks[:, :, i-1]] = i

    # apply watershed
    watershed_seg = skimage.morphology.watershed(im_gray, markers=watershed_start, connectivity=1, offset=None,
                                                 mask=im_mask, compactness=0, watershed_line=False)

    for i in range(1,nmask+1):
        newmask = watershed_seg == i
        if bclose:
            newmask = skimage.morphology.binary_closing(newmask)
        
        newmasks[:, :, i-1] = newmask

    if merge:
        newmasks = np.logical_or(newmasks, masks)
    
    if remove_bg:
        newim = im.copy()
        newim[watershed_seg == 0] = new_bgcolor
        
        return newmasks, newim 
        
    return newmasks

def pad_indices(im, value=None):
    """
    determines indices of original image from image padded with value
    :param im: r x c grayscale image or r x c x 3 rgb image, numpy array
    :param value: value of padding- either int or float- value is used directly. If 
                value is None, the first element in the grayscale image is used
                (eg im[0,0])
    :returns:
    minrow, maxrow, mincol, maxcol- tuple of ints where im[minrow:maxrow, mincol:maxcol] removes padding from image
    """
    
    r, c = im.shape[:2]  # image shape

    if len(im.shape) == 3:  # image is RGB
        imgray = skimage.color.rgb2gray(im)  # convert to grayscale for consistency when checking zero-padding
    else:
        imgray = im  # image is already grayscale

    if value is None:
        value = imgray[0, 0] # use first element in image
    
    im_bool = imgray != value
        
    rowsums = im_bool.sum(1) != 0  # sum of intensity values along each row
    colsums = im_bool.sum(0) != 0  # sum of intensity values along each column
    
    minrow = np.argmax(rowsums)  # min row index containing nonzero value
    maxrow = r - np.argmax(np.flip(rowsums))  # max row index containing nonzero value

    mincol = np.argmax(colsums)  # min col index containing nonzero value
    maxcol = c - np.argmax(np.flip(colsums))  # max col max col index containing nonzero value
    
    return minrow, maxrow, mincol, maxcol


def remove_pad_and_resize(im, masks=None, newshape=None, value=0):
    """
    removes zero-padding on image (caused from resizing in matterport mask R-CNN)
    and corresponding masks. Resizes image to newshape if requested.
    :param im: r x c grayscale image or r x c x 3 rgb image, numpy array
    :param masks: r x c x n_masks array of masks
    :param newshape: [new_r, new_c] new shape
    :param value: value of padding- default 0
    :return: im_new- reshaped image- new_r x new_c x [1 or 3] (color channels matches input)
             masks_new- reshaped masks- new_r x new_c x n_masks

    """
    r, c = im.shape[:2]  # image shape

    if len(im.shape) == 3:  # image is RGB
        imgray = skimage.color.rgb2gray(im)  # convert to grayscale for consistency when checking zero-padding
    else:
        imgray = im  # image is already grayscale

    #     rowsums = imgray.sum(1) != value  # sum of intensity values along each row
    #     colsums = imgray.sum(0) != value  # sum of intensity values along each column
    
    #     minrow = np.argmax(rowsums)  # min row index containing nonzero value
    #     maxrow = r - np.argmax(np.flip(rowsums))  # max row index containing nonzero value
    
    #     mincol = np.argmax(colsums)  # min col index containing nonzero value
    #     maxcol = c - np.argmax(np.flip(colsums))  # max col max col index containing nonzero value

    minrow, maxrow, mincol, maxcol = pad_indices(im, value=value)

    im_new = im[minrow:maxrow, mincol:maxcol]  # remove zero padding from image
    
    if masks is None:
        return im_new
        
    
    n1, n2, n3 = masks.shape  # mask array shape

    assert n1 == r and n2 == c  # masks must be same size as image

    masks_new = masks[minrow:maxrow, mincol:maxcol, :]  # remove zero padding from masks

    if newshape is not None:  # image resizing is requested
        im_new = skimage.transform.resize(im_new, *newshape)  # resize image
        temp = np.zeros((*newshape, n3), np.int)  # for masks

        for i in range(n3):
            temp[:, :, i] = skimage.transform.resize(masks_new[:, :, i], *newshape)  # resize each mask
            temp[:, :, i] = temp.astype(np.bool)  # convert back to bool

        masks_new = temp

    return im_new, masks_new

def random_colors(N, seed, bright=True): # controls randomstate for plotting consistentcy
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


def filter_small_masks(pred, thresh=100):
    """
    Removes instances with mask area lower than thresh
    :param pred: dictionary with keys ['masks', 'class_ids', 'bbox', 'colors' (optional, 'scores' (optional)]. 
    :param thresh: int or float- minum area in pixels that will be kept
    returns:
    pred_filtered- dictionary with same keys as pred where elements have been filtered to remove small instances
    """
    mask_filter_ = pred['masks'].sum((0,1)) >= thresh  # boolean mask array for instances with area greater or equal than thresh
    
    pred_filtered = {'masks' : pred['masks'][:,:,mask_filter_]}  # mask format is different so it is computed separately
    
    for key in [x for x in pred.keys() if x is not 'masks']:  # compute rest of masks
        pred_filtered[key] = pred[key][mask_filter_]
    
    return pred_filtered
