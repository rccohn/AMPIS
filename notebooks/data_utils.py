import numpy as np

def extract_boxes(masks, mask_mode='detectron2', box_mode='detectron2'):
    """
    Extracts bounding boxes from boolean masks. Can be formatted for use with
    either  detectron2 (default) or the matterport visualizer.
    Args:
        masks: boolean array of masks. Can be 2 dimensions for 1 mask or 3 dimensions for array of masks.
        mask shape specified by mask_mode.
        mask_mode: if 'detectron2,' masks are shape n_mask x r x c. if 'matterport,' masks are r x c x n_masks.
        box_mode: if 'detectron2', boxes will be returned in [x1,y1,x2,y2] floating point format.
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
    if box_mode == 'detectron2':
        dtype = np.float
        order = [1,0,3,2] # box [x1, y1, x2, y2]
    else:
        dtype= np.int32
        order = [0, 1, 2, 3] # box [y1, y2, x1, x2]

    boxes = np.zeros((masks.shape[0], 4), dtype=dtype)
    for i, m in enumerate(masks):

        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2], dtype=dtype)[order]

    return boxes


def instances_to_numpy(pred):
    """
    converts detectron2 instance object to dictionary of numpy arrays so that data processing and visualization
    can be done in environments without CUDA.
    :param pred: detectron2.structures.instances.Instances object, from generating predictions on data
    returns:
    pred_dict: Dictionary containing the following fields:
    'boxes': n_mask x 4 array of boxes
    'masks': n_mask x R x C array of masks
    'class': n_mask element array of class ids
    'scores': n_mask element array of confidence scores (from softmax)
    """

    pred_dict = {}

    for item, attribute in zip(['boxes', 'masks', 'class', 'scores'],
                               ['pred_boxes', 'pred_masks', 'pred_classes', 'scores']):
        if item is 'boxes':

            pred_dict[item] = pred.__getattribute__(attribute).tensor.to('cpu').numpy()
        else:
            pred_dict[item] = pred.__getattribute__(attribute).to('cpu').numpy()

    return pred_dict
