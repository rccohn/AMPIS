# Copyright (c) 2020 Ryan Cohn and Elizabeth Holm. All rights reserved.
# Licensed under the MIT License (see LICENSE for details)
# Written by Ryan Cohn
"""
Contains functions for visualizing images with segmentation masks overlaid.
"""
import colorsys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from . import structures


def random_colors(n, seed, bright=True):  # controls randomstate for plotting consistentcy
    """
    Generate random colors for mask visualization.

    To get visually distinct colors, generate colors in HSV with uniformly distributed hue and then  convert to RGB.
    Taken from Matterport Mask R-CNN visualize, but added seed to allow for reproducability.

    Parameters
    ----------
    n: int
        number of colors to generate

    seed: None or int
        seed used to control random number generater.
        If None, a randomly generated seed will be used

    bright: bool
        if True, V=1 used in HSV space for colors. Otherwise, V=0.7.

    Returns
    ---------
    colors: ndarray
        n x 3 array of RGB pixel values

    Examples
    ----------
    TODO quick example

    """

    rs = np.random.RandomState(seed=seed)

    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    rs.shuffle(colors)
    colors = np.asarray(colors)
    return colors


def display_ddicts(ddict, outpath=None, dataset='', gt=True, img_path=None,
                   suppress_labels=False, summary=True):
    """
    Visualize gt annotations overlaid on the image.

    Displays the image in img_path. Overlays the bounding boxes and segmentation masks of each instance in the image.

    Parameters
    ----------
    ddict: list(dict) or
        for ground truth- data dict containing masks. The format of ddict is described below in notes.

    outpath: str or path-like object, or None
        If None, figure is displayed with plt.show() and not written to disk
        If string/path, this is the location where figure will be saved to

    dataset: str
        name of dataset, included in filename and figure title.
        The dataset should be registered in both the DatasetCatalog and MetadataCatalog
        for proper plotting.
        (see detectron2 datasetcatalog for more info.)


    gt: bool
        if True, visualizer.draw_dataset_dict() is used for GROUND TRUTH instances
        if False, visualizer.draw_instance_predictions is used for PREDICTED instances

    img_path: str or path-like object
        if None, img_path is read from ddict (ground truth)
        otherwise, it is a string or path to the image file

    suppress_labels: bool
        if True, class names will not be shown on visualizer

    summary: bool
        If True, prints summary of the ddict to terminal

    Returns
    -------
    None

    Notes
    -------
    Ddict should have the following format:
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
    if img_path is None:
        img_path = ddict['file_name']
    img_path = Path(img_path)

    if suppress_labels:
        if gt:
            ids = [x['category_id'] for x in ddict['annotations']]
        else:
            ids = ddict['instances'].pred_classes
        u = np.unique(ids)
        metadata = {'thing_classes': ['' for x in u]}
    else:
        metadata = MetadataCatalog.get(dataset)

    visualizer = Visualizer(cv2.imread(str(img_path)), metadata=metadata, scale=1)

    if gt:  # TODO automatically detect gt vs pred?
        vis = visualizer.draw_dataset_dict(ddict)
        n = ddict['num_instances']
    else:
        vis = visualizer.draw_instance_predictions(ddict['instances'])
        n = len(ddict['instances'])

    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.imshow(vis.get_image())
    ax.axis('off')
    ax.set_title('{}\n{}'.format(dataset, img_path.name))
    fig.tight_layout()
    if outpath is not None:
        fig_path = Path(outpath, '{}-n={}_{}.png'.format(dataset, n, img_path.stem))
        fig.savefig(fig_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

    if summary:
        summary_string = 'ddict info:\n\tpath: {}\n\tnum_instances: {}'.format(img_path, n)
        print(summary_string)


def display_iset(img, iset, metadata=None, show_class_idx=False, show_scores=False, ax=None, colors=None,
                 apply_correction=False):
    """
    Visualize instances in *iset* overlaid on image *img*.

    Displays the image and overlays the instances (masks, boxes, labels, etc.) If no axis object is provided to
    *ax*, creates and displays the figure. Otherwise, visualization is plotted on *ax* in place.

    Parameters
    ----------
    img: ndarray
        r x c {x 3} array of pixel values. Can be grayscale (2 dimensions) or RGB (3 dimensions)

    iset: InstanceSet object
        iset.instances, a detectron2 Instances object, is used to get the masks, boxes, class_ids, scores
        that will be displayed on the visualization.

    metadata: dict or None
        If None, metadata (ie string class labels) will not be shown on image. Else, metadata
        contains the metadata passed to detectron2 visualizer. In most cases, this should be a dictionary with the
        following structure:
        {
        'thing_classes': list
            list of strings corresponding to integer indices of class labels.
            For example, if the classes are 0 for 'particle' and 1 for 'satellite',
            then metadata['thing_classes'] = ['particle','satellite']
        }

    show_class_idx: bool
        if True, displays the class label (metadata['thing_classes'][class_idx]) on each instance in the image
        default: False

    show_scores: bool
        if True, displays the confidence scores (output from softmax) on each instance in the image.
        default: False

    ax: matplotlib axis object or None
        If an axis is supplied, the visualization is displayed on the axis.
        If ax is None, a new figure is created, and plt.show() is called for the visualization.

    colors: ndarray or None
        Colors for each instance to be displayed.
        if colors is an ndarray, should be a n_mask x 3 array of colors for each mask.
        if colors is None and iset.instances.colors is defined, these colors are used.
        if colors is None and iset.instances.colors is not defined, colors are randomly assigned.

    apply_correction: bool
        The visualizer appears to fill in masks. Applying the mask correction forces hollow masks to
        appear correctly. This is mostly used when displaying the results from analyze.mask_perf_iset().
        In other cases, it is not needed.



    Returns
    -------
    None

    Notes
    -------
    Ddict should have the following format:
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

    # by default, colors will be extracted from instances. Otherwise, custom colors can be supplied.
    if colors is None:
        if iset.instances.has('colors'):
            colors = iset.instances.colors

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    V = Visualizer(img, metadata, scale=1)

    if show_class_idx:

        if show_scores:
            extra = ': '
        else:
            extra = ''
        class_idx = ['{}{}'.format(metadata['thing_classes'][idx], extra) for idx in iset.instances.class_idx]
    else:
        class_idx = ['' for x in range(len(iset.instances))]

    if show_scores:
        scores = ['{:.3f}'.format(x) for x in iset.instances.scores]
    else:
        scores = ['' for x in range(len(iset.instances))]  # gt do not have scores,
        # so must iterate through a field that exists


    labels = ['{}{}'.format(idx, score) for idx, score in zip(class_idx, scores)]

    if iset.instances.has('masks'):
        masktype = type(iset.instances.masks)
        if masktype == structures.RLEMasks:
            masks = iset.instances.masks.rle
        else:
            masks = iset.instances.masks
    else:
        masks = None

    if iset.instances.has('boxes'):
        boxes = iset.instances.boxes
    else:
        boxes = None

    vis = V.overlay_instances(boxes=boxes, masks=masks, labels=labels,
                              assigned_colors=colors)
    vis_img = vis.get_image()

    # detectron2 visualizer can fill in masks that are not completely full
    # In some cases, we don't want this. Thus, we manually overwrite areas
    # that are filled in with pixels from the original image.
    if apply_correction:
        bitmasks = structures.masks_to_bitmask_array(iset)
        bitmasks_reduced = np.logical_or.reduce(bitmasks, axis=0)
        mask_correction = np.logical_not(bitmasks_reduced)
        vis_img[mask_correction] = img[mask_correction]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7), dpi=150)
        ax.imshow(vis_img)
        ax.axis('off')
        plt.show()

    else:
        ax.imshow(vis_img)
        ax.axis('off')
