import colorsys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer


from . import structures


def random_colors(N, seed, bright=True):  # controls randomstate for plotting consistentcy
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    Taken from Matterport Mask R-CNN visualize
    """

    rs = np.random.RandomState(seed=seed)

    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    rs.shuffle(colors)
    colors = np.asarray(colors)
    return colors


def quick_visualize_ddicts(ddict, root, dataset, gt=True, img_path=None, suppress_labels=False, summary=True):
    """
    Visualize gt instances and save masks overlaid on images in target directory
    Args:
        ddict:for ground truth- data dict containing masks, see output of get_ddicts()
              for predictions- output['instances'] where output is generated from predictor
        root: path to save figures
        dataset: name data is registered to in datasetdict
        gt: if True, visualizer.draw_dataset_dict() is used for GROUND TRUTH instances
            if False, visualizer.draw_instance_predictions is used for PREDICTED instances
        img_path: if None, img_path is read from ddict (ground truth)
        otherwise, it is a string or path to the image file
        suppress_labels: if True, class names will not be shown on visualizer
        summary: prints summary of the ddict to terminal

    """
    if img_path is None:
        img_path = pathlib.Path(ddict['file_name'])
    img_path = pathlib.Path(img_path)

    metadata = MetadataCatalog.get(dataset)
    if suppress_labels:
        metadata = {'thing_classes': ['' for x in metadata.thing_classes]}

    visualizer = Visualizer(cv2.imread(str(img_path)), metadata=metadata, scale=1)

    if gt:  # TODO automatically detect gt vs pred?
        vis = visualizer.draw_dataset_dict(ddict)
        n = ddict['num_instances']
    else:
        vis = visualizer.draw_instance_predictions(ddict)
        n = len(ddict)  # TODO len(ddict['annotations?']) double check this

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.imshow(vis.get_image())
    ax.axis('off')
    ax.set_title('{}\n{}'.format(dataset, img_path.name))
    fig.tight_layout()
    fig_path = pathlib.Path(root, '{}-n={}\n{}.png'.format(dataset, n, img_path.stem))
    fig.savefig(fig_path, bbox_inches='tight')
    if matplotlib.get_backend() is not 'agg':  # if gui session is used, show images
        plt.show()
    plt.close(fig)

    if summary:
        summary_string = 'ddict info:\n\tpath: {}\n\tmask format: {}\n\tnum_instances: {}'.format(ddict['file_name'],
                                                                                               ddict['mask_format'],
                                                                                               n)
        print(summary_string)


def quick_visualize_iset(img, metadata, iset, show_class_idx=False, show_scores=False, ax=None, colors=None):
    """
    visualize instance set
    TODO finish docs
    Args:
        img:
        metadata:
        iset:
        show_class_idx:
        show_scores:
        ax:
        colors:

    Returns:

    """
    # by default, colors will be extracted from instances. Otherwise, custom colors can be supplied.
    if colors is None:
        if iset.instances.has('colors'):
            colors = iset.instances.colors
        else:
            colors = None

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
            masks = iset.instances.masks.masks
        else:
            masks = iset.instances.masks
    else:
        masks=None

    if iset.instances.has('boxes'):
        boxes = iset.instances.boxes
    else:
        boxes = None

    vis = V.overlay_instances(boxes=boxes, masks=masks, labels=labels,
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

