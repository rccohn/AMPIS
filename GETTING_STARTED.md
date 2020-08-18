Getting Started
================

AMPIS provides comprehensive example for data preparation, training, and evaluating instance segmentation models in `examples/powder` and `examples/spheroidite`. The powder example is the most comprehensive and shows how instance segmentation can be applied to measure powder samples on a particle-by-particle basis.   The spheroidite example shows the process for data formatted in a different way and shows how this technique can be used for segmentation of microconstituents in steel. 

These examples follow the same process, which is summarized here.

## Data labeling
Labeling data is a slow and painful process. Fortunately in these examples we provide labels for you! There are two formats for labels that can be used with AMPIS. The powder example shows how to use data annotated with the [vgg image annotator](http://www.robots.ox.ac.uk/~vgg/software/via/), which stores polygons for the segmentation masks. The spheroidite example shows how labels stored in separate images or arrays can be used instead.

#### How much data do I need? 
Surprisingly, not much. Deep learning models typically require thousainds, or even millions, of images to fully train (the ImageNet dataset currently contains over 14 million labeled images.) However, by leveraging transfer learning, the use of pre-trained models allow us to achieve good results with very few (10 or fewer) labeled images. Both examples show good results achieved from small datasets.

## Loading data
#### Loading data dictionaries
After labeling data it is ready to be loaded into AMPIS.  This is accomplished with the `data_utils.get_ddicts()` function. Currently, the data loader only works for single-class instance segmentation in the formats specified above in Data labeling. The powder example shows how separate label files can be used for multi-class segmentation (ie powder particles and satellites.) Detailed info for the format required can be found in the detectron2 documentation under [using custom datasets.](https://detectron2.readthedocs.io/tutorials/datasets.html)



#### Dataset registration
After loading the data, it must be registered. Registration associates a name with a method for retrieving the data in a format that can be used by a model. Both examples demonstrate how to register datasets. Again, detailed information can be found in the detectron2 docs with the above link.

After loading and/or registering data, you can visually verify that the data is correctly loaded using `visualize.display_ddicts()'.

## Model configuration and training

#### Model selection

After the data is annotated, loaded, and registered, it's time to set up the model. Detectron2 provides several pre-trained models with different configurations in the [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). Both examples in AMPIS use `mask_rcnn_R_50_FPN_3x`, which gives good performance for instance segmentation. By leveraging transfer learning, we can get away with achieving high-performance instance segmentation with very small sets of labeled training data.

#### Configurations

The models have many hyperparameters which can be selected to adjust their performance on different datasets. The model architecture and parameters are specified in the `config` settings. Both examples show the most common settings that should be defined to tune the performance of the models. A complete list of configuration options and descriptions of what each option does can be found in the detectron2 [config references](https://detectron2.readthedocs.io/modules/config.html#config-references).

#### Training
After defining the configurations, the model is very straightforward to train. The detectron2 DefaultTrainer (available in the [detectron2engine.defaults](https://detectron2.readthedocs.io/modules/engine.html#module-detectron2.engine.defaults) module) provides all the basic functionality needed for training. If you have enough training data to use a training, validation, and test set, we also provide the AmpisTrainer object, which can compute the validation loss during training. To do this, the validation dataset must be registered separately from the training dataset. Both examples use small training sets and therefore use the standard DefaultTrainer. 

## Model inference
After training the model detectron2 DefaultEvaluator (also located in [detectron2engine.defaults](https://detectron2.readthedocs.io/modules/engine.html#module-detectron2.engine.defaults)) is used to generate predictions on both the training images as well as unseen or held-out images.

The outputs of instance segmentation models are very large. Fortunately, the majority of the analysis and visualization can be done on a compressed representation of the data. `data_utils.format_output()` compresses the results and formats them for storage/later use.

After formatting, `visualize.display_ddicts()` can be used to view the model predictions on an image.

## Model evaluation
Evaluation consists of comparing the model predictions to the ground truth labels for a given image. The `structures.InstanceSet` class is suited for evaluation. InstanceSet objects can load either ground truth labels from ddicts or formatted model predictions that were saved to disk. InstanceSet objects can be visualized with `visualize.display_iset()`.

#### Detection and segmentation scores
Typical computer vision models for instance segmentation are evaluated by the COCO metrics such as AP50 or mAP. These metrics are good for large datasets. The detectron2 [COCOEvaluator](https://detectron2.readthedocs.io/modules/evaluation.html) can be used to compute these scores, if desired. (However, AMPIS provides metrics which provide more detail for smaller datasets. The scores are based on precision (ratio of true positive to all positive predictions) and recall (ratio of true positives to true positive and false negative predictions.)

These metrics are reported for two sets of scores. **Detection** scores describe how many predicted masks matched with a ground truth mask on the basis of IOU score. **Segmentation** scores describe how well each pair of matched masks agree with each other. Details for the metrics are provided in the examples and in the paper.

To compute the scores for each pair of ground truth/predicted results for a given image, use `analyze.det_seg_scores`.

The detection and segmentation results can also be visualized by first calling `analyze.det_perf_iset()` or `analyze.seg_perf_iset()`. This generates an InstanceSet containing the true positives, false positives, and false negative instances for each metric. Then, the InstanceSet can be visualized with the same `visualize.display_iset()` function. 


## Sample characterization 

Up until now, this process has been entirely computer vision. But AMPIS was designed for materials scientists! After training the model and generating predictions, it's time to use the model for scientific exploration!

InstanceSet objects contain the segmentation masks for each image. We can gather some information from these directly. Calling `iset.compute_rprops()` returns a table of mesaurements of the masks using `skimage.measure.regionprops_table()`. The full list of available quantities that can be measured is avaliable in the [skimage documents](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops).

More in-depth analysis requires additional functionality. `ampis.applications` is designated for modules with tools for specific applications. Currently, there is one module- `ampis.applications.powder`. This provides tools for quickly generating particle size distributions from image data as well as the ability to measure the satellite content of powder samples. The implementation of both of these techniques is included in the powder example. Currently this is the only method of directly measuring the satellite contents in powder samples, demonstrating the utility of instance segmentation for applications in materials characterization and quality control! 

## Documentation
For more information, see the [AMPIS documentation](https://ampis.readthedocs.io/). 
