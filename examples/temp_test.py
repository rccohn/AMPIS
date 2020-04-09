import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import skimage

import analyze
import evaluate
import characterize

with open('../data/interim/instance_sets/particle_gt_instance_sets.pickle', 'rb') as f:
    particle_gt_instances = pickle.load(f)
#with open('../data/interim/instance_sets/particle_pred_instance_sets.pickle', 'rb') as f:
#    particle_pred_instances = pickle.load(f)


with open('../data/interim/instance_sets/satellite_gt_instance_sets.pickle', 'rb') as f:
    satellite_gt_instances = pickle.load(f)
#with open('../data/interim/instance_sets/satellite_pred_instance_sets.pickle', 'rb') as f:
#    satellite_pred_instances = pickle.load(f)

particle = particle_gt_instances[0]
satellites = satellite_gt_instances[0]

x = characterize.fast_satellite_match(particle.masks, satellites.masks)

