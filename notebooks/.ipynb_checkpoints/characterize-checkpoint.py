import numpy as np
import os
import pathlib

import mrcnn_utils
import mrcnn_visualize
import evaluate
import analyze # TODO update fast_satellite_match doc to include correct file reference once refactoring occurs

def fast_satellite_match(particle_masks, satellite_masks, particle_bbox=None, satellite_bbox=None, match_thresh=0.5):
    """
    Given masks corresponding to individual powder particles and satellites in an image, respectively, 
    match the satellites to their corresponding particles. Follows an approach similar to
    fast_instance_match() in analyze.py TODO update file as needed
    
    Instances are matched on the basis of the fraction of the satellite covered by a given particle mask.
    A particle may match multiple satellites, but each satellite may only match one particle.
    
    inputs:
    :param gt_masks: r x c x n_mask_gt boolean numpy array of of ground truth masks
    :param pred_masks: r x c x n_mask_pred boolean numpy array  of predicted masks
    :param gt_bbox: n_mask_gt x 4 array of ground truth bounding box coordinates for 
                    each mask. If None, bounding boxes are extracted from labels_gt.
    :param gt_bbox: n_mask_gt x 4 array of bounding box coordinates for each predicted 
                    mask. If None, bounding boxes are extracted from labels_gt.
    :param match_thresh: float between 0 and 1 (inclusive,) if the max intersection for mask pairs
                       is less than or equal to the threshold, the mask will be considered 
                       to not have a valid match. Values above 0.5 ensure that masks do not 
                       have multiple matches.

    returns:
      :param results: dictionary with the following structure:
        {
        'particle_satellite_match_idx': dictionary. Keys of dictionary are indices of particles that are matched.
                                Values of dictionary are a list of indices of matched satellites for the particle.
        'satellite_matches': n_match x 2 element array of indices of matched satellite instances. First index indicates
                                index of particle, second element is index of satellite. 
        'particles_matched': n_match element array of indices of matched particle instances. 
                        for match i,
                        particle_masks[:,:,particle_idx[i]] matches satellite_masks[:,:,satellite_idx[i]]
        'unmatched_particle_idx' : n_particle_mask - n_match element array of indices indicating unmatched particle masks
        'unmatched_satellite_idx' : n_satellite_mask - n_match element array of indices indicating unmatched satellite masks
        'Intersection_scores': n_match element array of intersection scores for each match. Same order as satellite_matches.
        }
    """

    # get bboxes
    if particle_bbox is None:
        particle_bbox = mrcnn_utils.extract_bboxes(particle_masks)
    else: # TODO handle non-integer bboxes (with rounding and limits at edge of images)
         particle_bbox = particle_bbox.astype(np.int)  
    
    if satellite_bbox is None:
        satellite_bbox = mrcnn_utils.extract_bboxes(satellite_masks)
    else:
        satellite_bbox = satellite_bbox.astype(np.int) 
    
    ## Particles can have satellites, but satellites can only match one particle.
    ## Therefore, matching is done on the basis of satellites.
    particle_labels = analyze.project_masks(particle_masks)
    
    
    particle_matches = []
    satellite_matches = []
    Intersection_scores = []
    
    particles_matched_bool = np.zeros(len(particle_bbox), dtype=np.bool)
    satellites_unmatched = []

    ## TODO do this with multiprocessing
    
    for satellite_idx, (mask, box) in enumerate(zip(np.transpose(satellite_masks, (2,0,1)), 
                                                   satellite_bbox)): # for each particle
        s_r1, s_c1, s_r2, s_c2 = box
        
        # find satellite particles in neighborhood of particle
        neighbor_idx = np.unique(particle_labels[s_r1:s_r2, s_c1:s_c2])
        neighbor_idx = neighbor_idx[1:] if neighbor_idx[0] == -1 else neighbor_idx
        
        # if there's at least one match
        if neighbor_idx.shape[0] > 0:
            mask_candidates = np.transpose(particle_masks[...,neighbor_idx], (2,0,1))
            bbox_candidates = particle_bbox[neighbor_idx]
    
            Iscore_= np.zeros(bbox_candidates.shape[0], np.float) # intersection scores
    
            # loop through potential neighboring satellite masks
    
    
            for i, (pmask, pbox) in enumerate(zip(mask_candidates, bbox_candidates)):
                p_r1, p_c1, p_r2, p_c2 = pbox
    
                # extract the smallest window indices [r1:r2,c1:c2]
                # that fully includes the particle and satellite masks
    
                r1 = min(p_r1, s_r1)
                c1 = min(p_c1, s_c1)
                r2 = max(p_r2, s_r2)
                c2 = max(p_c2, s_c2)
    
                particle_mask_i = pmask[r1:r2,c1:c2]
                satellite_mask_i = mask[r1:r2,c1:c2]


                # find fraction of satellite pixels in intersection
                Iscore_[i] = np.logical_and(particle_mask_i, satellite_mask_i).sum()/satellite_mask_i.sum()

            Iscore_amax = Iscore_.argmax()
            Iscore_max = Iscore_[Iscore_amax]


            if Iscore_max > match_thresh:
                satellite_matches.append([neighbor_idx[Iscore_amax], satellite_idx])
                particles_matched_bool[neighbor_idx[Iscore_amax]] = True

                Intersection_scores.append(Iscore_max)
                
            else: # no match due to low ientersection score
                satellites_unmatched.append(satellite_idx)
        else: # no match due to no neighbors
            satellites_unmatched.append(satellite_idx)
    
    particle_satellite_match_idx = {}
    
    for match in satellite_matches:
        if match[0] not in particle_satellite_match_idx.keys():
            particle_satellite_match_idx[match[0]] = []
        particle_satellite_match_idx[match[0]].append(match[1])
    
    particle_idx = np.arange(len(particle_bbox), dtype=np.int)
    particles_matched = particle_idx[particles_matched_bool]
    particles_unmatched = particle_idx[np.logical_not(particles_matched_bool)]
    
    
    
    results = {
              'particle_satellite_match_idx': particle_satellite_match_idx,
              'satellite_matches': np.asarray(satellite_matches, np.int),
              'particles_matched': particles_matched,
              'satellites_unmatched': np.asarray(satellites_unmatched, np.int),
              'particles_unmatched': particles_unmatched,
              'Intersection_scores': Intersection_scores}
    
    
    return results



class powder_satellite_image(object):
    """
    Powder and satellite instance predictions for a single image
    """
    
    def __init__(self, particles=None, satellites=None, matches=None):
        self.particles = particles # instance set of particles
        self.satellites = satellites # instance set of satellites
        self.matches = matches # maps satellites to their corresponding particles, see output of fast_satellite_match()
    
    def compute_matches(self):
        """
        wrapper for fast_satellite_match
        """
        self.matches = fast_satellite_match(self.particles.masks, self.satellites.masks, 
                                            self.particles.boxes, self.satellites.boxes)
        
    def visualize_particle_with_satellites(self, p_idx, ax=None):
        '''
        visualize single particle with its associated satellites
        inputs:
            :param p_idx: - index of particle mask to be plotted. Should be a key in
                                    self.matches['particle_satellite_match_idx']
            :param ax: - matplotlib axis object to visualize results on. If None, new axis
            will be created and shown by mrcnn_visualize.display_instances() function.
        '''
    
        particle_mask = self.particles.masks[:,:,[p_idx]] # maintain shape
        particle_box = self.particles.boxes[[p_idx]]
        particle_class_idx = np.zeros([1], np.int)

        s_idx = self.matches['particle_satellite_match_idx'][p_idx]
        satellite_masks = self.satellites.masks[:,:,s_idx]
        satellite_box = self.satellites.boxes[s_idx]
        satellite_class_idx = np.ones([satellite_box.shape[0]])


        masks = np.concatenate((particle_mask, satellite_masks), axis=2)
        boxes = np.concatenate((particle_box, satellite_box), axis=0)
        labels = np.concatenate((particle_class_idx, satellite_class_idx), axis=0)

        label_map = {0:'particle', 1:'satellite'}

        minbox = boxes[:,:2].min(axis=0)
        maxbox = boxes[:,2:].max(axis=0)
        

        total_box = np.concatenate((minbox, maxbox), axis=0)
        r1, c1, r2, c2 = total_box
        
        img = self.particles.img[r1:r2,c1:c2,...] 
        masks = masks[r1:r2,c1:c2,:]
        
        boxes[:,[0,2]] -= r1
        boxes[:,[1,3]] -= c1
        

        mrcnn_visualize.display_instances(img, boxes, masks, labels, label_map,ax=ax)



