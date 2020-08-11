import copy
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pycocotools.mask as RLE
import skimage
import skimage.io
import sys


from detectron2.structures import Instances

from .. import analyze, visualize
from ..structures import boxes_to_array, mask_areas, masks_to_rle, InstanceSet


def _rle_satellite_match(particles, satellites, match_thresh=0.5):
    """
    TODO docs
    Args:
        particles:
        satellites:
        match_thresh:

    Returns:

    """

    particles = masks_to_rle(particles)
    satellites = masks_to_rle(satellites)

    satellite_matches = []
    intersection_scores = []

    particles_matched_bool = np.zeros(len(particles), dtype=np.bool)
    satellites_unmatched = []

    for satellite_idx, satellite_mask in enumerate(satellites):

        intersects = RLE.area([RLE.merge([satellite_mask, pmask], intersect=True) for pmask in particles]) \
            / RLE.area(satellite_mask)

        iscore_amax = np.argmax(intersects)
        iscore_max = intersects[iscore_amax]

        if iscore_max > match_thresh:
            satellite_matches.append([satellite_idx, iscore_amax])
            particles_matched_bool[iscore_amax] = True
            intersection_scores.append(iscore_max)

        else:
            satellites_unmatched.append(satellite_idx)

    particles_unmatched = np.array([i for i, matched in enumerate(particles_matched_bool) if not matched], np.int)
    satellite_matches = np.asarray(satellite_matches, np.int)
    satellites_unmatched = np.asarray(satellites_unmatched, np.int)
    intersection_scores = np.asarray(intersection_scores)

    match_pairs = {x: [] for x in np.unique(satellite_matches[:, 1])}
    for match in satellite_matches:
        match_pairs[match[1]].append(match[0])


    results = {'satellite_matches': satellite_matches,
               'satellites_unmatched': satellites_unmatched,
               'particles_unmatched': particles_unmatched,
               'intersection_scores': intersection_scores,
               'match_pairs': match_pairs}

    return results


class PowderSatelliteImage(object):
    """
    Powder and satellite instance predictions for a single image.
    """
    
    def __init__(self, particles=None, satellites=None, matches=None):
        """
        Initializes the PowderSatelliteImage instance.

        Attributes
        ------------
        particles, satellites: InstanceSet or None
            InstanceSet objects containing the particle and satellite instances for the same image, respectively.

        matches: dict or None
            dictionary in the format of self.compute_matches()

        """

        self.particles = particles  # instance set of particles
        self.satellites = satellites  # instance set of satellites
        self.matches = matches  # maps satellites to their corresponding particles, see output of fast_satellite_match()
    
    def compute_matches(self, thresh=0.5):
        """
        Wrapper for rle_satellite_match. Matches satellite masks to particle masks.

        Attributes
        ----------
        matches: dict
            dictionary

        See Also
        ---------
        rle_satellite_match :

        """
        self.matches = _rle_satellite_match(self.particles.instances,
                                            self.satellites.instances, thresh)
        
    def visualize_particle_with_satellites(self, p_idx, ax=None):
        """
        visualize single particle with its associated satellites

        Parameters
        -----------
        p_idx: int
         index of particle mask to be plotted. Should be a key in self.matches['particle_satellite_match_idx']

       ax:  matplotlib axis object
            Axis on which to visualize results on. If None, new figure and axis will be created
            and shown with plt.show().
       """

        particle_mask = self.particles.instances[[p_idx]]  # maintain shape
        particle_mask = masks_to_rle(particle_mask)

        particle_box = self.particles.instances.boxes[[p_idx]]
        particle_box = boxes_to_array(particle_box)

        particle_class_idx = np.zeros([1], np.int)

        s_idx = self.matches['match_pairs'][p_idx]
        satellite_masks = self.satellites.instances[s_idx]
        satellite_masks = masks_to_rle(satellite_masks)

        satellite_box = self.satellites.instances.boxes[s_idx]
        satellite_box = boxes_to_array(satellite_box)

        satellite_class_idx = np.ones(len(satellite_box), np.int)

        masks = particle_mask + satellite_masks
        boxes = np.concatenate((particle_box, satellite_box), axis=0).astype(np.int)
        labels = np.concatenate((particle_class_idx, satellite_class_idx), axis=0)

        label_map = {'thing_classes': ['particle', 'satellite']}

        minbox = boxes[:, :2].min(axis=0)
        maxbox = boxes[:, 2:].max(axis=0)

        total_box = np.concatenate((minbox, maxbox), axis=0)
        c1, r1, c2, r2 = total_box

        img = skimage.io.imread(self.particles.filepath)
        img = skimage.color.gray2rgb(img)
        img = img[r1:r2, c1:c2]

        # need to trim masks to correct size
        masks = RLE.decode(masks)
        masks = masks[r1:r2, c1:c2, :]
        masks = RLE.encode(np.asfortranarray(masks))
        
        boxes[:, [0, 2]] -= c1
        boxes[:, [1, 3]] -= r1

        image_size = (r2-r1, c2-c1)
        instances = Instances(image_size, **{'masks': masks, 'boxes': boxes, 'class_idx': labels})
        iset = InstanceSet(instances=instances)
        iset.instances.colors = visualize.random_colors(len(iset.instances), iset.randomstate)

        visualize.quick_visualize_iset(img, label_map, iset, ax=ax)

    def compute_satellite_metrics(self):
        """
        Computes the number of satellites and number of particles containing at least one satellite in the image.

        The number of particles and number of particles with at least one satellite
        can be determined from the length of each list. The results are returned as
        arrays of mask areas so that size filtering can be applied as necessary before
        computing the final results..

        Parameters
        ------------

        Returns
        ------------
        mask_areas_all, mask_areas_matched: ndarray
             element array where each element is the area of each particle mask.
             mask_areas_all includes all particle masks in the image, and mask_areas_matched
             only includes the subset where the particle matched at least one satellite.
        results: dict
            dictionary with the following format:
            {
            n_satellites: int- total number of satellites in image
            n_particles_matched: int- total number of matched particles in image
            n_particles_all: int- total number of particles in image
            mask_areas_matched: ndarray- n_particles_matched element array of mask areas of each matched particle
            mask_areas_all: ndarray- n_particles_all element array of mask areas of all particles
            }
        """
        # psi must have these defined to do match
        assert None not in (self.particles, self.satellites, self.matches)

        # total number of satellites in image
        n_satellites = len(self.satellites.instances)

        # number of particles with at least one satellite
        matched_particle_idx = list(self.matches['match_pairs'])
        n_particles_matched = len(matched_particle_idx)

        # total number of particles in image
        n_particles_all = len(self.particles.instances)

        # areas of masks retrieved so size thresholding can be applied
        particle_masks_all = masks_to_rle(self.particles.instances.masks.rle)
        mask_areas_all = RLE.area(particle_masks_all)
        mask_areas_matched = mask_areas_all[matched_particle_idx]

        results = {'n_satellites': n_satellites,
                   'n_particles_matched': n_particles_matched,
                   'n_particles_all': n_particles_all,
                   'mask_areas_matched': mask_areas_matched,
                   'mask_areas_all': mask_areas_all}

        return results

    def copy(self):
        """
        Return copy of the PowderSatelliteImage object.

        Parameters
        -----------
        None

        Returns
        ---------
        self: PowderSatelliteImage
            Copy of the object
        """

        return copy.deepcopy(self)


def psd(particles, xvals='d_eq', yvals='cvf', c=None, distance='length', ax=None, plot=True, return_results=False):
    """
    Computes and plots the cumulative particle size distribution from segmentation masks.


    Parameters
    ----------
    particles: list of InstanceSet or PowderSatelliteImage objects, or array
        List of objects containing the masks or mask areas.

    xvals: str
        Quantity to be plotted on x-axis.
        'd_eq' for equivalent circle diameter (circle with same area as mask)
        'area' for mask areas

    yvals: str
        Quantity to be plotted on y-axis
        'cvf' for cumulative volume fraction of particles
        'counts' for cumulative fraction of number-counts of instances

    c: list, float, tuple, or None
        Conversion from pixels to units of length (pixels are assumed to be square.)
        If None, a value will be inferred from the image_size and HFW values from each element in
        *particles*, if it is defined. If a float, the same value will be used for each element in *particles*.
        Otherwise, distance metrics will be given
        in terms of pixels. If a tuple, the first element is a value(float) or list of values in the format described
        above, and the second value is the units of length per one pixel corresponding to the values (ie 'um').

    distance: str
        'pixels': mask area/d_eq/V_eq are given in pixles.
        'length': quantites are calculated in units of length.

    ax: matpltotlib axis or None
        If an axis is specified, the psd will be plotted on that axis. Otherwise,
        if *plot* == True, a new figure will be created and displayed. Otherwise,
        the psd will not be plotted.

    plot: bool
        if True, and *ax*=None, the psd will be ploted on a new figure.

    return_results: bool
        if True, the x and y values for the PSD


    Returns
    -------
    x, y: ndarray
        Optional, only returned if *return_values*==True. n-element arrays containing the
        x and y values of the psd, respectively.

    Notes
    ------
    The equivalent diameter and volume of masks are determined by the following.
    .. math::
        d_{eq} = 2 * \sqrt(A/\pi)
        V_{eq} = 4/3 * \pi * (d_{eq}/2)^3
    """
    if type(c) == tuple:  # units are specified with c
        length_units = c[1]
        c = c[0]
    else:
        length_units = ''

    # handle case where only 1 object is supplied for particles
    if type(particles) in (InstanceSet, PowderSatelliteImage):
        particles = [particles]

    # extract InstanceSet from PSI
    if type(particles[0]) == PowderSatelliteImage:
        particles = [x.particles for x in particles]

    if type(particles[0] == InstanceSet):
        areas = [mask_areas(x) for x in particles]

    # particles areas are given in lists or arrays
    elif type(particles[0]) in (np.ndarray, list):
        areas = [np.asarray(x) for x in particles]

    # if c is not manually specified, and distance is set to 'length', get conversion from pixels to length units
    if distance.lower() == 'length':
        if c is None:
            if type(particles[0]) == InstanceSet:
                if particles[0].HFW is not None:
                    HFW = [x.HFW for x in particles]
                    assert all([x is not None for x in HFW]), 'all HFW values must be specified if c is not defined'

                    # HFW should have the same units for every element in particles
                    # Otherwise, units have to be converted, which adds another layer
                    # of complexity.
                    for iset in particles:
                        assert iset.HFW_units == particles[0].HFW_units, 'all HFW values should have same units'
                    length_units = particles[0].HFW_units
                    HFW = np.asarray([x.HFW for x in particles])
                    image_widths = np.asarray([x.instances.image_size[1] for x in particles], np.int)
                    # c is the horizontal field width (length) / horizontal width (pixels)
                    c = [h/w for h, w in zip(HFW, image_widths)]

                else:
                    raise ValueError('Cannot infer c because HFW is not defined')
            else:
                raise ValueError('Cannot infer c from particles (must be list of InstanceSet or PowderSatelliteImage '
                                 'objects')

        # compute areas in terms of length squared
        if type(c) in [list, np.ndarray]:  # different images have different c values
            assert len(c) == len(areas), 'if c (or c[0] if passed as tuple) is a list or array ' \
                                         'it must have the same length as particles.'
            areas = [a_i * c_i ** 2 for a_i, c_i in zip(areas, c)]

        elif type(c) in [int, float]:
            print("a{}".format(type(areas)))
            areas = [a_i * c ** 2 for a_i in areas]

        else:
            raise ValueError('c (or c[0] if passed as tuple) must be a list, array, int, or float')

    elif distance.lower() == 'pixels':
        length_units = 'px'
        areas = mask_areas(particles)
    else:
        raise ValueError('distance must be "length" or "pixels"')

    if type(areas[0]) in (list, np.ndarray):
        areas = np.concatenate(areas, axis=0)  # concatenate all areas into single array

    unique, counts = np.unique(areas, return_counts=True)  # find all unique elements and each of their frequencies
    if xvals.lower() == 'd_eq':
        # convert mask areas to equivalent circle diameter
        # A = pi * r ** 2
        # 2 * (A / pi) ** (1 / 2) = 2 * r = d
        unique = 2 * np.sqrt(unique / np.pi)
        xlabel = 'Equivalent diameter{}'.format(', {}'.format(length_units) if length_units else '')
    elif xvals.lower() == 'area':
        xlabel = 'Mask area{}'.format('- ${}^2$'.format(length_units) if length_units else '')

    else:
        raise ValueError('xvals must be "d_eq" or "area"')

    if yvals.lower() == 'cvf':
        # convert mask areas to equivalent sphere diameters
        # V = 4/3 * pi ** (-1/2) * A ** (3/2)
        volumes = 4/3 * np.pi ** (-1/2) * unique ** (3/2)
        counts = volumes * counts  # total volume contribution of each bin
        ylabel = 'cumulative volume fraction'

    elif yvals.lower() == 'counts':
        ylabel = 'counts (cumulative)'

    else:
        raise ValueError('yvals must be "cvf" or "counts"')

    counts = counts.cumsum()  # cumulative distribution
    counts = counts / counts[-1]  # normalize by total counts or volume

    x = unique
    y = counts

    if plot or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(dpi=300)
        ax.grid(axis='both', which='both', color=(0.85, 0.85, 0.85), linewidth=1, linestyle='--')
        ax.plot(x,y, '-.k')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    if plot:
        plt.show()

    if return_results:
        return x, y

def satellite_measurements(psi, print_summary=True, output_dict=False):
    """
    Measures the satellite content of samples in *psi*.

    The measurements are as follows (both printed and returned as dictionary items):
      * n_images: int
        total number of images included in analysis
      * n_particles: int
            total number of particles,
      * n_satellites: int
            total number of matched satellites,
      * n_satellites_unmatched: int
            total number of satellites that did not match with a powder particle
      * n_satellited_particles: int
            total number of satellited particles,
      * sat_frac: float
            fraction of satellited particles
      * mspp: float
            median number of satellites per satellited particle (Median Satellites Per Particle)
    Two additional quantities will be returned as dictionary items but not printed. These can be used
    to plot the distribution of the number of satellites per particle.
      * unique: ndarray
            unique numbers of satellites per particle across all particles
      * counts: ndarray
            relative counts for each element in *unique*.


    Parameters
    -----------
    psi: list(PowderSatelliteImage)
        list of psi objects from which to compute results. The results from all objects in the list
        will be combined for the final result. If psi[i].matches is None for any item, it will be called
        with the default settings for all items (ie may be recomputed for other list items.)

    print_summary: bool
        if True, summary of the results will be printed.

    output_dict: bool
        if True, dictionary containing the results in the format mentioned above will be returned.

    Returns
    ---------
    results: dict
        Optional- only returned if output_dict==True. Dictionary of results in the above mentioned format.

    """
    # validate inputs
    if type(psi) == PowderSatelliteImage:
        psi = [psi]

    assert all([type(x) == PowderSatelliteImage for x in psi]), 'psi must be list of PowderSatelliteImage objects!'

    # get matches from psi
    matches = [x.matches for x in psi]

    # if any psi is missing matches, compute matches for all elements of psi with default settings
    if any([x == None for x in matches]):
        for x in psi:
            x.compute_matches()
        matches = [x.matches for x in psi]

    n_images = len(psi)  # total number of images included
    n_particles_matched = sum([len(x['match_pairs'].keys()) for x in matches])  # total number of matched particles
    n_particles = n_particles_matched + sum([len(x['particles_unmatched']) for x in matches])  # total number of

    spp_list = [] # satellites per particle
    for m in matches:
        for v in m['match_pairs'].values():
            spp_list.append(len(v))
    spp_list = np.asarray(spp_list)
    n_satellites_matched = sum(spp_list)  # number of matched satellitse
    mspp = np.median(spp_list)  # median number of satellites per satellited particle

    # number of unmatched satellites
    n_satellites_unmatched = sum([len(x['satellites_unmatched']) for x in matches])  # number of unmatched satellites

    sat_frac = n_particles_matched / n_particles  # ration of satellited particles

    unique, counts = np.unique(spp_list, return_counts=True)
    assert counts.sum() == n_particles_matched
    assert n_particles == sum([len(x.particles.instances) for x in psi])
    assert n_satellites_matched + n_satellites_unmatched == sum([len(x.satellites.instances) for x in psi])


    counts = counts.cumsum()/counts.sum()

    keys = ['n_images', 'n_particles', 'n_satellites', 'n_satellites_unmatched', 'n_satellited_particels',
            'sat_frac', 'mspp', 'unique_satellites_per_particle', 'counts_satellites_per_particle']

    labels = ['number of images',
              'number of particles',
              'number of matched satellites',
              'number of unmatched satellites',
              'number of satellited particles',
              'fraction of satellited particles',
              'median number of satellites per\n'
              'satellited particle             ']

    values = [n_images, n_particles, n_satellites_matched, n_satellites_unmatched, n_particles_matched,
              sat_frac, mspp, unique, counts]

    if print_summary:
        for lab, v in zip(labels, values[:-2]):
            print('{:35}\t{}'.format(lab, v))

    if output_dict:
        return dict(zip(keys, values))
