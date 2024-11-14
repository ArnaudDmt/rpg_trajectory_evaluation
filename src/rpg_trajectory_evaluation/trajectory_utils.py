#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import random
import numpy as np
import transformations as tf
import numba


def get_rigid_body_trafo(quat, trans):
    T = tf.quaternion_matrix(quat)
    T[0:3, 3] = trans
    return T


def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:, 0:3], axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances

@numba.njit
def compute_comparison_indices_length(distances, dist, max_dist_diff, num_pairs = None):
    max_idx = len(distances)
    comparisons = []
    valid_indices = [idx for idx in range(max_idx) if distances[idx] + dist <= distances[-1]]

    # Choose random start indices from the valid range
    if num_pairs == None:
        chosen_indices = [idx for idx in range(max_idx)]
    else:
        chosen_indices = random.sample(valid_indices, min(num_pairs, len(valid_indices)))
    for idx in chosen_indices:
        best_idx = -1
        error = max_dist_diff
        for i in range(idx, max_idx):
            if np.abs(distances[i] - (distances[idx] + dist)) < error:
                best_idx = i
                error = np.abs(distances[i] - (distances[idx] + dist))
        if best_idx != -1:
            comparisons.append((idx, best_idx))

    return comparisons


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1)/2)))*180.0/np.pi
