#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import random
import numpy as np
import transformations as tf
import numba
import bisect

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
def binary_search_left(arr, target, lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

@numba.njit
def compute_comparison_indices_length(distances, dist, max_dist_diff, num_pairs=None):
    max_idx = len(distances)
    comparisons = []

    for idx in range(max_idx):
        if distances[idx] + dist > distances[-1]:
            continue

        target = distances[idx] + dist
        insert_idx = binary_search_left(distances, target, idx + 1, max_idx)

        best_idx = -1
        min_error = max_dist_diff

        # Check candidate points near insert_idx
        if insert_idx < max_idx:
            err = abs(distances[insert_idx] - target)
            if err < min_error:
                min_error = err
                best_idx = insert_idx

        if insert_idx - 1 > idx:
            err = abs(distances[insert_idx - 1] - target)
            if err < min_error:
                min_error = err
                best_idx = insert_idx - 1

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
