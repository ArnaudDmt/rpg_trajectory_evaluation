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
from alive_progress import alive_bar

def get_rigid_body_trafo(quat, trans):
    T = tf.quaternion_matrix(quat)
    T[0:3, 3] = trans
    return T


@numba.njit
def get_distance_from_start(gt_translation):
    n = gt_translation.shape[0]
    distances = np.empty(n)
    distances[0] = 0.0

    for i in range(1, n):
        dx = gt_translation[i, 0] - gt_translation[i - 1, 0]
        dy = gt_translation[i, 1] - gt_translation[i - 1, 1]
        dz = gt_translation[i, 2] - gt_translation[i - 1, 2]
        distances[i] = distances[i - 1] + (dx * dx + dy * dy + dz * dz) ** 0.5

    print(distances[-1])
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

    if num_pairs is not None and len(comparisons) > num_pairs:
        sampled_comparisons = []
        for i in range(num_pairs):
            sample_idx = (2 * i + 1) * len(comparisons) // (2 * num_pairs)
            sampled_comparisons.append(comparisons[sample_idx])
        return sampled_comparisons

    return comparisons


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1)/2)))*180.0/np.pi


@numba.njit
def compute_comparison_indices_ball(positions, distances, radius,
                                    snap_to_nearest, max_arc_factor):
    """
    Pair (start, end) indices by *displacement* instead of arc length.

    compute_comparison_indices_length walks the cumulative path length, so a
    base that jitters sideways while barely advancing still accumulates
    "travelled" distance and the window closes too early. Here the window from
    a given start pose is the shortest prefix whose end pose leaves the sphere
    of radius `radius` centred on that start pose -- the sub-trajectory really
    does span `radius` metres of ground.

    positions       Nx3 groundtruth translations.
    distances       cumulative arc length along the same trajectory. Only used
                    to skip ahead: arc length is an upper bound on
                    displacement, so the sphere cannot be left before the arc
                    length itself has covered `radius`.
    snap_to_nearest if true, keep whichever of the first pose outside and the
                    last pose inside sits closer to the sphere; otherwise
                    always take the first pose outside.
    max_arc_factor  give up on a start pose once its window has walked
                    max_arc_factor * radius of path without escaping. That
                    bounds the scan, and it also throws away windows where the
                    robot looped or shuffled in place instead of translating,
                    which are exactly the windows this pairing exists to
                    reject. Non-positive disables the cutoff.

    Returns the list of (start_idx, end_idx) pairs, one per usable start pose.
    """
    n = positions.shape[0]
    r2 = radius * radius
    comparisons = []

    for idx in range(n):
        # The sphere cannot be left while the path itself is shorter than its
        # radius, so start looking where the arc length first reaches it.
        lo = binary_search_left(distances, distances[idx] + radius, idx + 1, n)
        if lo >= n:
            # Not enough path left here, and even less for every later start.
            break

        if max_arc_factor > 0.0:
            arc_budget = distances[idx] + max_arc_factor * radius
        else:
            arc_budget = distances[n - 1]

        px = positions[idx, 0]
        py = positions[idx, 1]
        pz = positions[idx, 2]

        found = -1
        for k in range(lo, n):
            if distances[k] > arc_budget:
                break
            dx = positions[k, 0] - px
            dy = positions[k, 1] - py
            dz = positions[k, 2] - pz
            if dx * dx + dy * dy + dz * dz > r2:
                found = k
                break

        if found < 0:
            # Stayed inside the sphere for its whole arc budget: the robot
            # turned or marked time rather than covering ground. A later start
            # can still escape, so keep going.
            continue

        if snap_to_nearest and found - 1 > idx:
            outer = 0.0
            dx = positions[found, 0] - px
            dy = positions[found, 1] - py
            dz = positions[found, 2] - pz
            outer = (dx * dx + dy * dy + dz * dz) ** 0.5
            dx = positions[found - 1, 0] - px
            dy = positions[found - 1, 1] - py
            dz = positions[found - 1, 2] - pz
            inner = (dx * dx + dy * dy + dz * dz) ** 0.5
            if radius - inner < outer - radius:
                found = found - 1

        comparisons.append((idx, found))

    return comparisons
