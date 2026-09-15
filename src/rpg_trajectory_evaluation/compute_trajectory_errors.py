#!/usr/bin/env python2

import os
import numpy as np
import math
from alive_progress import alive_it

import trajectory_utils as tu
import transformations as tf

from trajectory_utils import compute_comparison_indices_length  # Direct import

def compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist, max_dist_diff,
                           accum_distances=[], scale=1.0, num_pairs=None,
                           comparisons=None):

    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)

    # Select pairs of (start_index, end_index). Callers that pair the poses
    # some other way -- compute_relative_error_ball, for one -- hand the pairs
    # in ready-made and only borrow the error maths below.
    if comparisons is None:
        comparisons = compute_comparison_indices_length(accum_distances, dist, max_dist_diff, num_pairs)

    n_samples = len(comparisons)
    print('number of samples = {0} '.format(n_samples))
    if n_samples < 2:
        print("Too few samples! Will not compute.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([]), np.array([])

    T_mc = np.linalg.inv(T_cm)
    errors = []
    for start_idx, end_idx in alive_it(comparisons, title="Computing relative errors"):
        T_c1 = tu.get_rigid_body_trafo(q_es[start_idx, :], p_es[start_idx, :])
        T_c2 = tu.get_rigid_body_trafo(q_es[end_idx, :], p_es[end_idx, :])
        T_c1_c2 = np.dot(np.linalg.inv(T_c1), T_c2)
        T_c1_c2[:3, 3] *= scale

        T_m1 = tu.get_rigid_body_trafo(q_gt[start_idx, :], p_gt[start_idx, :])
        T_m2 = tu.get_rigid_body_trafo(q_gt[end_idx, :], p_gt[end_idx, :])
        T_m1_m2 = np.dot(np.linalg.inv(T_m1), T_m2)

        T_m1_m2_in_c1 = np.dot(T_cm, np.dot(T_m1_m2, T_mc))
        T_error_in_c2 = np.dot(np.linalg.inv(T_m1_m2_in_c1), T_c1_c2)
        T_c2_rot = np.eye(4)
        T_c2_rot[0:3, 0:3] = T_c2[0:3, 0:3]
        T_error_in_w = np.dot(T_c2_rot, np.dot(T_error_in_c2, np.linalg.inv(T_c2_rot)))
        

        # Computing the error on the tilt estimate
        e_z = np.array([0, 0, 1])
        # Extract rotation matrices
        R_ground_truth = T_m1_m2[:3, :3]  # From T_m1_m2
        R_estimated = T_c1_c2[:3, :3]    # From T_c1_c2
        # Transform ez
        ez_ground_truth = np.dot(R_ground_truth, e_z)
        ez_estimated = np.dot(R_estimated, e_z)
        # Compute the scalar product
        scalar_product = np.dot(ez_ground_truth, ez_estimated)
        tilt_error = np.arccos(scalar_product)

        errors.append([T_error_in_w, tilt_error])

    error_trans_x_y_norm = []
    error_trans_z = []
    error_trans_norm = []
    error_trans_perc = []
    error_yaw = []
    error_tilt = []
    error_gravity = []
    e_rot = []
    e_rot_deg_per_m = []
    for e in errors:
        poseError = e[0]
        tiltError = e[1]
        tn = np.linalg.norm(poseError[0:3, 3])
        error_trans_z.append(np.abs(poseError[2, 3]))
        error_trans_x_y_norm.append(np.linalg.norm(poseError[0:2, 3]))
        error_trans_norm.append(tn)
        error_trans_perc.append(tn / dist * 100)
        ypr_angles = tf.euler_from_matrix(poseError, 'rzyx')
        e_rot.append(tu.compute_angle(poseError))
        error_yaw.append(np.rad2deg(np.array(abs(ypr_angles[0]))))
        error_tilt.append(np.rad2deg(np.array(tiltError)))
        error_gravity.append(
            np.sqrt(ypr_angles[1]**2+ypr_angles[2]**2)*180.0/np.pi)
        e_rot_deg_per_m.append(e_rot[-1] / dist)

    return errors, np.array(error_trans_norm), np.array(error_trans_z), np.array(error_trans_x_y_norm),  np.array(error_trans_perc),\
        np.array(error_yaw), np.array(error_tilt), np.array(error_gravity), np.array(e_rot),\
        np.array(e_rot_deg_per_m)


def compute_absolute_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans_vec = (p_gt-p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))
    e_trans_x_y = np.linalg.norm(e_trans_vec[:, 0:2], axis=1)
    e_trans_z = e_trans_vec[:,2]

    # orientation error
    e_rot = np.zeros((len(e_trans,)))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    e_tilt = np.zeros((len(e_trans,)))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = tf.matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = tf.matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = np.rad2deg(np.array(tf.euler_from_matrix(e_R, 'rzyx')))
        e_rot[i] = np.rad2deg(np.linalg.norm(tf.logmap_so3(e_R[:3, :3])))

        # Computing the error on the tilt estimate
        e_z = np.array([0, 0, 1])
        R_ground_truth = R_wg[:3, :3]  # From T_m1_m2
        R_estimated = R_we[:3, :3]    # From T_c1_c2
        # Transform ez
        ez_ground_truth = np.dot(R_ground_truth, e_z)
        ez_estimated = np.dot(R_estimated, e_z)
        # Compute the scalar product
        scalar_product = np.dot(ez_ground_truth, ez_estimated)

        e_tilt[i] = np.rad2deg(np.arccos(scalar_product))

        

    # scale drift
    motion_gt = np.diff(p_gt, 0)
    motion_es = np.diff(p_es_aligned, 0)
    dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
    dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
    e_scale_perc = np.abs((np.divide(dist_es, dist_gt)-1.0) * 100)

    return e_trans, e_trans_x_y, e_trans_z, e_trans_vec, e_rot, e_ypr, e_tilt, e_scale_perc



def subsample_comparisons(comparisons, num_pairs):
    """Thin `comparisons` down to num_pairs evenly spread pairs.

    Same rule compute_comparison_indices_length applies internally, pulled out
    so pairings built elsewhere can be thinned the same way.
    """
    if num_pairs is None or len(comparisons) <= num_pairs:
        return comparisons
    return [comparisons[(2 * i + 1) * len(comparisons) // (2 * num_pairs)]
            for i in range(num_pairs)]


def compute_relative_error_ball(p_es, q_es, p_gt, q_gt, T_cm, dist,
                                accum_distances=[], scale=1.0, num_pairs=None,
                                snap_to_nearest=True, max_arc_factor=5.0):
    """Relative error over sub-trajectories of `dist` metres of *displacement*.

    Same errors as compute_relative_error, different windows: each one runs
    from a start pose to the first pose that leaves the sphere of radius
    `dist` around it, so the base's small oscillations no longer count towards
    the length of the window. There is no max_dist_diff -- the crossing
    defines the window, and the residual overshoot is one sample of motion.
    """
    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)

    comparisons = tu.compute_comparison_indices_ball(
        np.ascontiguousarray(p_gt, dtype=np.float64),
        np.ascontiguousarray(accum_distances, dtype=np.float64),
        float(dist), bool(snap_to_nearest), float(max_arc_factor))
    comparisons = subsample_comparisons(list(comparisons), num_pairs)

    return compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist, -1.0,
                                  accum_distances, scale,
                                  comparisons=comparisons)


def _quaternions_to_matrices(q):
    """Batched tf.quaternion_matrix: (N,4) of (x, y, z, w) -> (N,3,3).

    Same normalisation and same term-by-term expansion as the scalar version,
    so the two agree to floating-point noise.
    """
    q = np.asarray(q, dtype=np.float64)
    nq = np.sum(q * q, axis=1)
    q = q * np.sqrt(2.0 / np.where(nq < 1e-12, 1.0, nq))[:, None]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1.0 - y * y - z * z
    R[:, 0, 1] = x * y - z * w
    R[:, 0, 2] = x * z + y * w
    R[:, 1, 0] = x * y + z * w
    R[:, 1, 1] = 1.0 - x * x - z * z
    R[:, 1, 2] = y * z - x * w
    R[:, 2, 0] = x * z - y * w
    R[:, 2, 1] = y * z + x * w
    R[:, 2, 2] = 1.0 - x * x - y * y
    # A degenerate quaternion falls back to the identity, as tf does.
    R[nq < 1e-12] = np.eye(3)
    return R


def _euler_rzyx(R):
    """Batched tf.euler_from_matrix(., 'rzyx') -> (N,3) of (yaw, pitch, roll)."""
    cy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    regular = cy > 1e-8
    yaw = np.where(regular, np.arctan2(R[:, 1, 0], R[:, 0, 0]), 0.0)
    pitch = np.arctan2(-R[:, 2, 0], cy)
    roll = np.where(regular,
                    np.arctan2(R[:, 2, 1], R[:, 2, 2]),
                    np.arctan2(-R[:, 1, 2], R[:, 1, 1]))
    return np.column_stack((yaw, pitch, roll))


def compute_relative_error_batched(p_es, q_es, p_gt, q_gt, T_cm, dist,
                                   max_dist_diff, accum_distances=[], scale=1.0,
                                   num_pairs=None, comparisons=None):
    """Vectorised twin of compute_relative_error -- same maths, no Python loop.

    The scalar version spends nearly all of its time on interpreter and numpy
    call overhead around 4x4 matrices, not on arithmetic; doing every pair at
    once removes that. Returns the same tuple, except that the leading
    `errors` element is an (N,4,4) array rather than a list of pairs.
    """
    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)
    if comparisons is None:
        comparisons = compute_comparison_indices_length(accum_distances, dist, max_dist_diff, num_pairs)

    comparisons = np.asarray(comparisons, dtype=np.int64)
    if comparisons.shape[0] < 2:
        print("Too few samples! Will not compute.")
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty, empty, empty, empty, empty
    start, end = comparisons[:, 0], comparisons[:, 1]
    print('number of samples = {0} '.format(len(start)))

    def relative(p, q, a, b):
        """Pose of b expressed in the frame of a, for every pair at once."""
        Ra = _quaternions_to_matrices(q[a, :])
        Rb = _quaternions_to_matrices(q[b, :])
        R = np.einsum('nki,nkj->nij', Ra, Rb)              # Ra^T Rb
        t = np.einsum('nki,nk->ni', Ra, p[b, :] - p[a, :])  # Ra^T (pb - pa)
        return R, t, Rb

    R_c, t_c, R_c2 = relative(p_es, q_es, start, end)
    t_c = t_c * scale
    R_m, t_m, _ = relative(p_gt, q_gt, start, end)

    # Express the groundtruth motion in the estimate's frame. T_cm is the
    # identity for every caller here, but keep the general form.
    C = np.asarray(T_cm)[:3, :3]
    c = np.asarray(T_cm)[:3, 3]
    R_m_in_c = C @ R_m @ C.T
    t_m_in_c = np.einsum('ij,nj->ni', C, t_m) - np.einsum('nij,j->ni', R_m_in_c, c) + c

    # error = inv(groundtruth motion) * estimated motion
    R_e = np.einsum('nki,nkj->nij', R_m_in_c, R_c)
    t_e = np.einsum('nki,nk->ni', R_m_in_c, t_c - t_m_in_c)

    # ...then rotated into the world frame by the end pose's orientation
    R_w = np.einsum('nik,nkl,njl->nij', R_c2, R_e, R_c2)
    t_w = np.einsum('nik,nk->ni', R_c2, t_e)

    # tilt: angle between the two motions' transported vertical, which is
    # just the third column of each rotation
    tilt = np.arccos(np.clip(np.sum(R_m[:, :, 2] * R_c[:, :, 2], axis=1), -1.0, 1.0))

    ypr = _euler_rzyx(R_w)
    trace = np.trace(R_w, axis1=1, axis2=2)
    e_rot = np.rad2deg(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    error_trans_norm = np.linalg.norm(t_w, axis=1)

    poses = np.zeros((len(start), 4, 4))
    poses[:, :3, :3] = R_w
    poses[:, :3, 3] = t_w
    poses[:, 3, 3] = 1.0

    return (poses, error_trans_norm, np.abs(t_w[:, 2]),
            np.linalg.norm(t_w[:, :2], axis=1), error_trans_norm / dist * 100.0,
            np.rad2deg(np.abs(ypr[:, 0])), np.rad2deg(tilt),
            np.rad2deg(np.sqrt(ypr[:, 1] ** 2 + ypr[:, 2] ** 2)),
            e_rot, e_rot / dist)
