#!/usr/bin/env python2

import os
import numpy as np
import math

import trajectory_utils as tu
import transformations as tf

from trajectory_utils import compute_comparison_indices_length  # Direct import

def compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist, max_dist_diff,
                           accum_distances=[], scale=1.0, num_pairs=None):

    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)
    
    # Use the updated function to get random pairs of (start_index, end_index)
    comparisons = compute_comparison_indices_length(accum_distances, dist, max_dist_diff, num_pairs)

    n_samples = len(comparisons)
    print('number of samples = {0} '.format(n_samples))
    if n_samples < 2:
        print("Too few samples! Will not compute.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([]), np.array([])

    T_mc = np.linalg.inv(T_cm)
    errors = []
    for (start_idx, end_idx) in comparisons:
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

