#! /usr/bin/env python2

import rospy, rospkg

import numpy as np
from scipy import ndimage
import tifffile as tf
import matplotlib.pyplot as plt

import cell_path_selector as cps
import cell_path_selector2 as cps2
import parameters_env_img_robot as common_parameters


# preprocessing data: traversability & visibility (all discrete cell)
def avg_situational_awareness(dynamic_traversability, dynamic_visibility):
    avg_traversability = ndimage.uniform_filter(dynamic_traversability, (common_parameters.cell_height,
                                                                         common_parameters.cell_width))
    avg_visibility = ndimage.uniform_filter(dynamic_visibility, (common_parameters.cell_height,
                                                                 common_parameters.cell_width))
    return avg_traversability, avg_visibility


# hash_map: discrete cell with traversability & visibility info
def gen_env(traversability_r1, visibility_r1, traversability_r2, visibility_r2, traversability_r3,
            visibility_r3, grid_height=common_parameters.grid_height, grid_width=common_parameters.grid_width,
            cell_height=common_parameters.cell_height, cell_width=common_parameters.cell_width):
    cell_dict = {}
    obs_list = [(14, 0), (11, 3), (0, 7), (14, 7), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8), (17, 8),
                (1, 9), (4, 10), (6, 10), (7, 10), (9, 10), (12, 11), (1, 12), (3, 12), (5, 12),
                (6, 12), (8, 12), (10, 12), (14, 12), (16, 12), (0, 13), (12, 13), (0, 14),
                (4, 14), (5, 14), (10, 14), (0, 15), (2, 15), (0, 16), (6, 16), (7, 16), (0, 17),
                (3, 17), (4, 17), (17, 17), (18, 18), (12, 19), (13, 19), (14, 19), (18, 19)]

    norm_traversability_r1, norm_visibility_r1 = avg_situational_awareness(traversability_r1, visibility_r1)
    norm_traversability_r2, norm_visibility_r2 = avg_situational_awareness(traversability_r2, visibility_r2)
    norm_traversability_r3, norm_visibility_r3 = avg_situational_awareness(traversability_r3, visibility_r3)

    for cy in range(0, grid_height):
        for cx in range(0, grid_width):
            index_x, index_y = cx * cell_width + int(cell_width / 2), cy * cell_height + int(cell_height / 2)
            cell_dict[(cx, cy)] = np.array(
                [[norm_traversability_r1[index_y][index_x], norm_visibility_r1[index_y][index_x]],
                 [norm_traversability_r2[index_y][index_x], norm_visibility_r2[index_y][index_x]],
                 [norm_traversability_r3[index_y][index_x], norm_visibility_r3[index_y][index_x]]])
    return cell_dict, obs_list


# testing data: predicted trust with ctm1
def cell_path_trust_estimation1(Z_1toK, Beta_true):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))

    for k in range(0, K):
        for i in range(0, 3):
            if k == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_1toK_pred[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], Beta_true.T)

    xall_1toK_pred_ = np.zeros((K, 3))
    xall_1toK_pred_[1:K, :] = xall_1toK_pred[0:K - 1, :]
    y_1toK_pred = xall_1toK_pred - xall_1toK_pred_

    return y_1toK_pred


# testing data: predicted trust with ctm1
def cell_path_trust_estimation1_(Z_1toK, Beta_true, yt_1toK):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))

    xall_obs = np.zeros((K, 3))
    xall_obs[0] = yt_1toK[0]
    for k in range(1, K):
        xall_obs[k] = xall_obs[k - 1] + yt_1toK[k]

    for k in range(0, K):
        for i in range(0, 3):
            if k == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_obs[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], Beta_true.T)

    xall_1toK_pred_ = np.zeros((K, 3))
    xall_1toK_pred_[1:K, :] = xall_1toK_pred[0:K - 1, :]
    y_1toK_pred = xall_1toK_pred - xall_1toK_pred_

    return y_1toK_pred


# testing data: predicted trust with ctm2
def cell_path_trust_estimation2(Z_1toK, Beta_true):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))

    for k in range(0, K):
        for i in range(0, 3):
            if i == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_1toK_pred[k, i - 1]

            if k == 0:
                Z_1toK_pred[k, i, 1] = 0.0
            else:
                Z_1toK_pred[k, i, 1] = xall_1toK_pred[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], Beta_true.T)

    xall_1toK_pred_ = np.zeros((K, 3))
    xall_1toK_pred_[1:K, :] = xall_1toK_pred[0:K - 1, :]
    y_1toK_pred = xall_1toK_pred - xall_1toK_pred_

    return y_1toK_pred, xall_1toK_pred


# testing data: predicted trust with ctm2
def cell_path_trust_estimation2_(Z_1toK, Beta_true, yt_1toK):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))

    xall_obs = np.zeros((K, 3))
    xall_obs[0] = yt_1toK[0]
    for k in range(1, K):
        xall_obs[k] = xall_obs[k - 1] + yt_1toK[k]

    for k in range(0, K):
        for i in range(0, 3):
            if i == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_obs[k, i - 1]

            if k == 0:
                Z_1toK_pred[k, i, 1] = 0.0
            else:
                Z_1toK_pred[k, i, 1] = xall_obs[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], Beta_true.T)

    xall_1toK_pred_ = np.zeros((K, 3))
    xall_1toK_pred_[1:K, :] = xall_1toK_pred[0:K - 1, :]
    y_1toK_pred = xall_1toK_pred - xall_1toK_pred_

    return y_1toK_pred, xall_1toK_pred


# testing data: aic metric
def aic_score(Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, x_1toK_list, Z_1toK, y_1toK):
    mode_sigma_square1 = b0_itr / (a0_itr + 1)
    mode_sigma_square2 = d0_itr / (c0_itr + 1)
    mode_x_1toK = np.mean(x_1toK_list, axis=0)

    log_prob = 0.0
    step_num, robot_num = y_1toK.shape
    for step in range(0, step_num):
        for robot in range(0, robot_num):
            log_prob += -0.5 * np.log(2 * np.pi * mode_sigma_square2) - \
                        (y_1toK[step, robot] - mode_x_1toK[step, robot]) ** 2 / mode_sigma_square2 / 2.0

    aic_scr = 2.0 / len(y_1toK) * (-log_prob + (len(Beta0_itr) + 2.0))
    return aic_scr


# start the main program ##########################
rospack = rospkg.RosPack()

# list all packages, equivalent to rospack list
rospack.list()

# get the file path for rospy_tutorials
trust_pkg_dir = rospack.get_path('trust_motion_plannar')

# read observation/output data from files
trust_iter5_tif = trust_pkg_dir + '/node/data/trust_iter5.tif'
trust_iter5_data = tf.imread(trust_iter5_tif)  # read the output variable file
y_1toK = cps2.cell_path_trust_tif(common_parameters.candidate_cell_path_list[0], trust_iter5_data)  # obtain output

# read input data from files
traversability_r1 = trust_pkg_dir + '/node/data/traversability_r1.tif'
traversability_r2 = trust_pkg_dir + '/node/data/traversability_r2.tif'
traversability_r3 = trust_pkg_dir + '/node/data/traversability_r3.tif'
visibility_r1 = trust_pkg_dir + '/node/data/visibility_r1.tif'
visibility_r2 = trust_pkg_dir + '/node/data/visibility_r2.tif'
visibility_r3 = trust_pkg_dir + '/node/data/visibility_r3.tif'

r1_dynamic_traversability = tf.imread(traversability_r1)
r1_dynamic_visibility = tf.imread(visibility_r1)
r2_dynamic_traversability = tf.imread(traversability_r2)
r2_dynamic_visibility = tf.imread(visibility_r2)
r3_dynamic_traversability = tf.imread(traversability_r3)
r3_dynamic_visibility = tf.imread(visibility_r3)

# map input data into the discrete environment
environment_dict, obs_list = gen_env(r1_dynamic_traversability,
                                     r1_dynamic_visibility,
                                     r2_dynamic_traversability,
                                     r2_dynamic_visibility,
                                     r3_dynamic_traversability,
                                     r3_dynamic_visibility)

# each participant: predicted trust value
Z_1toK_m1 = cps.cell_path_situational_awareness(common_parameters.candidate_cell_path_list[0],
                                                environment_dict)  # input 1
mean_beta_m1 = np.array(
    [1.0305797946404478, 0.5194448045710597, 0.33856118878330516, 0.038225280399539194])  # ctm1 model parameters
trust_change1 = cell_path_trust_estimation1(Z_1toK_m1, mean_beta_m1)  # ctm1 predicted trust

Z_1toK_m2 = cps2.cell_path_situational_awareness(common_parameters.candidate_cell_path_list[0],
                                                 environment_dict)  # input 2
mean_beta_m2 = np.array([-0.008494066272159142, 1.0814993693365587, 0.490698474426846, 0.676094564920843, 0.05452330960200013])  # ctm2 model parameters
trust_change2, trust_all = cell_path_trust_estimation2(Z_1toK_m2, mean_beta_m2)  # ctm2 predicted trust

# plot the prediction curve for participant
fig, ax = plt.subplots(3, 1)
fig.tight_layout()
ax[0].plot(y_1toK[:, 0], linestyle='--', marker='o', color='k', label="Observation")
# ax[0].plot(trust_change1[:, 0], linestyle='--', marker='o', color='g', label="CTM1")
ax[0].plot(trust_change2[:, 0], linestyle='--', marker='o', color='r', label="Prediction")
ax[0].set_ylabel("Trust change")
# ax[0, 0].fill_between(range(0, 10), lo_trust_change1[:,0], up_trust_change1[:,0], color='g', alpha=0.3, interpolate=True)
# ax[0, 0].fill_between(range(0, 10), lo_trust_change2[:,0], up_trust_change2[:,0], color='r', alpha=0.3, interpolate=True)
ax[0].legend()
ax[0].set_ylim([-0.5, 1])
ax[0].set_title("Prediction of robot r1")

ax[1].plot(y_1toK[:, 1], linestyle='--', marker='o', color='k', label="Observation")
# ax[1].plot(trust_change1[:, 1], linestyle='--', marker='o', color='g', label="CTM1")
ax[1].plot(trust_change2[:, 1], linestyle='--', marker='o', color='r', label="Prediction")
ax[1].set_ylabel("Trust change")
# ax[1, 0].fill_between(range(0, 10), lo_trust_change2[:,1], up_trust_change2[:,1], color='r', alpha=0.3, interpolate=True)
# ax[1, 0].fill_between(range(0, 10), lo_trust_change1[:,1], up_trust_change1[:,1], color='g', alpha=0.3, interpolate=True)
ax[1].legend()
ax[1].set_ylim([-0.5, 1])
ax[1].set_title("Prediction of robot r2")

ax[2].plot(y_1toK[:, 2], linestyle='--', marker='o', color='k', label="Observation")
# ax[2].plot(trust_change1[:, 2], linestyle='--', marker='o', color='g', label="CTM1")
ax[2].plot(trust_change2[:, 2], linestyle='--', marker='o', color='r', label="Prediction")
ax[2].set_ylabel("Trust change")
# ax[2, 0].fill_between(range(0, 10), lo_trust_change2[:,2], up_trust_change2[:,2], color='r', alpha=0.3, interpolate=True)
# ax[2, 0].fill_between(range(0, 10), lo_trust_change1[:,2], up_trust_change1[:,2], color='g', alpha=0.3, interpolate=True)
ax[2].legend()
ax[2].set_ylim([-0.9, 1])
ax[2].set_title("Prediction of robot r3")

fig.savefig('/home/i2r2020/Documents/huanfei/bo_data/prediction.tif', dpi=300, bbox_inches="tight")

# plot the traversability & visibility of each robot
fig1, ax1 = plt.subplots(3, 1)
fig1.tight_layout()

expected_z_m1 = Z_1toK_m1[:, :, 1:3]

traversability = expected_z_m1[:, :, 0]
ax1[0].plot(traversability[:, 0], linestyle='--', marker='o', color='r', label="r1")
ax1[0].plot(traversability[:, 1], linestyle='--', marker='o', color='g', label="r2")
ax1[0].plot(traversability[:, 2], linestyle='--', marker='o', color='b', label="r3")
ax1[0].set_ylabel("Traversability")
ax1[0].legend()
ax1[0].set_title("Traversability of three robots")

visibility = expected_z_m1[:, :, 1]
ax1[1].plot(visibility[:, 0], linestyle='--', marker='o', color='r', label="r1")
ax1[1].plot(visibility[:, 1], linestyle='--', marker='o', color='g', label="r2")
ax1[1].plot(visibility[:, 2], linestyle='--', marker='o', color='b', label="r3")
ax1[1].set_ylabel("Visibility")
ax1[1].legend()
ax1[1].set_title("Visibility of three robots")

ax1[2].plot(trust_all[:, 0], linestyle='--', marker='o', color='r', label="r1")
ax1[2].plot(trust_all[:, 1], linestyle='--', marker='o', color='g', label="r2")
ax1[2].plot(trust_all[:, 2], linestyle='--', marker='o', color='b', label="r3")
ax1[2].set_ylabel("Trust")
ax1[2].legend()
ax1[2].set_title("Predicted trust of three robots")

fig1.savefig('/home/i2r2020/Documents/huanfei/bo_data/situational_awareness.tif', dpi=300, bbox_inches="tight")

prediction_accuracy1 = 0
prediction_accuracy2 = 0
print y_1toK[0, :]

denominator = np.zeros(len(y_1toK))
for kk in range(1, len(y_1toK)):
    diff_y_1toK = y_1toK[kk:] - y_1toK[0:-kk]
    denominator[kk] = np.sum(np.abs(diff_y_1toK)) / len(diff_y_1toK) / len(diff_y_1toK[0])

for kk in range(0, len(y_1toK) - 1):
    for ii in range(0, len(y_1toK[kk])):
        prediction_accuracy1 += np.abs(y_1toK[kk, ii] - trust_change1[kk, ii]) / denominator[kk + 1]
        prediction_accuracy2 += np.abs(y_1toK[kk, ii] - trust_change2[kk, ii]) / denominator[kk + 1]

print prediction_accuracy1 / (len(y_1toK) - 1) / len(y_1toK[0]), prediction_accuracy2 / (len(y_1toK) -1) / len(y_1toK[0])
# mse1 = np.var(y_1toK - trust_change1)
# mse2 = np.var(y_1toK - trust_change2)

# print expected_z_m1.shape, trust_change1.shape
# expected_data1 = np.concatenate((expected_z_m1, trust_change1.reshape((len(y_1toK), 3, 1))), axis=2)
# expected_cov1 = np.cov(expected_data1.reshape(3, len(y_1toK)*3))
# tr_mat1 = np.matmul(observed_cov, np.linalg.inv(expected_cov1))
# chi_square1 = (np.log(np.linalg.det(expected_cov1)) - np.log(np.linalg.det(observed_cov)) +
#                np.trace(tr_mat1) - 3.0) * len(y_1toK)

# expected_data2 = np.concatenate((expected_z_m2, trust_change2.reshape((len(y_1toK), 3, 1))), axis=2)
# expected_cov2 = np.cov(expected_data2.reshape(3, len(y_1toK)*3))
# tr_mat2 = np.matmul(observed_cov, np.linalg.inv(expected_cov2))
# chi_square2 = (np.log(np.linalg.det(expected_cov2)) - np.log(np.linalg.det(observed_cov)) +
#                np.trace(tr_mat2) - 3.0) * len(y_1toK)
#
# print "chi_squares:", chi_square1, chi_square2


plt.show()
