#! /usr/bin/env python2

import numpy as np


def cell_path_trust(cell_path, trust_dict):
    K = len(cell_path)
    y_1toK = np.ones((K, 3))
    k = 0
    for cell in cell_path:
        y_1toK[k, :] = np.copy(trust_dict[cell])
        k = k + 1

    return y_1toK


def cell_path_trust_tif(cell_path, trust_arr):
    K = len(cell_path)
    y_1toK = np.ones((K, 3))
    k = 0
    for cell in cell_path:
        y_1toK[k, :] = np.copy(trust_arr[cell[1], cell[0]])
        k = k + 1

    return y_1toK


def cell_path_situational_awareness(cell_path, environment_dict):
    K = len(cell_path)
    Z_1toK = np.ones((K, 3, 4))
    k = 0
    for cell in cell_path:
        Z_1toK[k, :, 1:3] = np.copy(environment_dict[cell])
        k = k + 1

    return Z_1toK


def cell_path_trust_prediction_mode1(Z_1toK, means_Beta):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))
    for k in range(0, K):
        for i in range(0, 3):
            if k == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_1toK_pred[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], means_Beta.T)

    return xall_1toK_pred


# def average_trust_path(betas, Sigma, cell_path, environment_dict):
#
#     Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
#     trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, betas)
#     print "environment info:",  Z_1toK
#     print "trust of all robots:", trust_all_robots
#     average_trust = np.mean(trust_all_robots[:, 0])
#
#     return average_trust

# Greedy strategy
def optimal_cell_path(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, betas)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.mean(trust_all_robots)
        average_trust_list.append(average_trust)

    # print average_trust_list
    index = np.argmax(average_trust_list)

    return cell_path_list[index]


# Thompson sampling
def optimal_cell_path_ts(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    Beta_ts = np.random.multivariate_normal(betas, Sigma)
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_ts)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.mean(trust_all_robots)
        average_trust_list.append(average_trust)

    print average_trust_list
    index = np.argmax(average_trust_list)

    return cell_path_list[index]


# Thompson sampling
def optimal_cell_path_ucb(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    Beta_ucb = np.asarray([betas[0] - 2.0 * np.sqrt(Sigma[0][0]), betas[1] - 2.0 * np.sqrt(Sigma[1][1]),
                           betas[2] - 2.0 * np.sqrt(Sigma[2][2]), betas[3] - 2.0 * np.sqrt(Sigma[3][3])])
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_ucb)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.mean(trust_all_robots)
        average_trust_list.append(average_trust)

    print average_trust_list
    index = np.argmax(average_trust_list)

    return cell_path_list[index]


# Decision-field theory based probability improvement
def optimal_cell_path_pi(betas, Sigma, cell_path_list, environment_dict, f, gamma=0.0, sample_size=2000):
    J = len(cell_path_list)
    alpha = np.zeros(J)
    f_all = np.zeros((sample_size, J))
    sampled_betas = np.random.multivariate_normal(betas, Sigma, sample_size)

    sequence = 0
    for Beta_pi in sampled_betas:
        average_trust_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_pi)
            # print "environment info:", Z_1toK
            # print "trust of all robots:", trust_all_robots
            average_trust = np.mean(trust_all_robots)
            average_trust_list.append(average_trust)

        for j in range(0, J):
            delta_trust_j = (average_trust_list[j] * J - np.sum(average_trust_list)) / (J - 1)
            # print "delta_trust_j:", delta_trust_j
            f_all[sequence, j] = gamma * f[j] + delta_trust_j
            alpha[j] += f_all[sequence, j] / float(sample_size)

        sequence += 1

    print "alpha:", alpha, sequence
    index = np.argmax(alpha)

    return cell_path_list[index], np.mean(f_all, axis=0)