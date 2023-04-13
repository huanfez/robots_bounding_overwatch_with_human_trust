#! /usr/bin/env python2

import numpy as np
from scipy.stats import norm, multivariate_normal


def cell_path_trust_estimation(Z_1toK, Beta_true, delta_w_true):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))
    epislon_w = multivariate_normal.rvs(mean=np.zeros(3), cov=delta_w_true, size=K, random_state=None)

    for k in range(0, K):
        for i in range(0, 3):
            if i == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_1toK_pred[k, i-1]

            if k == 0:
                Z_1toK_pred[k, i, 1] = 0.0
            else:
                Z_1toK_pred[k, i, 1] = xall_1toK_pred[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], Beta_true.T) + epislon_w[k, i]

    return xall_1toK_pred


def simulated_human_data(beta_true, delta_w_true, delta_v_true, Z_1toK):
    trust_all_robots = cell_path_trust_estimation(Z_1toK, beta_true, delta_w_true)

    K = len(Z_1toK)
    trust_all_robots_ = np.zeros((K, 3))
    trust_all_robots_[1:K, :] = trust_all_robots[0:K - 1, :]
    epislon_v = norm.rvs(loc=0, scale=np.sqrt(delta_v_true), size=(K, 3))
    y_1toK = trust_all_robots - trust_all_robots_ + epislon_v

    return y_1toK
