#! usr/bin/env python2

import numpy as np
from scipy.stats import multivariate_normal, invgamma
from numpy.linalg import inv


# Kalman filter
def kalman_filter2(x_k1_k1, Phi, Z_k, Gamma, y_k, A, P_k1_k1, Q, R):
    """
    Parameters: x_k1_k1: ndarray  -  state estimation
                Phi: ndarray - dynamic matrix of state
                Z_k: ndarray
                Gamma: ndarray
                y_k: ndarray
                A: ndarray
                P_k1_k1: ndarray
                Q: ndarray
                R: ndarray

    Returns:    x_k_k: ndarray
                P_k_k: ndarray
    """
    x_k_k1 = np.matmul(Phi, x_k1_k1) + np.matmul(Gamma, Z_k)
    P_k_k1 = np.matmul(np.matmul(Phi, P_k1_k1), Phi.T) + Q

    Sigma_k = np.matmul(np.matmul(A, P_k_k1), A.T) + R
    K_k = np.matmul(np.matmul(P_k_k1, A.T), np.linalg.inv(Sigma_k))

    x_k_k = x_k_k1 + np.matmul(K_k, (y_k - np.matmul(A, x_k_k1)))
    P_k_k = P_k_k1 - np.matmul(np.matmul(K_k, A), P_k_k1)

    return x_k_k, P_k_k


# Forward filtering backward sampling
def forward_filter_backward_sample(xall_1toK, Z_flatten_k, Phi, Gamma, Q, y_1toK, A, R, x0_mean, x0_var):
    """
    xall_1toK,
    Z_flatten_k,
    Phi,
    Gamma,
    Q,
    y_1toK,
    A,
    R,
    x0_mean,
    x0_var
    """
    K = len(y_1toK)

    smoothed_xall_1toK = np.zeros((K, 6))
    smoothed_var_xall_1toK = np.zeros((K, 6, 6))
    # Be careful about the dimension and tranposition
    for k in range(0, K):
        if k == 0:
            x_k_k, P_k_k = kalman_filter2(np.array([x0_mean]).T, Phi, np.array([Z_flatten_k[k]]).T, Gamma,
                                          np.array([y_1toK[k]]).T, A, x0_var, Q, R)
        else:
            x_k_k, P_k_k = kalman_filter2(np.array([smoothed_xall_1toK[k - 1]]).T, Phi, np.array([Z_flatten_k[k]]).T,
                                          Gamma,
                                          np.array([y_1toK[k]]).T, A, smoothed_var_xall_1toK[k - 1], Q, R)
        smoothed_xall_1toK[k] = np.copy(x_k_k.T)
        smoothed_var_xall_1toK[k] = np.copy(P_k_k)

    x_1toK_s = np.zeros((K, 6))
    x_1toK_s[K - 1] = np.random.multivariate_normal(smoothed_xall_1toK[K - 1], smoothed_var_xall_1toK[K - 1])
    for k in range(K - 2, -1, -1):
        x_k2_k = np.matmul(Phi, np.array([smoothed_xall_1toK[k]]).T) + np.matmul(Gamma, np.array([Z_flatten_k[k]]).T)
        P_k2_k = np.matmul(np.matmul(Phi, smoothed_var_xall_1toK[k]), Phi.T) + Q
        # print("P_k2_k", P_k2_k, Xi_square)

        J_k = np.matmul(np.matmul(smoothed_var_xall_1toK[k], Phi.T), inv(P_k2_k))
        mean_x_k_s = np.array([smoothed_xall_1toK[k]]).T + np.matmul(J_k, np.array([x_1toK_s[k + 1]]).T - x_k2_k)
        var_x_k_s = smoothed_var_xall_1toK[k] - np.matmul(np.matmul(J_k, P_k2_k), J_k.T)
        x_1toK_s[k] = np.random.multivariate_normal(mean_x_k_s.flatten(), var_x_k_s)

    return x_1toK_s[:, 0:3]


# Predict x value
def predict_xall(Z_1toK, Beta):
    """
    # note Z_1toK = np.random.rand(K, 3, 4)
    # Z_1toK[:, :, 3] = 1.0
    """
    Z_tilde_1toK = np.copy(Z_1toK)
    data_dimension = Z_tilde_1toK.shape
    K = data_dimension[0]  # steps
    I = data_dimension[1]  # robot number

    xall_0 = np.zeros((1, I))
    xall_1toK = np.zeros((K, I))
    for k in range(0, K):
        for i in range(0, I):
            if k == 0:
                Z_tilde_1toK[k, i, 0] = xall_0[0, i]
            else:
                Z_tilde_1toK[k, i, 0] = xall_1toK[k - 1, i]

            xall_1toK[k, i] = np.matmul(Z_tilde_1toK[k, i], Beta)

    return xall_1toK


# The first iteration
def first_iteration_mcmc(Z_1toK, Beta0=np.asarray([0.25, 0.25, 0.25, 0.25]),
                         Sigma0=np.diag([1e5, 1e5, 1e5, 1e5]), a0=2.0, b0=1.0,
                         c0=2.0, d0=1.0):
    """
    # state: linear regression estimator - betas ~ normal distribution

    # state: residue - epsilon_w ~ zero-mean normal with varicance xi_square
    # xi_square ~ IG(a0_t, b0_t)

    # Observation: residue - epsilon_v ~ zero-mean normal with varicance xi2_square
    # xi_square ~ IG(a0_t, b0_t)
    """

    # Beta: multivariate normal
    Beta_s = multivariate_normal.rvs(mean=Beta0, cov=Sigma0, size=1, random_state=None)

    # residue: inverse gamma
    delta_w_square_s = invgamma.rvs(a0, 0.0, b0)  # initial sampling

    # Observation residue: inverse gamma
    delta_v_square_s = invgamma.rvs(c0, 0.0, d0)  # initial sampling

    xall_1toK_s = predict_xall(Z_1toK, Beta_s)

    return Beta_s, delta_w_square_s, delta_v_square_s, xall_1toK_s


# sample latent variable
def sample_xall(y_1toK, Z_1toK, xall_1toK_s, Beta_s, xi_square_s, Alpha, xi2_square_s,
                xall_0, xstate_all_0_mean, xstate_all_0_variance):
    xall_0toK1_s = np.concatenate((xall_0, xall_1toK_s[0:-1]), axis=0)  # x^0:k-1 data
    xstate_all_1toK_s = np.concatenate((xall_1toK_s, xall_0toK1_s), axis=1)  # state space's states

    K = len(y_1toK)
    Z_flatten_1toK = (Z_1toK[:, :, 1:]).reshape(K, 9)  # state space's inputs

    # state space model: parameter matrix
    Phi11 = np.diag([Beta_s[0], Beta_s[0], Beta_s[0]])
    Phi1 = np.concatenate((Phi11, np.zeros((3, 3))), axis=1)
    Phi2 = np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)
    Phi = np.concatenate((Phi1, Phi2), axis=0)
    Gamma0 = np.concatenate((Beta_s[1:], np.zeros(6)))
    Gamma1 = np.concatenate((np.zeros(3), Beta_s[1:], np.zeros(3)))
    Gamma2 = np.concatenate((np.zeros(6), Beta_s[1:]))
    Gamma012 = np.array([Gamma0, Gamma1, Gamma2])
    Gamma345 = np.zeros((3, 9))
    Gamma = np.concatenate((Gamma012, Gamma345), axis=0)
    Q = np.diag([xi_square_s, xi_square_s, xi_square_s, 0.0, 0.0, 0.0])
    R = np.diag([xi2_square_s, xi2_square_s, xi2_square_s])

    # forward filtering backward sampling
    xall_1toK_s = forward_filter_backward_sample(xstate_all_1toK_s, Z_flatten_1toK,
                                                 Phi, Gamma, Q, y_1toK, Alpha, R,
                                                 xstate_all_0_mean, xstate_all_0_variance)
    return xall_1toK_s


# sample_Beta
def sample_Beta(Z_tilde_1toK_s_flatten, xall_1toK_s_flatten, delta_w_square_s, Beta0, Sigma0):
    V = inv(inv(Sigma0) + np.matmul(Z_tilde_1toK_s_flatten.T, Z_tilde_1toK_s_flatten) / delta_w_square_s)  # posterior
    E_ = np.matmul(inv(Sigma0), Beta0) + np.matmul(Z_tilde_1toK_s_flatten.T, xall_1toK_s_flatten) / delta_w_square_s
    E = np.matmul(V, E_)  # posterior
    Beta_s = np.random.multivariate_normal(E.flatten(), V, size=None)  # Gibbs sampling
    return Beta_s


def sample_delta_w_square(Z_tilde_1toK_s_flatten, xall_1toK_s_flatten, Beta_s, a0, b0):
    K3 = len(Z_tilde_1toK_s_flatten)
    error_1toK_s = xall_1toK_s_flatten - np.matmul(Z_tilde_1toK_s_flatten, Beta_s)  # vector operation
    a = a0 + K3 / 2.0  # posterior
    b = b0 + np.matmul(error_1toK_s, error_1toK_s) / 2.0  # posterior
    delta_w_square_s = invgamma.rvs(a, 0.0, b)
    return delta_w_square_s


def sample_delta_v_square(y_1toK, xall_1toK_s, xall_0, Alpha, c0, d0):
    K = len(y_1toK)
    xall_0toK1_s = np.concatenate((xall_0, xall_1toK_s[0:-1]), axis=0)
    xstate_all_1toK_s = np.concatenate((xall_1toK_s, xall_0toK1_s), axis=1)
    error2_1toK_s = y_1toK - np.matmul(Alpha, xstate_all_1toK_s.T).T  # vector operation
    c = c0 + 3.0 * K / 2.0  # posterior
    d = d0 + np.matmul(error2_1toK_s.flatten(order='F'), error2_1toK_s.flatten(order='F')) / 2.0  # posterior
    delta_v_square_s = invgamma.rvs(c, 0.0, d)
    return delta_v_square_s


def iterated_sampling(y_1toK, Z_1toK, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, Alpha, iters=5000):
    """
    Generate source data:
        input variable Z1_K, Z2_k, Z3_k
        hidden state x
        observation y
        weight of linear regression $betas$
        residue variance of linear regression $xi$
    """
    # data set
    samples_x_1toK = []
    samples_Beta = []
    samples_delta_w_square = []
    samples_delta_v_square = []

    # rearrange data set z
    Z_tilde_1toK_s = np.copy(Z_1toK)
    Z_tilde_1toK_s[:, :, 0] = 0.0
    Z_tilde_1toK_s[:, :, 3] = 1.0

    # prepare initial data xall_0 and xstate_all_0
    xall_0_mean = np.array([0.0, 0.0, 0.0])
    xstate_all_0_mean = np.concatenate((xall_0_mean, xall_0_mean), axis=0)
    xstate_all_0_variance = np.zeros((6, 6))
    xall_0 = np.zeros((1, 3))

    # Start a 1st iteration
    Beta_s, delta_w_square_s, delta_v_square_s, xall_1toK_s = first_iteration_mcmc(
        Z_1toK, Beta0=Beta0_itr, Sigma0=Sigma0_itr, a0=a0_itr, b0=b0_itr, c0=c0_itr, d0=d0_itr)

    # Start the iterated sampling
    for itr in range(0, iters):
        # x (latent): normal
        xall_1toK_s = sample_xall(y_1toK, Z_1toK, xall_1toK_s, Beta_s, delta_w_square_s, Alpha, delta_v_square_s,
                                  xall_0, xstate_all_0_mean, xstate_all_0_variance)
        samples_x_1toK.append(np.copy(xall_1toK_s))

        # update z_tilde
        K = len(y_1toK)
        for k in range(K - 1, -1, -1):
            for i in range(0, 3):
                if k == 0:
                    Z_tilde_1toK_s[k, i, 0] = 0.0  # need to carefully tune
                else:
                    Z_tilde_1toK_s[k, i, 0] = xall_1toK_s[k - 1, i]

        # Reshape Z_tilde and x_all
        Z_tilde_1toK_s_flatten = np.concatenate((Z_tilde_1toK_s[:, 0, :], Z_tilde_1toK_s[:, 1, :],
                                                 Z_tilde_1toK_s[:, 2, :]))
        xall_1toK_s_flatten = xall_1toK_s.flatten(order='F')

        # Beta: multivariate normal
        Beta_s = sample_Beta(Z_tilde_1toK_s_flatten, xall_1toK_s_flatten, delta_w_square_s,
                             Beta0=Beta0_itr, Sigma0=Sigma0_itr)
        samples_Beta.append(np.copy(Beta_s))

        # residue: inverse gamma
        delta_w_square_s = sample_delta_w_square(Z_tilde_1toK_s_flatten, xall_1toK_s_flatten, Beta_s,
                                                 a0=a0_itr, b0=b0_itr)
        samples_delta_w_square.append(np.copy(delta_w_square_s))

        # Observation: residue 2
        delta_v_square_s = sample_delta_v_square(y_1toK, xall_1toK_s, xall_0, Alpha, c0=c0_itr, d0=d0_itr)
        samples_delta_v_square.append(np.copy(delta_v_square_s))

    return samples_x_1toK, samples_Beta, samples_delta_w_square, samples_delta_v_square


def mean_value_model_parameters(samples_x_1toK, samples_Beta, samples_delta_w_square, samples_delta_v_square):
    # traces of posterior Beta
    means_Beta = np.mean(samples_Beta[1000:], axis=0)
    variance_Beta = np.cov(np.asarray(samples_Beta[1000:]).T)
    print "posterior of beta:"
    for jj in range(0, 4):
        print "mean:", means_Beta[jj], "variance:", variance_Beta[jj][jj], "creditable interval:", means_Beta[jj] - \
                                                                                               2.5 * np.sqrt(
            variance_Beta[jj][jj]), means_Beta[jj] + 2.5 * np.sqrt(variance_Beta[jj][jj])

    # sampled posterior mean of X
    means_x_1toK = np.mean(samples_x_1toK, axis=0)

    # traces of posterior delta_w_square
    means_delta_w_square = np.mean(samples_delta_w_square[1000:], axis=0)
    variance_delta_w_square = np.var(samples_delta_w_square[1000:], axis=0)
    print "posterior of sigma_square:"
    print "mean:", means_delta_w_square, "variance:", variance_delta_w_square, "creditable interval:", \
        means_delta_w_square - 2.5 * np.sqrt(variance_delta_w_square), means_delta_w_square + \
                                                                       2.5 * np.sqrt(variance_delta_w_square)

    # traces of posterior delta_v_square
    means_delta_v_square = np.mean(samples_delta_v_square[1000:], axis=0)
    variance_delta_v_square = np.var(samples_delta_v_square[1000:], axis=0)
    print "posterior of sigma_square:"
    print "mean:", means_delta_v_square, "variance:", variance_delta_v_square, "creditable interval:", \
        means_delta_v_square - 2.5 * np.sqrt(variance_delta_v_square), means_delta_v_square + \
                                                                       2.5 * np.sqrt(variance_delta_v_square)

    return means_Beta, variance_Beta, means_delta_w_square, variance_delta_w_square, means_delta_v_square, \
           variance_delta_v_square
