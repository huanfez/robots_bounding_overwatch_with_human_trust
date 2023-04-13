#! /usr/bin/env python2

import rospy
import actionlib
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetLinkState
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellGoal, NeighborCellResult, NeighborCellFeedback

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

import cell_path_selector2 as cps2
import cell_path_selector as cps
import parameter3_mcmc as pmc3
import parameters_env_img_robot as common_parameters
import simulated_human3 as simh3
import simulated_human as simh


# read image
def avg_situational_awareness(dynamic_traversability, dynamic_visibility):
    avg_traversability = ndimage.uniform_filter(dynamic_traversability, (common_parameters.cell_height,
                                                                         common_parameters.cell_width))
    avg_visibility = ndimage.uniform_filter(dynamic_visibility, (common_parameters.cell_height,
                                                                 common_parameters.cell_width))
    return avg_traversability, avg_visibility


# discrete cell info
def gen_env(grid_height, grid_width, cell_height, cell_width, norm_traversability_r1, norm_line_of_sight_r1,
            norm_traversability_r2, norm_line_of_sight_r2, norm_traversability_r3, norm_line_of_sight_r3):
    cell_dict = {}
    obs_list = [(14, 0), (11, 3), (0, 7), (14, 7), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8), (17, 8),
                (1, 9), (4, 10), (6, 10), (7, 10), (9, 10), (12, 11), (1, 12), (3, 12), (5, 12),
                (6, 12), (8, 12), (10, 12), (14, 12), (16, 12), (0, 13), (12, 13), (0, 14),
                (4, 14), (5, 14), (10, 14), (0, 15), (2, 15), (0, 16), (6, 16), (7, 16), (0, 17),
                (3, 17), (4, 17), (17, 17), (18, 18), (12, 19), (13, 19), (14, 19), (18, 19)]

    for cy in range(0, grid_height):
        for cx in range(0, grid_width):
            index_x, index_y = cx * cell_width + int(cell_width / 2), cy * cell_height + int(cell_height / 2)
            cell_dict[(cx, cy)] = np.array(
                [[norm_traversability_r1[index_y][index_x], norm_line_of_sight_r1[index_y][index_x]],
                 [norm_traversability_r2[index_y][index_x], norm_line_of_sight_r2[index_y][index_x]],
                 [norm_traversability_r3[index_y][index_x], norm_line_of_sight_r3[index_y][index_x]]])
    return cell_dict, obs_list


def create_trust_map(grid_height=common_parameters.grid_height, grid_width=common_parameters.grid_width):
    trust_dict = {}
    for cy in range(0, grid_height):
        for cx in range(0, grid_width):
            trust_dict[(cx, cy)] = np.array([0.0, 0.0, 0.0])
    return trust_dict


def update_odom():
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_husky_base_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        husky_base_link_state = get_husky_base_link_state("/::base_link", "world")

        set_huksy_odom_state = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=10)
        huksy_reset_state = PoseWithCovarianceStamped()
        huksy_reset_state.header.frame_id = 'odom'
        huksy_reset_state.pose.pose = husky_base_link_state.link_state.pose
        set_huksy_odom_state.publish(huksy_reset_state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def reset_robot_simulation():
    rospy.wait_for_service('/gazebo/reset_world')
    try:
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset = reset_world()
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    update_odom()


def bic_score(Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, x_1toK_list, Z_1toK, y_1toK):
    mode_sigma_square1 = b0_itr / (a0_itr + 1)
    mode_sigma_square2 = d0_itr / (c0_itr + 1)
    mode_x_1toK = np.mean(x_1toK_list, axis=0)

    log_prob = 0.0
    step_num, robot_num = y_1toK.shape
    for step in range(0, step_num):
        for robot in range(0, robot_num):
            log_prob += -0.5 * np.log(2 * np.pi * mode_sigma_square2) - \
                        (y_1toK[step, robot] - mode_x_1toK[step, robot])**2 / mode_sigma_square2 / 2.0

    # bic_score = -2.0*log_prob + (len(Beta0_itr) + 2.0) * np.log(len(y_1toK))
    bic_score = -log_prob
    return bic_score


if __name__ == '__main__':
    try:
        rospy.init_node('wayofpoints', anonymous=True)

        # action client 1: squad 1 send cells for low-level to bounding overwatch
        # cell_sender_client1 = actionlib.SimpleActionClient('/server1_localcells', NeighborCellAction)
        # cell_sender_client1.wait_for_server()
        # cell0 = NeighborCellGoal()
        # cell0.in_cell_x = 0
        # cell0.in_cell_y = 0
        # cell0.to_cell_x = 1
        # cell0.to_cell_y = 0
        # cell_sender_client1.send_goal(cell0)
        # cell_sender_client1.wait_for_result()

        # Initial prior for MCMC
        f = np.zeros(len(common_parameters.candidate_cell_path_list))
        common_parameters.trust_dict = create_trust_map()
        alpha1 = np.diag([1.0, 1.0, 1.0])
        alpha2 = -np.diag([1.0, 1.0, 1.0])
        Alpha = np.concatenate((alpha1, alpha2), axis=1)
        Beta0_itr = np.asarray([0.25, 0.85, 0.5, 0.5, -0.4])
        Sigma0_itr = np.diag([1e5, 1e5, 1e5, 1e5, 1e5])
        # Beta0_itr = np.array([0.20229389,  1.02564017,  0.66432487,  0.92761691, -0.17033281])
        # Sigma0_itr = np.array([[6.20522167e-04, -3.09280518e-04, -5.07732966e-05,
        #         -6.22956600e-04, -2.56605702e-05],
        #        [-3.09280518e-04, 2.54733209e-04, 4.56512529e-05,
        #         3.07794848e-04, -4.21685867e-05],
        #        [-5.07732966e-05, 4.56512529e-05, 5.93014445e-02,
        #         -1.38626836e-02, -1.32341544e-02],
        #        [-6.22956600e-04, 3.07794848e-04, -1.38626836e-02,
        #         1.37272852e-02, 2.17522929e-03],
        #        [-2.56605702e-05, -4.21685867e-05, -1.32341544e-02,
        #         2.17522929e-03, 3.64149298e-03]])
        a0_itr = 3.0
        b0_itr = 1.0
        c0_itr = 3.0
        d0_itr = 1.0
        posterior_list_beta = []
        posterior_list_sigma = []
        trust_gains = []

        # plot posterior distribution
        fig, ax = plt.subplots(3, 2, figsize=(8,6))
        fig.tight_layout()
        x = np.linspace(norm.ppf(0.1), norm.ppf(0.99), 1000)
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[0], np.sqrt(Sigma0_itr[0, 0])), 'r-', lw=1, alpha=0.9,
                      label=r'$\beta_{-1}$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[1], np.sqrt(Sigma0_itr[1, 1])), 'g-', lw=1, alpha=0.9,
                      label=r'$\beta_0$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[2], np.sqrt(Sigma0_itr[2, 2])), 'b-', lw=1, alpha=0.9,
                      label=r'$\beta_1$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[3], np.sqrt(Sigma0_itr[3, 3])), 'y-', lw=1, alpha=0.9,
                      label=r'$\beta_2$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[4], np.sqrt(Sigma0_itr[4, 4])), 'c-', lw=1, alpha=0.9, label=r'$b$')
        ax[0, 0].legend(loc='upper right', frameon=False)
        ax[0, 0].set_ylim([1.26e-3, 1.265e-3])
        ax[0, 0].ticklabel_format(style='sci')
        # ax[0, 0].set_xlim([-400, 400])
        ax[0, 0].set_ylabel('PDF')
        ax[0, 0].set_title('Prior')

        ax[0, 0].axvline(x=common_parameters.beta_true3[0], color='r', linestyle='-.', lw=1.5)
        ax[0, 0].axvline(x=common_parameters.beta_true3[1], color='g', linestyle='-.', lw=1.5)
        ax[0, 0].axvline(x=common_parameters.beta_true3[2], color='b', linestyle='-.', lw=1.5)
        ax[0, 0].axvline(x=common_parameters.beta_true3[3], color='y', linestyle='-.', lw=1.5)
        ax[0, 0].axvline(x=common_parameters.beta_true3[4], color='c', linestyle='-.', lw=1.5)

        # Iterative trials
        for iteration in range(0, 21):
            # Read input and output data
            normalized_traversability_r1, normalized_visibility_r1 = avg_situational_awareness(
                common_parameters.r1_dynamic_traversability, common_parameters.r1_dynamic_visibility)
            normalized_traversability_r2, normalized_visibility_r2 = avg_situational_awareness(
                common_parameters.r2_dynamic_traversability, common_parameters.r2_dynamic_visibility)
            normalized_traversability_r3, normalized_visibility_r3 = avg_situational_awareness(
                common_parameters.r3_dynamic_traversability, common_parameters.r3_dynamic_visibility)

            environment_dict, obs_list = gen_env(common_parameters.grid_height, common_parameters.grid_width,
                                                 common_parameters.cell_height, common_parameters.cell_width,
                                                 normalized_traversability_r1, normalized_visibility_r1,
                                                 normalized_traversability_r2, normalized_visibility_r2,
                                                 normalized_traversability_r3, normalized_visibility_r3)

            # cell_path = common_parameters.candidate_cell_path_list[1]
            # print cell_path
            # Bayesian optimization: greedy strategy in iteration
            # Thompson sampling strategy:
            # optimo_cell_path = cps.optimal_cell_path_ucb(Beta0_itr, Sigma0_itr, common_parameters.candidate_cell_path_list,
            #                                              environment_dict)

            # probability improvement strategy:
            optimo_cell_path, f = cps2.optimal_cell_path_pi2(Beta0_itr, Sigma0_itr, common_parameters.candidate_cell_path_list,
                                                            environment_dict, f, gamma=0.0, sample_size=4000)

            print optimo_cell_path
            # exit()

            # # Send path to motion server and collect data
            # cell_index = 0
            # steps = len(optimo_cell_path)
            # while cell_index < steps - 1:
            #     cells = NeighborCellGoal()
            #     cells.in_cell_x = optimo_cell_path[cell_index][0]
            #     cells.in_cell_y = optimo_cell_path[cell_index][1]
            #     cells.to_cell_x = optimo_cell_path[cell_index + 1][0]
            #     cells.to_cell_y = optimo_cell_path[cell_index + 1][1]
            #
            #     cell_sender_client1.send_goal(cells)
            #     cell_sender_client1.wait_for_result()
            #
            #     cell_index += 1

            # Save traversability & visibility as image file: optional

            # MCMC training: update model parameters
            Z_1toK = cps2.cell_path_situational_awareness(optimo_cell_path, environment_dict)
            y_1toK = simh3.simulated_human_data(common_parameters.beta_true3, common_parameters.delta_w_true3,
                                                common_parameters.delta_v_true3, Z_1toK)
            # Z_1toK_data = Z_1toK[:, :, 1:]
            # y_1toK = simh.simulated_human_data(common_parameters.beta_true, common_parameters.delta_w_true,
            #                                    common_parameters.delta_v_true, Z_1toK_data)
            trust_gains.append(np.sum(y_1toK, axis=0))
            samples_x_1toK, samples_Beta, samples_delta_w_square, samples_delta_v_square, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, \
                c0_itr, d0_itr = pmc3.iterated_sampling(y_1toK, Z_1toK, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, Alpha, iters=7000)

            # Bayesian optimization: sample model parameters & optimal path
            means_Beta, variance_Beta, means_delta_w_square, variance_delta_w_square, means_delta_v_square, \
                variance_delta_v_square = pmc3.mean_value_model_parameters(samples_x_1toK, samples_Beta,
                                                                           samples_delta_w_square,
                                                                           samples_delta_v_square)

            # Set posterior to be prior
            Beta0_itr = means_Beta
            Sigma0_itr = variance_Beta
            a0_itr = 2 + means_delta_w_square**2 / variance_delta_w_square
            b0_itr = means_delta_w_square * (a0_itr - 1)
            c0_itr = 2 + means_delta_v_square**2 / variance_delta_v_square
            d0_itr = means_delta_v_square * (c0_itr - 1)

            posterior_list_beta.append(np.copy(Beta0_itr))
            posterior_list_sigma.append(np.copy(Sigma0_itr))

            # reset gazebo model state
            # reset_robot_simulation()

            # print (np.array(samples_Beta)).shape
            # if iteration > 18:
            #     ax[0, 2].plot((np.array(samples_Beta))[1000:, 0], linestyle=':')
            #     ax[1, 2].plot((np.array(samples_Beta))[1000:, 1], linestyle=':')
            #     ax[2, 2].plot((np.array(samples_Beta))[1000:, 2], linestyle=':')
            #     ax[3, 2].plot((np.array(samples_Beta))[1000:, 3], linestyle=':')
            if iteration < 10 and iteration % 5 == 0:
                print "beta_posterior:", Beta0_itr, Sigma0_itr
                ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, Beta0_itr[0], np.sqrt(Sigma0_itr[0, 0])), 'r-', lw=1,
                                              alpha=0.9,
                                              label=r'$\beta_{-1}$')
                ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, Beta0_itr[1], np.sqrt(Sigma0_itr[1, 1])), 'g-', lw=1,
                                              alpha=0.9,
                                              label=r'$\beta_0$')
                ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, Beta0_itr[2], np.sqrt(Sigma0_itr[2, 2])), 'b-', lw=1,
                                              alpha=0.9,
                                              label=r'$\beta_1$')
                ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, Beta0_itr[3], np.sqrt(Sigma0_itr[3, 3])), 'y-', lw=1,
                                              alpha=0.9, label=r'$\beta_2$')
                ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, Beta0_itr[4], np.sqrt(Sigma0_itr[4, 4])), 'c-', lw=1,
                                              alpha=0.9,
                                              label=r'$b$')
                ax[iteration / 5 + 1, 0].legend(loc='upper right', frameon=False)
                ax[iteration / 5 + 1, 0].set_ylabel('PDF')
                ax[iteration / 5 + 1, 0].set_title('Posterior of trial ')

                ax[iteration / 5 + 1, 0].axvline(x=common_parameters.beta_true3[0], color='r', linestyle='-.', lw=1.5)
                ax[iteration / 5 + 1, 0].axvline(x=common_parameters.beta_true3[1], color='g', linestyle='-.', lw=1.5)
                ax[iteration / 5 + 1, 0].axvline(x=common_parameters.beta_true3[2], color='b', linestyle='-.', lw=1.5)
                ax[iteration / 5 + 1, 0].axvline(x=common_parameters.beta_true3[3], color='y', linestyle='-.', lw=1.5)
                ax[iteration / 5 + 1, 0].axvline(x=common_parameters.beta_true3[4], color='c', linestyle='-.', lw=1.5)
                ax[iteration / 5 + 1, 0].set_ylim([0, 20])

            if 21 > iteration >= 10 and iteration % 5 == 0:
                ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, Beta0_itr[0], np.sqrt(Sigma0_itr[0, 0])), 'r-', lw=1,
                                                 alpha=0.9,
                                                 label=r'$\beta_{-1}$')
                ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, Beta0_itr[1], np.sqrt(Sigma0_itr[1, 1])), 'g-', lw=1,
                                                 alpha=0.9,
                                                 label=r'$\beta_0$')
                ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, Beta0_itr[2], np.sqrt(Sigma0_itr[2, 2])), 'b-', lw=1,
                                                 alpha=0.9,
                                                 label=r'$\beta_1$')
                ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, Beta0_itr[3], np.sqrt(Sigma0_itr[3, 3])), 'y-', lw=1,
                                                 alpha=0.9,
                                                 label=r'$\beta_2$')
                ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, Beta0_itr[4], np.sqrt(Sigma0_itr[4, 4])), 'c-', lw=1,
                                                 alpha=0.9, label=r'$b$')
                ax[(iteration - 10) / 5, 1].legend(loc='upper right', frameon=False)
                ax[(iteration - 10) / 5, 1].set_ylabel('PDF')
                ax[(iteration - 10) / 5, 1].set_title('Posterior of trial ')

                ax[(iteration - 10) / 5, 1].axvline(x=common_parameters.beta_true3[0], color='r', linestyle='-.',
                                                    lw=1.5)
                ax[(iteration - 10) / 5, 1].axvline(x=common_parameters.beta_true3[1], color='g', linestyle='-.',
                                                    lw=1.5)
                ax[(iteration - 10) / 5, 1].axvline(x=common_parameters.beta_true3[2], color='b', linestyle='-.',
                                                    lw=1.5)
                ax[(iteration - 10) / 5, 1].axvline(x=common_parameters.beta_true3[3], color='y', linestyle='-.',
                                                    lw=1.5)
                ax[(iteration - 10) / 5, 1].axvline(x=common_parameters.beta_true3[4], color='c', linestyle='-.',
                                                    lw=1.5)
                ax[(iteration - 10) / 5, 1].set_ylim([0, 20])

            y_1toK_pred = simh3.simulated_human_data(Beta0_itr, means_delta_w_square, means_delta_v_square, Z_1toK)

            # ax[0, 3].clear()
            # ax[1, 3].clear()
            # ax[2, 3].clear()
            # ax[0, 3].plot(y_1toK_pred[:, 0])
            # ax[0, 3].plot(y_1toK[:, 0])
            # ax[1, 3].plot(y_1toK_pred[:, 1])
            # ax[1, 3].plot(y_1toK[:, 1])
            # ax[2, 3].plot(y_1toK_pred[:, 2])
            # ax[2, 3].plot(y_1toK[:, 2])
            # plt.draw()
            plt.pause(1e-17)

            if iteration == 20:
                bic_result = bic_score(Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, samples_x_1toK,
                                       Z_1toK, y_1toK)
                print "BIC score:", bic_result
        fig.savefig('/home/i2r2020/Documents/huanfei/bo_data/sim_posterior.tif', dpi=100, bbox_inches="tight")

        print "trust change:", trust_gains
        print "overall trust gains:", np.sum(np.array(trust_gains))

         # print "posterior list beta:", posterior_list_beta, "posterior list sigma:", posterior_list_sigma
        posterior_list_beta_lb = []
        posterior_list_beta_ub = []
        length = len(posterior_list_beta)
        for index in range(0, length):
            beta_lb = posterior_list_beta[index] - 2.5 * np.sqrt(posterior_list_sigma[index].diagonal())
            beta_ub = posterior_list_beta[index] + 2.5 * np.sqrt(posterior_list_sigma[index].diagonal())
            posterior_list_beta_lb.append(beta_lb)
            posterior_list_beta_ub.append(beta_ub)
        iter_num = range(0, 21)
        list_beta = np.array([[0.1731554060144376, 0.24328366514961094], [0.9922078329268567, 1.03997759059263],
                              [-0.04439291019097563, 0.7433698250183551], [0.6465972898583757, 1.0219648551116165],
                              [-0.2361401599472543, -0.020858684577622466]])
        fig2, ax2 = plt.subplots(5, 1)
        fig2.tight_layout()
        for i in range(0, 5):
            ax2[i].fill_between(iter_num, np.array(posterior_list_beta_lb)[:, i], np.array(posterior_list_beta_ub)[:, i],
                                color='g', alpha=0.6, interpolate=True)
            # ax2[i].fill_between(iter_num, np.ones(21) * list_beta[i, 0], np.ones(21) * list_beta[i, 1],
            #                     color='r', alpha=0.7, interpolate=True)
            ax2[i].plot(iter_num, np.ones(21)*common_parameters.beta_true3[i], color='k')
            ax2[i].set_xlabel('Trial number')
            ax2[i].set_xticks(range(0,21))
        ax2[0].set_ylabel(r'$\beta_{-1}$')
        ax2[1].set_ylabel(r'$\beta_{0}$')
        ax2[2].set_ylabel(r'$\beta_{1}$')
        ax2[3].set_ylabel(r'$\beta_{2}$')
        ax2[4].set_ylabel(r'$b$')
        fig2.savefig('/home/i2r2020/Documents/huanfei/bo_data/convergence.tif', dpi=300, bbox_inches="tight")
        plt.show()
    except rospy.ROSInterruptException:
        pass
