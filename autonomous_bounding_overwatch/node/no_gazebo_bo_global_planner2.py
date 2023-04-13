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
import parameter2_mcmc as pmc2
import parameters_env_img_robot as common_parameters
import simulated_human2 as simh2


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
        f = np.zeros(4)
        common_parameters.trust_dict = create_trust_map()
        alpha1 = np.diag([1.0, 1.0, 1.0])
        alpha2 = -np.diag([1.0, 1.0, 1.0])
        Alpha = np.concatenate((alpha1, alpha2), axis=1)
        Beta0_itr = np.asarray([0.5, 0.25, 0.25, 0.25, 0.25])
        Sigma0_itr = np.diag([1e-4, 1e6, 1e6, 1e6, 1e6])
        a0_itr = 4.0
        b0_itr = np.diag([1.0, 1.0, 1.0])
        c0_itr = 3.0
        d0_itr = 1.0
        fig, ax = plt.subplots(4, 2)
        # Iterative trials
        for iteration in range(0, 20):
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
            optimo_cell_path, f = cps2.optimal_cell_path_pi(Beta0_itr, Sigma0_itr, common_parameters.candidate_cell_path_list,
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
            # betas = np.array([1.0, 0.10, -0.800, -0.10])
            # y_1toK = cps.cell_path_trust(optimo_cell_path, common_parameters.trust_dict)
            Z_1toK = cps2.cell_path_situational_awareness(optimo_cell_path, environment_dict)
            y_1toK = simh2.simulated_human_data(common_parameters.beta_true2, common_parameters.delta_w_true2,
                                                common_parameters.delta_v_true2, Z_1toK)
            samples_x_1toK, samples_Beta, samples_delta_w_square, samples_delta_v_square, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, \
                c0_itr, d0_itr = pmc2.iterated_sampling(y_1toK, Z_1toK, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, Alpha, iters=7000)

            # Bayesian optimization: sample model parameters & optimal path
            means_Beta, variance_Beta, means_delta_w_square, variance_delta_w_square, means_delta_v_square, \
                variance_delta_v_square = pmc2.mean_value_model_parameters(samples_x_1toK, samples_Beta,
                                                                           samples_delta_w_square,
                                                                           samples_delta_v_square)

            # Set posterior to be prior
            # Beta0_itr = means_Beta
            # Sigma0_itr = variance_Beta
            # a0_itr = 2 + means_delta_w_square**2 / variance_delta_w_square
            # b0_itr = means_delta_w_square * (a0_itr - 1)
            # c0_itr = 2 + means_delta_v_square**2 / variance_delta_v_square
            # d0_itr = means_delta_v_square * (c0_itr - 1)

            # reset gazebo model state
            # reset_robot_simulation()

            # print (np.array(samples_Beta)).shape
            # ax[1, 0].plot((np.array(samples_Beta))[1000:, 0], linestyle=':')
            # ax[2, 0].plot((np.array(samples_Beta))[1000:, 1], linestyle=':')
            # ax[3, 0].plot((np.array(samples_Beta))[1000:, 2], linestyle=':')
            # ax[2, 1].plot((np.array(samples_Beta))[1000:, 3], linestyle=':')
            # ax[3, 1].plot((np.array(samples_Beta))[1000:, 4], linestyle=':')
            x = np.linspace(norm.ppf(0.1), norm.ppf(0.99), 1000)
            if iteration < 4:
                print "beta_posterior:", Beta0_itr, Sigma0_itr
                ax[iteration, 0].plot(x, norm.pdf(x, Beta0_itr[0], np.sqrt(Sigma0_itr[0,0])), 'r-', lw=1, alpha=0.9,
                              label=r'$\beta_{-1}$')
                ax[iteration, 0].plot(x, norm.pdf(x, Beta0_itr[1], np.sqrt(Sigma0_itr[1,1])), 'g-', lw=1, alpha=0.9,
                              label=r'$\beta_0$')
                ax[iteration, 0].plot(x, norm.pdf(x, Beta0_itr[2], np.sqrt(Sigma0_itr[2,2])), 'b-', lw=1, alpha=0.9,
                              label=r'$\beta_1$')
                ax[iteration, 0].plot(x, norm.pdf(x, Beta0_itr[3], np.sqrt(Sigma0_itr[3,3])), 'y-', lw=1,
                                      alpha=0.9, label=r'$\beta_2$')
                ax[iteration, 0].plot(x, norm.pdf(x, Beta0_itr[4], np.sqrt(Sigma0_itr[4,4])), 'c-', lw=1, alpha=0.9,
                              label=r'$b$')
                ax[iteration, 0].legend(loc='upper right', frameon=False)

                ax[iteration, 0].axvline(x=common_parameters.beta_true2[0], color='r', linestyle='-.', lw=1.5)
                ax[iteration, 0].axvline(x=common_parameters.beta_true2[1], color='g', linestyle='-.', lw=1.5)
                ax[iteration, 0].axvline(x=common_parameters.beta_true2[2], color='b', linestyle='-.', lw=1.5)
                ax[iteration, 0].axvline(x=common_parameters.beta_true2[3], color='y', linestyle='-.', lw=1.5)
                ax[iteration, 0].axvline(x=common_parameters.beta_true2[4], color='c', linestyle='-.', lw=1.5)

            if iteration > 15:
                ax[iteration-16, 1].plot(x, norm.pdf(x, means_Beta[0], np.sqrt(variance_Beta[0,0])), 'r-', lw=1, alpha=0.9,
                              label=r'$\beta_{-1}$')
                ax[iteration-16, 1].plot(x, norm.pdf(x, means_Beta[1], np.sqrt(variance_Beta[1,1])), 'g-', lw=1, alpha=0.9,
                              label=r'$\beta_0$')
                ax[iteration-16, 1].plot(x, norm.pdf(x, means_Beta[2], np.sqrt(variance_Beta[2,2])), 'b-', lw=1, alpha=0.9,
                              label=r'$\beta_1$')
                ax[iteration-16, 1].plot(x, norm.pdf(x, means_Beta[3], np.sqrt(variance_Beta[3,3])), 'y-', lw=1, alpha=0.9,
                              label=r'$\beta_2$')
                ax[iteration - 16, 1].plot(x, norm.pdf(x, means_Beta[4], np.sqrt(variance_Beta[4, 4])), 'c-', lw=1,
                                           alpha=0.9, label=r'$b$')
                ax[iteration-16, 1].legend(loc='upper right', frameon=False)

                ax[iteration-16, 1].axvline(x=common_parameters.beta_true2[0], color='r', linestyle='-.', lw=1.5)
                ax[iteration-16, 1].axvline(x=common_parameters.beta_true2[1], color='g', linestyle='-.', lw=1.5)
                ax[iteration-16, 1].axvline(x=common_parameters.beta_true2[2], color='b', linestyle='-.', lw=1.5)
                ax[iteration-16, 1].axvline(x=common_parameters.beta_true2[3], color='y', linestyle='-.', lw=1.5)
                ax[iteration - 16, 1].axvline(x=common_parameters.beta_true2[4], color='c', linestyle='-.', lw=1.5)

            plt.draw()
            plt.pause(1e-17)

        plt.show()
    except rospy.ROSInterruptException:
        pass
