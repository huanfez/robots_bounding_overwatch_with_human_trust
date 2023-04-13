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
import tifffile as tf
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm

import cell_path_selector as cps
import cell_path_executor as cpe
import parameter_mcmc as pmc
import parameters_env_img_robot as common_parameters


#
def plot_posterior_pdf(means_Beta, variance_Beta, iteration, ax, x):
    if iteration < 10 and iteration % 5 == 0:
        ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, means_Beta[0], np.sqrt(variance_Beta[0, 0])), 'r-', lw=1,
                                      alpha=0.9, label=r'$\beta_0$')
        ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, means_Beta[1], np.sqrt(variance_Beta[1, 1])), 'g-', lw=1,
                                      alpha=0.9, label=r'$\beta_1$')
        ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, means_Beta[2], np.sqrt(variance_Beta[2, 2])), 'b-', lw=1,
                                      alpha=0.9, label=r'$\beta_2$')
        ax[iteration / 5 + 1, 0].plot(x, norm.pdf(x, means_Beta[3], np.sqrt(variance_Beta[3, 3])), 'y-', lw=1,
                                      alpha=0.9, label=r'$b$')
        ax[iteration / 5 + 1, 0].legend(loc='upper right', frameon=False)
        ax[iteration / 5 + 1, 0].set_ylabel('PDF')
        ax[iteration / 5 + 1, 0].set_title('Posterior')

    if iteration >= 10 and iteration % 5 == 0:
        ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, means_Beta[0], np.sqrt(variance_Beta[0, 0])), 'r-',
                                         lw=1, alpha=0.9, label=r'$\beta_0$')
        ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, means_Beta[1], np.sqrt(variance_Beta[1, 1])), 'g-',
                                         lw=1, alpha=0.9, label=r'$\beta_1$')
        ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, means_Beta[2], np.sqrt(variance_Beta[2, 2])), 'b-',
                                         lw=1, alpha=0.9, label=r'$\beta_2$')
        ax[(iteration - 10) / 5, 1].plot(x, norm.pdf(x, means_Beta[3], np.sqrt(variance_Beta[3, 3])), 'y-',
                                         lw=1, alpha=0.9, label=r'$b$')
        ax[(iteration - 10) / 5, 1].legend(loc='upper right', frameon=False)
        ax[(iteration - 10) / 5, 1].set_ylabel('PDF')
        ax[(iteration - 10) / 5, 1].set_title('Posterior')


# read image
def avg_situational_awareness(dynamic_traversability, dynamic_visibility):
    avg_traversability = ndimage.uniform_filter(dynamic_traversability, (common_parameters.cell_height,
                                                                         common_parameters.cell_width))
    avg_visibility = ndimage.uniform_filter(dynamic_visibility, (common_parameters.cell_height,
                                                                 common_parameters.cell_width))
    return avg_traversability, avg_visibility


# discrete cell info
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


def initialize_odom():
    target_ori = [0.0, 0.0, 0.0, 1.0]
    move_base_result = cpe.movebase_client(1.0, 0.0, target_ori, robot_name='')
    reset_robot_simulation()


# Main program: train the model parameter and obtain optimal path
if __name__ == '__main__':
    try:
        rospy.init_node('wayofpoints', anonymous=True)

        # action client 1: squad 1 send cells for low-level to bounding overwatch
        cell_sender_client1 = actionlib.SimpleActionClient('/server1_localcells', NeighborCellAction)
        cell_sender_client1.wait_for_server()

        # action client 2: squad 2 send cells for low-level to bounding overwatch
        cell_sender_client2 = actionlib.SimpleActionClient('/server2_localcells', NeighborCellAction)
        cell_sender_client2.wait_for_server()

        # Initialize husky's odom
        initialize_odom()

        # Initial prior for MCMC
        alpha1 = np.diag([1.0, 1.0, 1.0])
        alpha2 = -np.diag([1.0, 1.0, 1.0])
        Alpha = np.concatenate((alpha1, alpha2), axis=1)
        Beta0_itr = np.asarray([0.25, 0.25, 0.25, 0.25])
        Sigma0_itr = np.diag([1e5, 1e5, 1e5, 1e5])
        a0_itr = 3.0
        b0_itr = 1.0
        c0_itr = 3.0
        d0_itr = 1.0

        # Initialization for trust_tiff, environment and preference
        zeros = np.zeros((common_parameters.grid_height, common_parameters.grid_width, 3), dtype=float)
        tf.imwrite(common_parameters.trust_temp_tif, zeros)
        environment_dict, obs_list = gen_env(common_parameters.r1_dynamic_traversability,
                                             common_parameters.r1_dynamic_visibility,
                                             common_parameters.r2_dynamic_traversability,
                                             common_parameters.r2_dynamic_visibility,
                                             common_parameters.r3_dynamic_traversability,
                                             common_parameters.r3_dynamic_visibility)
        f = np.zeros(len(common_parameters.candidate_cell_path_list))

        # plotting for prior
        fig, ax = plt.subplots(3, 2)
        fig.tight_layout()
        x = np.linspace(norm.ppf(0.1), norm.ppf(0.99), 1000)
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[0], np.sqrt(Sigma0_itr[0, 0])), 'r-', lw=1, alpha=0.9, label=r'$\beta_0$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[1], np.sqrt(Sigma0_itr[1, 1])), 'g-', lw=1, alpha=0.9, label=r'$\beta_1$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[2], np.sqrt(Sigma0_itr[2, 2])), 'b-', lw=1, alpha=0.9, label=r'$\beta_2$')
        ax[0, 0].plot(x, norm.pdf(x, Beta0_itr[3], np.sqrt(Sigma0_itr[3, 3])), 'y-', lw=1, alpha=0.9, label=r'$b$')
        ax[0, 0].legend(loc='upper right', frameon=False)
        ax[0, 0].ticklabel_format(style='sci')
        ax[0, 0].set_ylabel('PDF')
        ax[0, 0].set_title('Prior')

        # Iterative trials
        for iteration in range(0, 11):
            # probability improvement strategy:
            optimo_cell_path, f = cps.optimal_cell_path_pi(Beta0_itr, Sigma0_itr,
                                                           common_parameters.candidate_cell_path_list,
                                                           environment_dict, f, gamma=0.0, sample_size=4000)

            print optimo_cell_path
            # exit()

            # Send path to motion server and collect data
            cell_index = 0
            steps = len(optimo_cell_path)
            while cell_index < steps - 1:
                cells = NeighborCellGoal()
                cells.in_cell_x = optimo_cell_path[cell_index][0]
                cells.in_cell_y = optimo_cell_path[cell_index][1]
                cells.to_cell_x = optimo_cell_path[cell_index + 1][0]
                cells.to_cell_y = optimo_cell_path[cell_index + 1][1]

                cell_sender_client1.send_goal(cells)
                cell_sender_client1.wait_for_result()

                cell_sender_client2.send_goal(cells)
                cell_sender_client2.wait_for_result()

                cell_index += 1

            # Bayesian inference: update model parameters
            # Update and arrange the output variable
            trust_temp_data = tf.imread(common_parameters.trust_temp_tif)
            y_1toK = cps.cell_path_trust_tif(optimo_cell_path, trust_temp_data)

            # Update and arrange the input variables
            common_parameters.r1_dynamic_traversability = tf.imread(common_parameters.traversability_r1)
            common_parameters.r1_dynamic_visibility = tf.imread(common_parameters.visibility_r1)
            common_parameters.r2_dynamic_traversability = tf.imread(common_parameters.traversability_r2)
            common_parameters.r2_dynamic_visibility = tf.imread(common_parameters.visibility_r2)
            common_parameters.r3_dynamic_traversability = tf.imread(common_parameters.traversability_r3)
            common_parameters.r3_dynamic_visibility = tf.imread(common_parameters.visibility_r3)
            environment_dict, obs_list = gen_env(common_parameters.r1_dynamic_traversability,
                                                 common_parameters.r1_dynamic_visibility,
                                                 common_parameters.r2_dynamic_traversability,
                                                 common_parameters.r2_dynamic_visibility,
                                                 common_parameters.r3_dynamic_traversability,
                                                 common_parameters.r3_dynamic_visibility)
            Z_1toK = cps.cell_path_situational_awareness(optimo_cell_path, environment_dict)

            # mcmc sampling
            samples_x_1toK, samples_Beta, samples_delta_w_square, samples_delta_v_square = pmc.iterated_sampling(
                y_1toK, Z_1toK, Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, Alpha, iters=5000)

            # mean value of posterior model parameters
            means_Beta, variance_Beta, means_delta_w_square, variance_delta_w_square, means_delta_v_square, \
                variance_delta_v_square = pmc.mean_value_model_parameters(samples_x_1toK, samples_Beta,
                                                                          samples_delta_w_square, samples_delta_v_square)

            # Set posterior to be prior
            Beta0_itr = means_Beta
            Sigma0_itr = variance_Beta
            a0_itr = 2 + means_delta_w_square**2 / variance_delta_w_square
            b0_itr = means_delta_w_square * (a0_itr - 1)
            c0_itr = 2 + means_delta_v_square**2 / variance_delta_v_square
            d0_itr = means_delta_v_square * (c0_itr - 1)

            # Save model parameter value and trust data as tiff
            trust_tif = common_parameters.trust_pkg_dir + '/node/data/trust_iter{0}.tif'.format(iteration)
            tf.imwrite(trust_tif, trust_temp_data)

            # reset gazebo model state
            reset_robot_simulation()

            # plotting posterior
            plot_posterior_pdf(Beta0_itr, Sigma0_itr, iteration, ax, x)
            plt.draw()
            plt.pause(1e-17)

        plt.show()

    except rospy.ROSInterruptException:
        pass
