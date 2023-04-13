#! /usr/bin/env python2

import rospy
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ModelStates
from turtlesim.msg import Pose
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32
from trust_motion_plannar.srv import trust, trustResponse

import numpy as np
import copy
import tifffile as tf

import parameters_env_img_robot as common_parameters


def update_pose(data):
    """Callback function which is called when a new message of type Pose is
    received by the subscriber."""
    name_arr = data.name
    for name_id in range(0, len(name_arr)):
        if name_arr[name_id] == '/':  # replace the name with robot's name
            position_orientation = copy.deepcopy(data.pose[name_id])
            pos = position_orientation.position
            ori = position_orientation.orientation
            robot_alpha_pose.x = round(pos.x, 4)
            robot_alpha_pose.y = round(pos.y, 4)
            euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            robot_alpha_pose.theta = euler_list[2]
        elif name_arr[name_id] == 'summit_xl_beta':  # replace the name with robot's name
            position_orientation = copy.deepcopy(data.pose[name_id])
            pos = position_orientation.position
            ori = position_orientation.orientation
            robot_beta_pose.x = round(pos.x, 4)
            robot_beta_pose.y = round(pos.y, 4)
            euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            robot_beta_pose.theta = euler_list[2]
        elif name_arr[name_id] == 'summit_xl_gamma':  # replace the name with robot's name
            position_orientation = copy.deepcopy(data.pose[name_id])
            pos = position_orientation.position
            ori = position_orientation.orientation
            robot_gamma_pose.x = round(pos.x, 4)
            robot_gamma_pose.y = round(pos.y, 4)
            euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            robot_gamma_pose.theta = euler_list[2]


def update_trust_vector(data):
    robots_trust_vector = data

    cw = common_parameters.cell_width
    ch = common_parameters.cell_height
    # Update position
    robot_alpha_img_pose = common_parameters.envPos2imgPos([robot_alpha_pose.x, robot_alpha_pose.y])
    robot_beta_img_pose = common_parameters.envPos2imgPos([robot_beta_pose.x, robot_beta_pose.y])
    robot_gamma_img_pose = common_parameters.envPos2imgPos([robot_gamma_pose.x, robot_gamma_pose.y])

    # update trust dictionary
    cell_x_alpha, cell_y_alpha = int(robot_alpha_img_pose[0] / cw), int(robot_alpha_img_pose[1] / ch)
    # (common_parameters.trust_dict[(cell_x_alpha, cell_y_alpha)])[0] = robots_trust_vector.x

    cell_x_beta, cell_y_beta = int(robot_beta_img_pose[0] / cw), int(robot_beta_img_pose[1] / ch)
    # (common_parameters.trust_dict[(cell_x_beta, cell_y_beta)])[1] = robots_trust_vector.y

    cell_x_gamma, cell_y_gamma = int(robot_gamma_img_pose[0] / cw), int(robot_gamma_img_pose[1] / ch)
    # (common_parameters.trust_dict[(cell_x_gamma, cell_y_gamma)])[2] = robots_trust_vector.z

    trust_temp_data[cell_y_alpha, cell_x_alpha, 0] = robots_trust_vector.x / 100.0
    trust_temp_data[cell_y_beta, cell_x_beta, 1] = robots_trust_vector.y / 100.0
    trust_temp_data[cell_y_gamma, cell_x_gamma, 2] = robots_trust_vector.z / 100.0
    tf.imwrite(common_parameters.trust_temp_tif, trust_temp_data)


try:
    rospy.init_node('human_trust_info', anonymous=True)

    trust_temp_data = tf.imread(common_parameters.trust_temp_tif)
    # robots' pose in environment
    robot_alpha_pose = Pose()
    robot_beta_pose = Pose()
    robot_gamma_pose = Pose()
    pose_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, update_pose)

    # robots' trust value
    robots_trust_subscriber = rospy.Service('/get_trust_vector', trust, update_trust_vector)

    rospy.spin()
except rospy.ROSInterruptException:
    pass
