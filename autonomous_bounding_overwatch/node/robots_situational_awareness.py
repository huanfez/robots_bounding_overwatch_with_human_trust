#! /usr/bin/env python2

import rospy
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ModelStates
from turtlesim.msg import Pose
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Imu

import numpy as np
import math
import copy
import parameters_env_img_robot as common_parameters


def offline_traversability(pose):
    offline_traversability_map = np.asarray(common_parameters.traversability_img)
    img_pose = common_parameters.envPos2imgPos(pose)
    offline_traversability_ = offline_traversability_map[img_pose[1]][img_pose[0]]

    normalized_traversability = (1 - np.exp(offline_traversability_ * 100.0 - 2.2)) / (
            1 + np.exp(offline_traversability_ * 100.0 - 2.2))
    # normalized_traversability = (0.1 - np.asarray(common_parameters.traversability_img))*10.0
    return normalized_traversability


def offline_visibility(pose):
    offline_visibility_map = np.asarray(common_parameters.visibility_img)
    img_pose = common_parameters.envPos2imgPos(pose)
    offline_visibility_ = offline_visibility_map[img_pose[1]][img_pose[0]]

    normalized_visibility = (1 - np.exp(offline_visibility_ - 1.5)) / (1 + np.exp(offline_visibility_ - 1.5))
    # normalized_visibility = (1.50 - np.asarray(common_parameters.visibility_img)) / 10.0
    return normalized_visibility


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


def update_imu_alpha(data):
    global robot_alpha_online_traversability
    robot_alpha_online_traversability = 1 - np.abs(data.linear_acceleration.z - 9.8)
    robot_alpha_online_traversability = (1 - np.exp(-robot_alpha_online_traversability)) / (
            1 + np.exp(-robot_alpha_online_traversability)) * 2


def update_laserscan_alpha(data):
    global robot_alpha_online_visibility
    laser_range = np.array(data.ranges)
    robot_alpha_online_visibility = np.mean(laser_range[np.isfinite(laser_range)]) - 10.0
    robot_alpha_online_visibility = (1 - np.exp(-robot_alpha_online_visibility)) / (
            1 + np.exp(-robot_alpha_online_visibility))


def update_imu_beta(data):
    global robot_beta_online_traversability
    robot_beta_online_traversability = 1.0 - np.abs(data.linear_acceleration.z - 9.8)
    robot_beta_online_traversability = (1 - np.exp(-robot_beta_online_traversability)) / (
            1 + np.exp(-robot_beta_online_traversability)) * 2


def update_laserscan_beta(data):
    global robot_beta_online_visibility
    laser_range = np.array(data.ranges)
    robot_beta_online_visibility = np.mean(laser_range[np.isfinite(laser_range)]) - 6.0
    robot_beta_online_visibility = (1 - np.exp(-robot_beta_online_visibility)) / (
            1 + np.exp(-robot_beta_online_visibility))


def update_imu_gamma(data):
    global robot_gamma_online_traversability
    robot_gamma_online_traversability = 1.0 - np.abs(data.linear_acceleration.z - 9.8)
    robot_gamma_online_traversability = (1 - np.exp(-robot_gamma_online_traversability)) / (
            1 + np.exp(-robot_gamma_online_traversability)) * 2


def update_laserscan_gamma(data):
    global robot_gamma_online_visibility
    laser_range = np.array(data.ranges)
    robot_gamma_online_visibility = np.mean(laser_range[np.isfinite(laser_range)]) - 6.0
    robot_gamma_online_visibility = (1 - np.exp(-robot_gamma_online_visibility)) / (
            1 + np.exp(-robot_gamma_online_visibility))


def update_trust_vector(data):
    global robots_trust_vector
    robots_trust_vector = data


try:
    rospy.init_node('situational_awareness_info', anonymous=True)
    rate = rospy.Rate(2.0)

    # robots' trust value
    robots_trust_subscriber = rospy.Subscriber("trust_vector", Vector3, update_trust_vector)
    robots_trust_vector = Vector3()

    # robots' online traversability and visibility
    robot_alpha_online_traversability = 0.0
    robot_alpha_online_visibility = 0.0
    robot_beta_online_traversability = 0.0
    robot_beta_online_visibility = 0.0
    robot_gamma_online_traversability = 0.0
    robot_gamma_online_visibility = 0.0

    # Publisher of robots' traversability & visibility
    robot_alpha_traversability_publisher = rospy.Publisher("/husky_alpha/traversability", Float32, queue_size=10)
    robot_alpha_visibility_publisher = rospy.Publisher("/husky_alpha/visibility", Float32, queue_size=10)

    robot_beta_traversability_publisher = rospy.Publisher("/summit_xl_beta/traversability", Float32, queue_size=10)
    robot_beta_visibility_publisher = rospy.Publisher("/summit_xl_beta/visibility", Float32, queue_size=10)

    robot_gamma_traversability_publisher = rospy.Publisher("/summit_xl_gamma/traversability", Float32, queue_size=10)
    robot_gamma_visibility_publisher = rospy.Publisher("/summit_xl_gamma/visibility", Float32, queue_size=10)

    # robots' pose in environment
    robot_alpha_pose = Pose()
    robot_beta_pose = Pose()
    robot_gamma_pose = Pose()
    pose_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, update_pose)

    # robots' real-time sensing information
    robot_alpha_imu_subscriber = rospy.Subscriber('/imu/data', Imu, update_imu_alpha)
    robot_alpha_lasercan_subscriber = rospy.Subscriber('/scan', LaserScan, update_laserscan_alpha)

    robot_beta_imu_subscriber = rospy.Subscriber('/summit_xl_beta/imu/data_raw', Imu, update_imu_beta)
    robot_beta_lasercan_subscriber = rospy.Subscriber('/summit_xl_beta/front_laser/scan',
                                                      LaserScan, update_laserscan_beta)

    robot_gamma_imu_subscriber = rospy.Subscriber('/summit_xl_gamma/imu/data_raw', Imu, update_imu_gamma)
    robot_gamma_lasercan_subscriber = rospy.Subscriber('/summit_xl_gamma/front_laser/scan',
                                                       LaserScan, update_laserscan_gamma)

    # discrete_path_file = '/home/hz/catkin_ws/src/trust_motion_plannar/node/summit_xl_c.csv'
    # csvfile = open(discrete_path_file, 'w')
    # fieldnames = ['time', 'traversability', 'visibility']
    # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # writer.writeheader()

    # rate.sleep()
    # keep updating
    while not rospy.is_shutdown():
        # robot alpha situational awareness information
        robot_alpha_offline_traversability = offline_traversability([robot_alpha_pose.x, robot_alpha_pose.y])
        raot = robot_alpha_offline_traversability + robot_alpha_online_traversability
        robot_alpha_offline_visibility = offline_visibility([robot_alpha_pose.x, robot_alpha_pose.y])
        raov = robot_alpha_offline_visibility + robot_alpha_online_visibility
        # Update map
        robot_alpha_img_pose = common_parameters.envPos2imgPos([robot_alpha_pose.x, robot_alpha_pose.y])
        common_parameters.r1_dynamic_traversability[robot_alpha_img_pose[1]][robot_alpha_img_pose[0]] = raot
        common_parameters.r1_dynamic_visibility[robot_alpha_img_pose[1]][robot_alpha_img_pose[0]] = raov
        # Publish
        robot_alpha_traversability_publisher.publish(raot)
        robot_alpha_visibility_publisher.publish(raov)

        # robot beta: situational awareness information
        robot_beta_offline_traversability = offline_traversability([robot_beta_pose.x, robot_beta_pose.y])
        rbot = robot_beta_offline_traversability + robot_beta_online_traversability
        robot_beta_offline_visibility = offline_visibility([robot_beta_pose.x, robot_beta_pose.y])
        rbov = robot_beta_offline_visibility + robot_beta_online_visibility
        # Update map
        robot_beta_img_pose = common_parameters.envPos2imgPos([robot_beta_pose.x, robot_beta_pose.y])
        common_parameters.r2_dynamic_traversability[robot_beta_img_pose[1]][robot_beta_img_pose[0]] = rbot
        common_parameters.r2_dynamic_visibility[robot_beta_img_pose[1]][robot_beta_img_pose[0]] = rbov
        # Publish
        robot_beta_traversability_publisher.publish(rbot)
        robot_beta_visibility_publisher.publish(rbov)

        # robot gamma: situational awareness information
        robot_gamma_offline_traversability = offline_traversability([robot_gamma_pose.x, robot_gamma_pose.y])
        rgot = robot_gamma_offline_traversability + robot_gamma_online_traversability
        robot_gamma_offline_visibility = offline_visibility([robot_gamma_pose.x, robot_gamma_pose.y])
        rgov = robot_gamma_offline_visibility + robot_gamma_online_visibility
        # Update map
        robot_gamma_img_pose = common_parameters.envPos2imgPos([robot_gamma_pose.x, robot_gamma_pose.y])
        common_parameters.r3_dynamic_traversability[robot_gamma_img_pose[1]][robot_gamma_img_pose[0]] = rgot
        common_parameters.r3_dynamic_visibility[robot_gamma_img_pose[1]][robot_gamma_img_pose[0]] = rgov
        # Publish
        robot_gamma_traversability_publisher.publish(rgot)
        robot_gamma_visibility_publisher.publish(rgov)

        # update trust dictionary
        cell_x, cell_y = robot_alpha_img_pose[0] / common_parameters.cell_width, \
                         robot_alpha_img_pose[1] / common_parameters.cell_height
        common_parameters.trust_dict[(cell_x, cell_y)] = np.array([robots_trust_vector.x, robots_trust_vector.y,
                                                                   robots_trust_vector.z])
        # writer.writerow({'time':rospy.get_rostime(), 'traversability':offline_traversability_t,
        #                  'visibility':offline_visibility_t})
        rate.sleep()
    # csvfile.close()
    rospy.spin()
except rospy.ROSInterruptException:
    pass
