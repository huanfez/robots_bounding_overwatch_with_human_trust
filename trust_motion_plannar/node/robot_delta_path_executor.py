#!/usr/bin/env python
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import actionlib
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellGoal, NeighborCellResult, NeighborCellFeedback
from gazebo_msgs.msg import ModelStates
from turtlesim.msg import Pose
import cell_path_executor as cpe

import numpy as np
import copy


def update_pose(data):
    """Callback function which is called when a new message of type Pose is
    received by the subscriber."""
    name_arr = data.name
    for name_id in range(0, len(name_arr)):
        if name_arr[name_id] == 'summit_xl_delta':  # replace the name with robot's name
            position_orientation = copy.deepcopy(data.pose[name_id])
            pos = position_orientation.position
            ori = position_orientation.orientation
            robot_delta_pose.x = round(pos.x, 4)
            robot_delta_pose.y = round(pos.y, 4)
            euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            robot_delta_pose.theta = euler_list[2]
            velocity = copy.deepcopy(data.twist[name_id])
        elif name_arr[name_id] == '/':  # replace the name with robot's name
            leader_position_orientation = copy.deepcopy(data.pose[name_id])
            leader_pos = leader_position_orientation.position
            leader_ori = leader_position_orientation.orientation
            leader_robot_pose.x = round(leader_pos.x, 4)
            leader_robot_pose.y = round(leader_pos.y, 4)
            leader_euler_list = euler_from_quaternion([leader_ori.x, leader_ori.y, leader_ori.z, leader_ori.w])
            leader_robot_pose.theta = leader_euler_list[2]
            leader_velocity = copy.deepcopy(data.twist[name_id])


def execExplore(goal_cells):
    next_cell_x, next_cell_y = goal_cells.to_cell_x, goal_cells.to_cell_y
    next_cell_target_pos = cpe.gen_space_target2(next_cell_x, next_cell_y)
    target_ori = quaternion_from_euler(0.0, 0.0, np.arctan2(-(next_cell_y - goal_cells.in_cell_y),
                                                            next_cell_x - goal_cells.in_cell_x))
    # print "delta's goal cell:", next_cell_x, next_cell_y, "target ori:", target_ori

    while not rospy.is_shutdown():
        delta_distance_x = next_cell_target_pos[0][0] - robot_delta_pose.x
        delta_distance_y = next_cell_target_pos[1][0] - robot_delta_pose.y
        distance = np.sqrt(delta_distance_x**2 + delta_distance_y**2)

        leader_distance_x = leader_robot_pose.x - robot_delta_pose.x
        leader_distance_y = leader_robot_pose.y - robot_delta_pose.y
        leader_distance = np.sqrt(leader_distance_x**2 + leader_distance_y**2)

        if distance < 8.0 or leader_distance < 6.0:
            break
        rate.sleep()

    print "->->-> REMINDER: human operated robot reached the temporary goal !!!\n"
    print "-------------------------------------------\n"
    result = NeighborCellResult()
    result.at_cell_x = next_cell_x
    result.at_cell_y = next_cell_y
    # print "result is:", result.at_cell_x, result.at_cell_y
    cellExploreServer2.set_succeeded(result, "target reached")


if __name__ == '__main__':
    try:
        rospy.init_node('robot_delta_path_executor', anonymous=True)
        rate = rospy.Rate(2)

        robot_delta_pose = Pose()
        leader_robot_pose = Pose()
        pose_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, update_pose)

        cellExploreServer2 = actionlib.SimpleActionServer('/server2_localcells', NeighborCellAction, execExplore,
                                                              False)
        cellExploreServer2.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass