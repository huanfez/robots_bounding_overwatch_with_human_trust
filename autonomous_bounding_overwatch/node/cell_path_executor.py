#!/usr/bin/env python
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import actionlib
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellGoal, NeighborCellResult, NeighborCellFeedback
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

import parameters_env_img_robot as common_parameters
import numpy as np


# Select the point with the best LoS in next cell as the target
def gen_space_target2(next_cell_x, next_cell_y, margin_width=10, margin_height=10, obstacle_pixel=150):
    node_dict = {}

    dsm_img = np.asarray(common_parameters.dsm_img)

    # select target from the pixels in the next cell
    start_window_x = next_cell_x * common_parameters.cell_width
    end_window_x = next_cell_x * common_parameters.cell_width + common_parameters.cell_width
    start_window_y = next_cell_y * common_parameters.cell_height
    end_window_y = next_cell_y * common_parameters.cell_height + common_parameters.cell_height
    for px in range(start_window_x + margin_width, end_window_x - margin_width):
        for py in range(start_window_y + margin_height, end_window_y - margin_height):
            if dsm_img[py][px] > obstacle_pixel:
                continue
            node_dict[(px, py)] = dsm_img[py][px]

    key_min = min(node_dict, key=node_dict.get)

    target_env_pos = common_parameters.imgPos2envPos(np.array([[key_min[0]], [key_min[1]]]))
    print "target pixel:", target_env_pos
    return target_env_pos


def movebase_client(targetx, targety, target_ori, robot_name='husky_alpha'):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

    # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'odom'
    goal.target_pose.header.stamp = rospy.Time.now()
    # Move 0.5 meters forward along the x axis of the "map" coordinate frame
    goal.target_pose.pose.position.x = targetx
    goal.target_pose.pose.position.y = targety
    # No rotation of the mobile base frame w.r.t. map frame
    goal.target_pose.pose.orientation.x = target_ori[0]
    goal.target_pose.pose.orientation.y = target_ori[1]
    goal.target_pose.pose.orientation.z = target_ori[2]
    goal.target_pose.pose.orientation.w = target_ori[3]

    # Sends the goal to the action server.
    client.send_goal(goal)
    # Waits for the server to finish performing the action.
    wait = client.wait_for_result()
    # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        # Result of executing the action
        return client.get_result()
