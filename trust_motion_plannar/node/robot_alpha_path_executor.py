#!/usr/bin/env python
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import actionlib
from gazebo_msgs.srv import GetLinkState
from geometry_msgs.msg import PoseWithCovarianceStamped
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellGoal, NeighborCellResult, NeighborCellFeedback

import parameters_env_img_robot as common_parameters
import numpy as np
import cell_path_executor as cpe


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


def execExplore(goal_cells):
    next_cell_x, next_cell_y = goal_cells.to_cell_x, goal_cells.to_cell_y
    next_cell_target_pos = cpe.gen_space_target2(next_cell_x, next_cell_y)
    target_ori = quaternion_from_euler(0.0, 0.0, np.arctan2(-(next_cell_y - goal_cells.in_cell_y),
                                                            next_cell_x - goal_cells.in_cell_x))
    # print "alpha's goal cell:", next_cell_x, next_cell_y, "target ori:", target_ori
    update_odom()
    move_base_result = cpe.movebase_client(next_cell_target_pos[0][0],
                                           next_cell_target_pos[1][0],
                                           target_ori, robot_name='')
    # - common_parameters.husky_alpha_init.x - common_parameters.husky_alpha_init.y

    result = NeighborCellResult()
    result.at_cell_x = next_cell_x
    result.at_cell_y = next_cell_y
    print "!!! REMINDER: autonomous robots reached their temporary goal. " \
          "Please provide TRUST change with the HCI !!! ->->->\n"
    cellExploreServer.set_succeeded(result, "temporary target reached")


if __name__ == '__main__':
    try:
        rospy.init_node('robot_alpha_path_executor', anonymous=True)
        update_odom()
        cellExploreServer = actionlib.SimpleActionServer('/server1_localcells', NeighborCellAction, execExplore, False)
        cellExploreServer.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
