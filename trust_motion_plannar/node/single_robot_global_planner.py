#! /usr/bin/env python2

import rospy
import actionlib
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetLinkState
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from trust_motion_plannar.msg import NeighborCellAction, NeighborCellGoal, NeighborCellResult, NeighborCellFeedback

import cell_path_plannar as cpr


def reset_robot_states():
    alpha_state = ModelState()
    alpha_state.model_name = '/'
    alpha_state.pose.position.x = -35.0
    alpha_state.pose.position.y = -85.0
    alpha_state.pose.position.z = -14.0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(alpha_state)

    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


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

        """ action client 1: team member 1 send cells for low-level to bounding overwatch """
        cell_sender_client1 = actionlib.SimpleActionClient('/server1_localcells', NeighborCellAction)
        cell_sender_client1.wait_for_server()

        # env_discrete_path, cell_path = cpr.gen_discrete_path()
        cell_path = [[8, 13], [7, 13], [7, 12], [7, 11]]

        iteration = 0
        while iteration < 5:
            cell_index = 0
            steps = len(cell_path)
            while cell_index < steps-1:
                cells = NeighborCellGoal()
                cells.in_cell_x = cell_path[cell_index][0]
                cells.in_cell_y = cell_path[cell_index][1]
                cells.to_cell_x = cell_path[cell_index + 1][0]
                cells.to_cell_y = cell_path[cell_index + 1][1]

                cell_sender_client1.send_goal(cells)
                cell_sender_client1.wait_for_result()

                cell_index += 1

            reset_robot_simulation()
            iteration += 1
        # reset_robot_states()

        # cell_path = [[7, 13], [6, 13], [8, 13]]
        # cell_index = 0
        # steps = len(cell_path)
        # while cell_index < steps - 1:
        #     cells = NeighborCellGoal()
        #     cells.in_cell_x = cell_path[cell_index][0]
        #     cells.in_cell_y = cell_path[cell_index][1]
        #     cells.to_cell_x = cell_path[cell_index + 1][0]
        #     cells.to_cell_y = cell_path[cell_index + 1][1]
        #
        #     cell_sender_client1.send_goal(cells)
        #     cell_sender_client1.wait_for_result()
        #
        #     cell_index += 1

        rospy.spin()
    except rospy.ROSInterruptException:
        pass