#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from turtlesim.msg import Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import numpy as np
import copy


class Follower:
    def __init__(self, node_name, follower_name, leader_name='husky_alpha'):
        # Creates a node with name 'turtlebot_controller' and make sure it is a
        # unique node (using anonymous=True).
        rospy.init_node(node_name, anonymous=True)

        # Publisher which will publish to the topic '/turtle1/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(follower_name + '/cmd_vel',
                                                  Twist, queue_size=10)

        # A subscriber to the topic '/turtle1/pose'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates,
                                                self.update_pose)

        self.follower_name = follower_name
        self.leader_name = leader_name
        self.pose = Pose()
        self.velocity = Twist()
        self.leader_pose = Pose()
        self.leader_velocity = Twist()

        self.rate = rospy.Rate(10)

    def update_pose(self, data):
        """Callback function which is called when a new message of type Pose is
        received by the subscriber."""
        name_arr = data.name
        for name_id in range(0, len(name_arr)):
            if name_arr[name_id] == self.follower_name:  # replace the name with robot's name
                position_orientation = copy.deepcopy(data.pose[name_id])
                pos = position_orientation.position
                ori = position_orientation.orientation
                self.pose.x = round(pos.x, 4)
                self.pose.y = round(pos.y, 4)
                euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
                self.pose.theta = euler_list[2]
                self.velocity = copy.deepcopy(data.twist[name_id])
            elif name_arr[name_id] == self.leader_name:  # replace the name with leader robot's name
                position_orientation = copy.deepcopy(data.pose[name_id])
                pos = position_orientation.position
                ori = position_orientation.orientation
                self.leader_pose.x = round(pos.x, 4)
                self.leader_pose.y = round(pos.y, 4)
                euler_list = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
                self.leader_pose.theta = euler_list[2]
                self.leader_velocity = copy.deepcopy(data.twist[name_id])

    def euclidean_distance_leader_follower(self):
        """Euclidean distance between current pose and the goal."""
        return np.sqrt(pow((self.leader_pose.x - self.pose.x), 2) +
                       pow((self.leader_pose.y - self.pose.y), 2))

    def linear_vel(self, safe_distance=2.0, constant=0.40):
        delta_vel = constant * (self.euclidean_distance_leader_follower() - safe_distance)
        vel_x = self.leader_velocity.linear.x + delta_vel * np.cos(self.steering_angle())
        vel_y = self.leader_velocity.linear.y + delta_vel * np.sin(self.steering_angle())
        return vel_x * np.cos(self.pose.theta) + vel_y * np.sin(self.pose.theta)

    def steering_angle(self):
        return np.arctan2(self.leader_pose.y - self.pose.y, self.leader_pose.x - self.pose.x)

    def angular_vel(self, constant=0.80):
        delta_angle = (self.steering_angle() - self.pose.theta)
        if delta_angle > np.pi:
            delta_angle -= 2*np.pi
        elif delta_angle < -np.pi:
            delta_angle += 2*np.pi
        return constant * delta_angle + self.velocity.angular.z

    def move2goal(self):
        """Moves the turtle to the goal."""
        vel_msg = Twist()

        while not rospy.is_shutdown():
            # Linear velocity in the x-axis.
            vel_msg.linear.x = self.linear_vel()
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0

            # Angular velocity in the z-axis.
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = self.angular_vel()

            # Publishing our vel_msg
            if (self.euclidean_distance_leader_follower() - 6.0) > 0.0 and abs(self.velocity.linear.x) < 0.1:
                vel_msg.linear.x = -0.2 # vel_msg.linear.x
                vel_msg.angular.z = -vel_msg.angular.z + 0.8
                self.velocity_publisher.publish(vel_msg)
            else:
                self.velocity_publisher.publish(vel_msg)

            # Publish at the desired rate.
            self.rate.sleep()

        # Stopping our robot after the movement is over.
        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

        # If we press control + C, the node will stop.
        rospy.spin()


if __name__ == '__main__':
    try:
        follower_gamma = Follower('robot_gamma_path_executor', 'summit_xl_gamma',
                                 leader_name='summit_xl_beta')
        follower_gamma.move2goal()
    except rospy.ROSInterruptException:
        pass
