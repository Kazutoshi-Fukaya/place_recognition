cmake_minimum_required(VERSION 3.0.2)
project(place_recognition_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  message_generation
)

add_message_files(
  FILES
  PoseStamped.msg
)

generate_messages(
    DEPENDENCIES
    geometry_msgs std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs message_runtime
)
