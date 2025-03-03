cmake_minimum_required (VERSION 2.8)
project(msg_action)

# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

include_directories(/opt/ros/noetic/include)
find_package(OpenCV 3.4 REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(catkin REQUIRED COMPONENTS
  std_msgs
  rospy
  genmsg
  actionlib_msgs
  actionlib
  sensor_msgs
  cv_bridge
  geometry_msgs
)

add_message_files(
  DIRECTORY msgs
  FILES Sequence.msg
  CloudMap.msg
  Descriptor.msg
  KeyFrame.msg
  KeyPoint.msg
  MapPoint.msg
  Observation.msg
)

add_action_files(
  DIRECTORY actions
  FILES CloudSlam.action
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  actionlib_msgs
)

add_executable(
  AC
  src/client.cpp
  )

target_link_libraries(AC
  ${OpenCV_LIBS}
  # ${Boost_LIBRARIES}
  # -lboost_system
  -lboost_thread
  ${catkin_LIBRARIES}
  )
