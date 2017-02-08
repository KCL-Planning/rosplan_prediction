cmake_minimum_required(VERSION 2.8.3)
project(rosplan_interface_prediction)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  rosplan_knowledge_msgs
  squirrel_prediction_msgs
  diagnostic_msgs
)

###################################
## catkin specific configuration ##
###################################

## Declare things to be passed to dependent projects
catkin_package(
  INCLUDE_DIRS include ${catkin_INCLUDE_DIRS}
  CATKIN_DEPENDS roscpp rospy std_msgs rosplan_knowledge_msgs squirrel_prediction_msgs diagnostic_msgs
  DEPENDS
)

###########
## Build ##
###########

## include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)
  
## Declare cpp executables
add_executable(rpprediction src/RPPrediction.cpp)
target_link_libraries(rpprediction ${catkin_LIBRARIES})
add_dependencies(rpprediction ${catkin_EXPORTED_TARGETS})
