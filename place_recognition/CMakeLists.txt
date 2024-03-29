cmake_minimum_required(VERSION 3.0.2)
project(place_recognition)

add_compile_options(-o2 -g -Wall)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dom_estimator_msgs
  object_detector_msgs
  place_recognition_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

# OpenCV
find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV 3.4.3 QUIET)
    if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
  endif()
endif()

catkin_package(
  INCLUDE_DIRS include
  ${OpenCV_INCLUDE_DIRS}
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# DBoW
add_library(DBoW3 SHARED
  dbow3/descriptors_manipulator.cpp
  dbow3/feature_vector.cpp
  dbow3/scoring_object.cpp
  dbow3/quicklz.cpp
  dbow3/vocabulary.cpp
  dbow3/query_results.cpp
  dbow3/bow_vector.cpp
  dbow3/database.cpp
)

# place_recognition
add_executable(place_recognition src/place_recognition.cpp)
target_link_libraries(place_recognition
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# dom_place_recognition
add_executable(dom_place_recognition
  src/dom_place_recognition_interface.cpp
  src/dom_place_recognition_interfaces.cpp
  src/dom_place_recognition.cpp
)
target_link_libraries(dom_place_recognition
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# batch_place_recognition
add_executable(batch_place_recognition src/batch_place_recognition.cpp)
target_link_libraries(batch_place_recognition
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# multi_place_recognition
add_executable(multi_place_recognition src/multi_place_recognition.cpp)
target_link_libraries(multi_place_recognition
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# ========================= UTILS =========================

# dbow3_demo
add_executable(dbow3_demo ros_utils/dbow3_demo.cpp)
target_link_libraries(dbow3_demo
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# image_searcher
add_executable(image_searcher ros_utils/image_searcher.cpp)
target_link_libraries(image_searcher
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  DBoW3
)

# keypoints_detector
add_executable(keypoints_detector ros_utils/keypoints_detector.cpp)
target_link_libraries(keypoints_detector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# keypoints_recorder
add_executable(keypoints_recorder ros_utils/keypoints_recorder.cpp)
target_link_libraries(keypoints_recorder
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# mismatch_calculator
add_executable(mismatch_calculator ros_utils/mismatch_calculator.cpp)
target_link_libraries(mismatch_calculator ${catkin_LIBRARIES})

# mismatch_visualizer
add_executable(mismatch_visualizer ros_utils/mismatch_visualizer.cpp)
target_link_libraries(mismatch_visualizer ${catkin_LIBRARIES})

# reference_data_creator
add_executable(reference_data_creator ros_utils/reference_data_creator.cpp)
target_link_libraries(reference_data_creator
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
