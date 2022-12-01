cmake_minimum_required(VERSION 3.0.2)
project(place_recognition)

add_compile_options(-o2 -g -Wall)

find_package(catkin REQUIRED COMPONENTS
    cv_bridge
    eigen_conversions
    image_transport
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

# CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
endif()

catkin_package(
    INCLUDE_DIRS include
    ${OpenCV_INCLUDE_DIRS}
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

# keypoints_detector
add_executable(keypoints_detector src/keypoints_detector.cpp)
target_link_libraries(keypoints_detector
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
