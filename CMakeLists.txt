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

# keypoints_detector
add_executable(keypoints_detector src/keypoints_detector.cpp)
target_link_libraries(keypoints_detector
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# image_creator
add_executable(image_creator src/image_creator.cpp)
target_link_libraries(image_creator
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# reference_data_creator
add_executable(reference_data_creator src/reference_data_creator.cpp)
target_link_libraries(reference_data_creator
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

# dbow3_demo
add_executable(dbow3_demo src/dbow3_demo.cpp)
target_link_libraries(dbow3_demo
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    DBoW3
)

# images_searcher
add_executable(images_searcher src/images_searcher.cpp)
target_link_libraries(images_searcher
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    DBoW3
)