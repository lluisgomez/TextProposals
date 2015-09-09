cmake_minimum_required(VERSION 2.8)
project(opencv_sandbox)

# Select a default build configuration if none was chosen
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(img2hierarchy fast_clustering.cpp main.cpp agglomerative_clustering.cpp region.cpp min_bounding_box.cpp)

FIND_PACKAGE(OpenCV REQUIRED)
IF(OpenCV_FOUND)
  TARGET_LINK_LIBRARIES(img2hierarchy ${OpenCV_LIBS})
ENDIF()
