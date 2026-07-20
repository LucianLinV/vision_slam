cmake_minimum_required(VERSION 3.10)
project(vslamp LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type: Debug, Release, RelWithDebInfo, MinSizeRel"
        FORCE
    )
endif()

set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres 2.2.0 EXACT REQUIRED CONFIG)
# find_package(Sophus 1.24.6 EXACT REQUIRED CONFIG)
# find_package(Ceres REQUIRED CONFIG)
find_package(Sophus REQUIRED CONFIG)
find_package(g2o REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS C)
find_package(PCL REQUIRED COMPONENTS common io)
add_definitions(${PCL_DEFINITIONS})

add_executable(vslamp
  main.cpp
  src/vmath.cpp
)

target_include_directories(vslamp PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(vslamp PRIVATE
  Eigen3::Eigen
  Ceres::ceres
  Sophus::Sophus
  g2o::core
  g2o::stuff
  g2o::solver_dense
  g2o::solver_eigen
  g2o::types_sba
  g2o::types_slam3d
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
