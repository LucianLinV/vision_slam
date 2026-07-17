cmake_minimum_required(VERSION 3.10)
project(vslamp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
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
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
