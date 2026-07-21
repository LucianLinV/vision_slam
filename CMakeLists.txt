cmake_minimum_required(VERSION 3.18)

project(vslamp LANGUAGES C CXX)

# ============================================================
# Build type
# ============================================================

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(
        CMAKE_BUILD_TYPE Release
        CACHE STRING "Build type: Debug, Release, RelWithDebInfo, MinSizeRel"
        FORCE
    )
endif()

set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
        Debug
        Release
        RelWithDebInfo
        MinSizeRel
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================
# C++ standard
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Basic dependencies
# ============================================================

find_package(Eigen3 REQUIRED)

set(VSLAMP_OPENCV_COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
    features2d
    calib3d
)

find_package(OpenCV REQUIRED COMPONENTS ${VSLAMP_OPENCV_COMPONENTS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# ============================================================
# Ceres: prefer CUDA build, otherwise use CPU build
# ============================================================

# 常见的用户安装前缀。
set(_ceres_scan_prefixes
    "$ENV{HOME}/.local"
    "/usr/local"
)

# 自动匹配类似：
#
# ~/.local/ceres-2.2.0
# ~/.local/ceres-2.2.0-cuda
# ~/.local/ceres-2.3.0
file(
    GLOB _ceres_user_prefixes
    LIST_DIRECTORIES true
    "$ENV{HOME}/.local/ceres*"
)

list(APPEND
    _ceres_scan_prefixes
    ${_ceres_user_prefixes}
)

list(REMOVE_DUPLICATES _ceres_scan_prefixes)

set(_ceres_gpu_prefixes)
set(_ceres_cpu_prefixes)

foreach(_prefix IN LISTS _ceres_scan_prefixes)
    if(NOT IS_DIRECTORY "${_prefix}")
        continue()
    endif()

    # 判断当前前缀中是否存在 CeresConfig.cmake。
    file(
        GLOB _ceres_config_files
        "${_prefix}/lib/cmake/Ceres/CeresConfig.cmake"
        "${_prefix}/lib64/cmake/Ceres/CeresConfig.cmake"
        "${_prefix}/lib/*/cmake/Ceres/CeresConfig.cmake"
    )

    list(LENGTH _ceres_config_files _ceres_config_count)

    if(_ceres_config_count EQUAL 0)
        continue()
    endif()

    set(
        _ceres_config_header
        "${_prefix}/include/ceres/internal/config.h"
    )

    if(NOT EXISTS "${_ceres_config_header}")
        continue()
    endif()

    # 只匹配完整的 CERES_NO_CUDA 宏。
    # 不会误匹配 CERES_NO_CUDA_SPARSE 等其他宏。
    file(
        STRINGS
        "${_ceres_config_header}"
        _ceres_no_cuda_lines
        REGEX
        "^[ \t]*#[ \t]*define[ \t]+CERES_NO_CUDA([ \t]+1)?[ \t]*$"
    )

    if(_ceres_no_cuda_lines)
        list(APPEND _ceres_cpu_prefixes "${_prefix}")
    else()
        list(APPEND _ceres_gpu_prefixes "${_prefix}")
    endif()
endforeach()

# 搜索优先级：
#
# 1. 用户安装的 CUDA Ceres
# 2. 用户安装的 CPU Ceres
# 3. 外部 CMAKE_PREFIX_PATH
# 4. /usr、/usr/local 等系统默认位置
set(
    CMAKE_PREFIX_PATH
    ${_ceres_gpu_prefixes}
    ${_ceres_cpu_prefixes}
    ${CMAKE_PREFIX_PATH}
)

# 至少使用 Ceres 2.2。
# 不使用 EXACT，允许使用兼容的新版 Ceres 2.x。
find_package(Ceres 2.2 REQUIRED CONFIG)

# ============================================================
# Detect selected Ceres backend
# ============================================================

get_target_property(
    _ceres_include_dirs
    Ceres::ceres
    INTERFACE_INCLUDE_DIRECTORIES
)

set(VSLAMP_CERES_HAS_CUDA FALSE)
set(_ceres_backend_detected FALSE)
set(_selected_ceres_config_header "")

foreach(_include_dir IN LISTS _ceres_include_dirs)
    # 跳过 CMake 生成器表达式。
    if(_include_dir MATCHES "^\\$<")
        continue()
    endif()

    set(
        _config_header
        "${_include_dir}/ceres/internal/config.h"
    )

    if(NOT EXISTS "${_config_header}")
        continue()
    endif()

    # 精确检查是否定义了 CERES_NO_CUDA。
    file(
        STRINGS
        "${_config_header}"
        _selected_no_cuda_lines
        REGEX
        "^[ \t]*#[ \t]*define[ \t]+CERES_NO_CUDA([ \t]+1)?[ \t]*$"
    )

    set(_ceres_backend_detected TRUE)
    set(_selected_ceres_config_header "${_config_header}")

    if(NOT _selected_no_cuda_lines)
        set(VSLAMP_CERES_HAS_CUDA TRUE)
    endif()

    break()
endforeach()

message(STATUS "Ceres config directory: ${Ceres_DIR}")

if(DEFINED Ceres_VERSION)
    message(STATUS "Ceres version: ${Ceres_VERSION}")
endif()

if(NOT _ceres_backend_detected)
    message(WARNING
        "Unable to determine whether the selected Ceres has CUDA support"
    )
elseif(VSLAMP_CERES_HAS_CUDA)
    message(STATUS "Ceres backend: CUDA")
    message(STATUS
        "Ceres config header: ${_selected_ceres_config_header}"
    )

    if(DEFINED CUDAToolkit_VERSION)
        message(STATUS
            "CUDA Toolkit version: ${CUDAToolkit_VERSION}"
        )
    endif()
else()
    message(STATUS "Ceres backend: CPU")
    message(STATUS
        "Ceres config header: ${_selected_ceres_config_header}"
    )
endif()

# ============================================================
# Other dependencies
# ============================================================

#find_package(Sophus 1.24.6 EXACT REQUIRED CONFIG)
find_package(Sophus REQUIRED CONFIG)
find_package(g2o REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS C)
find_package(PCL REQUIRED COMPONENTS common io)

add_definitions(${PCL_DEFINITIONS})

include(CTest)

# ============================================================
# Project library and executables
# ============================================================

add_library(vslamp_core
    src/vmath.cpp
)

target_include_directories(vslamp_core PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(vslamp_core PUBLIC
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

# 将检测结果传递给 C++。
if(VSLAMP_CERES_HAS_CUDA)
    target_compile_definitions(
        vslamp_core PUBLIC
        VSLAMP_CERES_HAS_CUDA=1
    )
else()
    target_compile_definitions(
        vslamp_core PUBLIC
        VSLAMP_CERES_HAS_CUDA=0
    )
endif()

add_executable(vslamp
    main.cpp
)

target_link_libraries(vslamp PRIVATE
    vslamp_core
)

function(vslamp_add_test_executable test_source)
    get_filename_component(test_name "${test_source}" NAME_WE)

    add_executable("${test_name}"
        "${test_source}"
    )

    target_link_libraries("${test_name}" PRIVATE
        vslamp_core
    )

    if(BUILD_TESTING)
        add_test(NAME "${test_name}" COMMAND "${test_name}")
    endif()
endfunction()

file(GLOB VSLAMP_TEST_SOURCES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/test/*.cpp"
)

foreach(_test_source IN LISTS VSLAMP_TEST_SOURCES)
    vslamp_add_test_executable("${_test_source}")
endforeach()
