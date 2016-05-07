cmake_minimum_required(VERSION 2.8)
project(sycl_bug)

set(CMAKE_PREFIX_PATH ${HPX_LOCATION} ${CMAKE_PREFIX_PATH})
find_package(HPX REQUIRED)
#get_target_property(COMPILE_OPTIONS hpx INTERFACE_COMPILE_OPTIONS)
#get_target_property(COMPILE_DEFINITIOMS hpx INTERFACE_COMPILE_DEFINITIONS)
#message("${INTERFACE_COMPILE_DEFINITIONS}")
#message("${INTERFACE_COMPILE_OPTIONS}")
#message("${HPX_INCLUDE_DIRS}")
#message("${HPX_LIBRARIES}")
include_directories(${HPX_INCLUDE_DIRS})


set(CMAKE_MODULE_PATH ${COMPUTECPP_LOCATION}/cmake/Modules/ ${CMAKE_MODULE_PATH})
include(${COMPUTECPP_LOCATION}/cmake/common.cmake)
set(DEVICE_COMPILER_PATH "${COMPUTECPP_LOCATION}/bin/")
set(DEVICE_COMPILER_INCLUDE_PATH "${COMPUTECPP_LOCATION}/include/device_compiler")
set(DEVICE_COMPILER_NAME "compute++")
add_custom_target(compiler)
set(DEVICE_COMPILER "${DEVICE_COMPILER_PATH}/${DEVICE_COMPILER_NAME}" CACHE FILEPATH "device compiler")
set(RUNTIME_INCLUDES "${COMPUTECPP_LOCATION}/include")
include_directories(${RUNTIME_INCLUDES})

#add_custom_target(${SYCL_LIB_NAME})
set(SYCL_LIB_PATH "${COMPUTECPP_LOCATION}/lib/${SYCL_LIB_FILE}")

build_sycl("stream" "SPIR" "stream" ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
#build_sycl("executor_test_no_hpx" "SPIR" "executor_test_no_hpx" ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
#build_sycl("async_execution" "SPIR" "async_execution" ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
#build_sycl("async_execution_no_hpx" "SPIR" "async_execution_no_hpx" ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
set(stream_FLAGS DEPENDENCIES iostreams_component)
hpx_setup_target(stream)
#add_hpx_executable(stream
#                     SOURCES stream.cpp
##                     EXCLUDE_FROM_ALL
#                     HPX_PREFIX ${HPX_BUILD_PREFIX}
#                     FOLDER "Benchmarks/Local/${benchmark}")
#hpx_setup_target(executor_test_no_hpx)
#hpx_setup_target(async_execution)
#hpx_setup_target(async_execution_no_hpx)
