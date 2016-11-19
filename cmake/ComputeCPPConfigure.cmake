# Copyright (c) 2015 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(computecpp_configure_cxx)

  #ComputeCPP requires loading its own modules

endmacro()

macro(computecpp_configure)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HPX_WITH_COMPUTECPP}/cmake/Modules/")
  include(FindComputeCpp)
  hpx_add_config_define(HPX_WITH_SYCL)
  hpx_add_config_define(HPX_WITH_GPU_EXECUTOR)
  hpx_add_config_define(HPX_HAVE_SYCL)

  #temporary workaround to enable compilation
  hpx_add_compile_flag(-I${COMPUTECPP_INCLUDE_DIRECTORY})
  hpx_libraries(${COMPUTECPP_RUNTIME_LIBRARY} ${OPENCL_LIBRARIES})
  #other mallocs, i.e. tcmalloc or jemalloc caused by problems with Kalmar/HC
  #FIXME: revise whether it may be a problem for SYCL
  set(HPX_WITH_MALLOC custom CACHE STRING "" FORCE)
  #native TLS is not supported by Kalmar (cross-compilation to 32-bit code on GPUs)
  #FIXME: revise for SYCL
  set(HPX_WITH_NATIVE_TLS OFF CACHE BOOL "" FORCE)

endmacro()
