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
  #include(${HPX_WITH_COMPUTECPP}/cmake/common.cmake)
  hpx_add_config_define(HPX_WITH_SYCL)
  hpx_add_config_define(HPX_WITH_GPU_EXECUTOR)

  #temporary workaround to enable compilation
  #hpx_add_compile_flag(-I${HPX_WITH_COMPUTECPP}/include)
  hpx_add_compile_flag(-I${COMPUTECPP_INCLUDE_DIRECTORY})
  #using hpx_add_link_flag will modify also static linking flags
  #set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  #set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  #other mallocs, i.e. tcmalloc or jemalloc cause a segfault with Kalmar
  message(${HPX_WITH_MALLOC})
  set(HPX_WITH_MALLOC custom CACHE STRING "" FORCE)
  message(${HPX_WITH_MALLOC})
  #native TLS is not supported by Kalmar (cross-compilation to 32-bit code on GPUs)
  set(HPX_WITH_NATIVE_TLS OFF CACHE BOOL "" FORCE)

endmacro()
