# Copyright (c) 2015 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(kalmar_configure_cxx)

  execute_process(COMMAND find ${HPX_WITH_KALMAR} -name clang++ -print OUTPUT_VARIABLE KALMAR_CXX OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND find ${HPX_WITH_KALMAR} -name clamp-config -print OUTPUT_VARIABLE KALMAR_CONFIG OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${KALMAR_CONFIG} --build --cxxflags OUTPUT_VARIABLE KALMAR_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${KALMAR_CONFIG} --build --ldflags OUTPUT_VARIABLE KALMAR_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(CMAKE_CXX_COMPILER ${KALMAR_CXX})

endmacro()

macro(kalmar_configure)

  add_definitions(-DHPX_WITH_AMP)

  hpx_add_compile_flag(${KALMAR_CXX_FLAGS})
  #using hpx_add_link_flag will modify also static linking flags
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  #other mallocs, i.e. tcmalloc or jemalloc cause a segfault with Kalmar
  set(HPX_WITH_MALLOC "custom" CACHE STRING "" FORCE)
  #native TLS is not supported by Kalmar (cross-compilation to 32-bit code on GPUs)
  set(HPX_WITH_NATIVE_TLS OFF CACHE BOOL "" FORCE)
endmacro()
