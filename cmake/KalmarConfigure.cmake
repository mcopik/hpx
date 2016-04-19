# Copyright (c) 2015 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(kalmar_configure_cxx)

  if(HPX_WITH_KALMAR)
	set(compiler_directory ${HPX_WITH_KALMAR})
	set(config_app "clamp-config")
  elseif(HPX_WITH_HCC)
	set(compiler_directory ${HPX_WITH_HCC})
	set(config_app "hcc-config")
  else()
        set(compiler_directory ${HPX_WITH_HCC_STDCXX})
        set(config_app "hcc-config")
  endif()

  execute_process(COMMAND find ${compiler_directory} -name clang++ -print COMMAND head -n 1 OUTPUT_VARIABLE KALMAR_CXX OUTPUT_STRIP_TRAILING_WHITESPACE)  
  execute_process(COMMAND find ${compiler_directory} -name ${config_app} -print COMMAND head -n 1 OUTPUT_VARIABLE KALMAR_CONFIG OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${KALMAR_CONFIG} --build --cxxflags OUTPUT_VARIABLE KALMAR_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${KALMAR_CONFIG} --build --ldflags OUTPUT_VARIABLE KALMAR_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  if("${KALMAR_CXX}" STREQUAL "")
	message(FATAL_ERROR "Clang compiler in Kalmar directory could not be found!")
  else()
	message("${KALMAR_CXX}")
  endif()
  set(CMAKE_CXX_COMPILER "${KALMAR_CXX}")
  if("${KALMAR_CONFIG}" STREQUAL "")
        message(FATAL_ERROR "clamp-config in Kalmar directory could not be found!")
  endif()

  if(HPX_WITH_HCC_STDCXX)
        string(REPLACE "-hc" "" KALMAR_CXX_FLAGS ${KALMAR_CXX_FLAGS})
        string(REPLACE "-std=c++amp" "" KALMAR_CXX_FLAGS ${KALMAR_CXX_FLAGS})
        string(REPLACE "-hc" "" KALMAR_LD_FLAGS ${KALMAR_LD_FLAGS})
        string(REPLACE "-std=c++amp" "" KALMAR_LD_FLAGS ${KALMAR_LD_FLAGS})
        string(REPLACE "-lmcwamp" "" KALMAR_LD_FLAGS ${KALMAR_LD_FLAGS})
  endif()

  set(CMAKE_CXX_COMPILER ${KALMAR_CXX})
  set(HPX_WITH_NATIVE_TLS OFF CACHE BOOL "" FORCE)
endmacro()

macro(kalmar_configure)

  if(NOT HPX_WITH_HCC_STDCXX)
        add_definitions(-DHPX_WITH_AMP)
  endif()

  hpx_add_compile_flag("${KALMAR_CXX_FLAGS}")
  #using hpx_add_link_flag will modify also static linking flags
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${KALMAR_LD_FLAGS}")
  #other mallocs, i.e. tcmalloc or jemalloc cause a segfault with Kalmar
  set(HPX_WITH_MALLOC "custom" CACHE STRING "" FORCE)
  #native TLS is not supported by Kalmar (cross-compilation to 32-bit code on GPUs)
endmacro()
