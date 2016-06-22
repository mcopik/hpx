# Copyright (c) 2015-2016 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(hcc_configure_cxx)

    if(HPX_WITH_HCC)
        set(compiler_directory ${HPX_WITH_HCC})
        set(config_app "clamp-config")
    endif()

    execute_process(
                COMMAND find ${compiler_directory} -name clang++ -print
                COMMAND head -n 1 OUTPUT_VARIABLE HCC_CXX
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    execute_process(
                COMMAND find ${compiler_directory} -name ${config_app} -print
                COMMAND head -n 1 OUTPUT_VARIABLE HCC_CONFIG
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    execute_process(
                COMMAND ${HCC_CONFIG} --cxxflags OUTPUT_VARIABLE HCC_CXX_FLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    execute_process(
                COMMAND ${HCC_CONFIG} --ldflags OUTPUT_VARIABLE HCC_LD_FLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

    if("${HCC_CXX}" STREQUAL "")
        message(FATAL_ERROR "Clang compiler in HCC directory could not be found!")
    endif()
    if("${HCC_CONFIG}" STREQUAL "")
        message(FATAL_ERROR "${config_app} in HCC directory could not be found!")
    endif()

    set(CMAKE_CXX_COMPILER ${HCC_CXX})
    set(HPX_WITH_NATIVE_TLS OFF CACHE BOOL "" FORCE)
endmacro()

macro(hcc_configure)

    set(HPX_WITH_COMPUTE On)
    hpx_add_config_define(HPX_HAVE_AMP)
    hpx_add_config_define(HPX_HAVE_COMPUTE)

    hpx_add_compile_flag("${HCC_CXX_FLAGS}")
    #using hpx_add_link_flag will modify also static linking flags
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HCC_LD_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${HCC_LD_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${HCC_LD_FLAGS}")
    #other mallocs, i.e. tcmalloc or jemalloc cause a segfault with HCC
    set(HPX_WITH_MALLOC "custom" CACHE STRING "" FORCE)
    #native TLS is not supported by Kalmar (cross-compilation to 32-bit code on GPUs)
endmacro()
