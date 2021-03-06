# Copyright (c) 2007-2015 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Dylan Stark
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
    1d_hydro
    1d_stencil
    accumulators
    allgather
    async_io
    balancing
    cancelable_action
    future_reduce
    heartbeat
    interpolate1d
    jacobi
    jacobi_smp
    nqueen
    performance_counters
    quickstart
    qt
    random_mem_access
    spell_check
    startup_shutdown
    thread_aware_timer
    throttle
    tuplespace
    transpose
   )

if(HPX_WITH_QUEUE_COMPATIBILITY)
  set(subdirs ${subdirs}
      queue
     )
endif()

if(HPX_WITH_FORTRAN)
  set(subdirs ${subdirs}
    sheneos
    )
endif()

if(HPX_WITH_APEX)
  set(subdirs ${subdirs}
      apex
     )
endif()

if(HPX_WITH_COMPUTE)
  set(subdirs ${subdirs}
      compute
     )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.${subdir})
endforeach()

