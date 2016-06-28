# Copyright (c) 2016 Hartmut Kaiser
# Copyright (c) 2016 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs
   )

if(HPX_WITH_CUDA)
  set(subdirs ${subdirs}
      cuda
     )
endif()

if(HPX_WITH_COMPUTE_AMP)
    set(subdirs ${subdirs}
        amp
        )
endif()

if(HPX_WITH_COMPUTE_HC)
    set(subdirs ${subdirs}
        hc
        )
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(examples.compute.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(examples examples.compute.${subdir})
endforeach()

