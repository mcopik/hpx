# Copyright (c) 2016 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs)

if(HPX_WITH_SYCL)
  set(example_programs ${example_programs}
      vector_functionality
      stl_algorithms
    )

  set(vector_functionality_SYCL On)
  set(stl_algorithms_SYCL On)
endif()

foreach(example_program ${example_programs})

  source_group("Source Files" FILES ${sources})
  set(sources ${example_program}.cpp)
  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     ${${example_program}_FLAGS}
                     FOLDER "Examples/Compute/SYCL")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.compute.sycl.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.sycl
                              examples.compute.sycl.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.sycl.${example_program}
                              ${example_program}_exe)
endforeach()

