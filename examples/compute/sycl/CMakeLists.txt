# Copyright (c) 2016 Thomas Heller
# Copyright (c) 2016 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs)

if(HPX_WITH_SYCL)
  set(example_programs ${example_programs}
      vector_functionality
    )
endif()

foreach(example_program ${example_programs})

  source_group("Source Files" FILES ${sources})
  set(sources ${example_program}.cpp)
  # add example executable
  # FIXME: change after getting decent cmake support - multiple source
  #add_hpx_executable(${example_program}
  #                   SOURCES ${sources}
  #                   ${${example_program}_FLAGS}
  #                   FOLDER "Examples/Compute/SYCL")
  add_executable(${example_program} ${sources})
  hpx_setup_target(${example_program})
  add_sycl_to_target(${example_program} ${CMAKE_CURRENT_SOURCE_DIR}/${sources} ${CMAKE_CURRENT_BINARY_DIR})

  # add a custom target for this example
  add_hpx_pseudo_target(examples.compute.sycl.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.sycl
                              examples.compute.sycl.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.compute.sycl.${example_program}
                              ${example_program})
endforeach()

