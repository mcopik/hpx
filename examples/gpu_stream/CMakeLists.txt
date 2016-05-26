# Copyright (c) 2016 Marcin Copik
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

add_hpx_executable(gpu_stream
  SOURCES gpu_stream.cpp
  FOLDER "Examples/gpu_stream")

#add_hpx_pseudo_target(examples.gpu_stream.gpu_stream)
#add_hpx_pseudo_dependencies(examples.gpu_stream examples.queue.queue_client)
#add_hpx_pseudo_dependencies(examples.gpu_stream.gpu_stream gpu_stream_exe)
