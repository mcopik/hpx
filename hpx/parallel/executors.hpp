//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_MAY_11_2015_0728PM)
#define HPX_PARALLEL_EXECUTORS_MAY_11_2015_0728PM

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/executor_information_traits.hpp>
#include <hpx/parallel/executors/timed_executor_traits.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>

#include <hpx/parallel/executors/thread_executor_traits.hpp>
#include <hpx/parallel/executors/thread_executor_information_traits.hpp>
#include <hpx/parallel/executors/thread_timed_executor_traits.hpp>
#include <hpx/parallel/executors/thread_executor_parameter_traits.hpp>

#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/sequential_executor.hpp>
#if defined(HPX_WITH_AMP)
  #include <hpx/parallel/executors/gpu_amp_executor.hpp>
#endif
#if defined(HPX_WITH_SYCL)
  #include <hpx/parallel/executors/gpu_sycl_executor.hpp>
#endif
#include <hpx/parallel/executors/distribution_policy_executor.hpp>
#include <hpx/parallel/executors/service_executors.hpp>
#include <hpx/parallel/executors/this_thread_executors.hpp>
#include <hpx/parallel/executors/thread_pool_executors.hpp>
#include <hpx/parallel/executors/thread_pool_os_executors.hpp>
#include <hpx/parallel/executors/thread_pool_attached_executors.hpp>
#include <hpx/parallel/executors/default_executor.hpp>

#endif
