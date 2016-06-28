//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPUTE_MAY_10_2016_1244PM)
#define HPX_COMPUTE_MAY_10_2016_1244PM

#include <hpx/compute/host.hpp>
#include <hpx/compute/vector.hpp>
#include <hpx/compute/serialization/vector.hpp>

#if defined(HPX_HAVE_CUDA)
#include <hpx/compute/cuda.hpp>
#elif defined(HPX_HAVE_AMP)
#include <hpx/compute/amp.hpp>
#elif defined(HPX_HAVE_HC)
#include <hpx/compute/hc.hpp>
#endif

#endif

